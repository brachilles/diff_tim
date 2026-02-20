# graph_guidance.py
# Selective graph-guided smoother for CSDI time-series diffusion
# + PAIRS-ONLY guidance (topology-free) with correlation-weighted spill (dynamic or CSV).
#
# GRAPH path:
#   - Spill modes: "uniform", "adj", "heat" (default: "heat")
#   - prox_floor / prox_gamma to shape proximity for unguided
#   - delta broadcast via heat kernel
#
# PAIRS-ONLY path (topology-free):
#   - Only the listed pairs are nudged (symmetric pull)
#   - Optional uniform broadcast of guided deltas to ALL features
#   - broadcast_mode: "mean" | "sum" | "sumabs" | "rms"
#   - broadcast_gain: extra multiplier on the aggregated Δ
#   - unguided_spill_* — topology-free or correlation-weighted nudge guided → unguided
#   - NEGATIVE pair weights allowed (anti-align).
#
# New in this version:
#   * unguided_spill_mode supports "corr" (signed, correlation-weighted spill)
#   * You can supply:
#       - a dynamic correlation computed from the current history (corr_dynamic=True), or
#       - a static correlation CSV (corr_csv=...).
#   * Order-invariant pair updates (uses clones).

from __future__ import annotations
import numpy as np
import torch
import pandas as pd
from typing import List, Tuple, Optional

DEBUG_GUIDANCE = True


def _to_tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)


# =========================
# GRAPH-BASED GUIDANCE
# =========================
def load_adjacency(csv_path: str, fmt: str = "dense", D: int | None = None) -> np.ndarray:
    if fmt == "dense":
        A = pd.read_csv(csv_path, header=None).values.astype(np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"[graph] dense csv must be square, got {A.shape}")
        np.fill_diagonal(A, 0.0)
        A = np.maximum(A, 0.0)
        A = 0.5 * (A + A.T)
    elif fmt == "edges":
        if D is None:
            raise ValueError("fmt='edges' requires D (#features)")
        M = pd.read_csv(csv_path, header=None).values
        A = np.zeros((D, D), dtype=np.float64)
        for row in M:
            if len(row) < 3:
                continue
            try:
                i, j, w = int(row[0]), int(row[1]), float(row[2])
            except Exception:
                continue
            if i == j or i < 0 or j < 0 or i >= D or j >= D:
                continue
            w = max(0.0, w)
            A[i, j] = A[j, i] = w
    else:
        raise ValueError(f"Unknown format {fmt}")

    if (A > 0).any():
        p95 = np.percentile(A[A > 0], 95)
        scale = max(p95, 1e-6)
        A = np.clip(A / scale, 0.0, 1.0)

    deg = A.sum(axis=1) + 1e-12
    Dmh = 1.0 / np.sqrt(deg)
    A = (Dmh[:, None] * A) * Dmh[None, :]
    np.fill_diagonal(A, 0.0)
    return A


class HeatKernelSmoother:
    def __init__(self, A_np: np.ndarray):
        if A_np.shape[0] != A_np.shape[1]:
            raise ValueError("Adjacency must be square.")
        A_np = np.maximum(A_np, 0.0)
        A_np = 0.5 * (A_np + A_np.T)
        np.fill_diagonal(A_np, 0.0)

        deg = A_np.sum(axis=1) + 1e-12
        Dmh = np.diag(1.0 / np.sqrt(deg))
        L = np.eye(A_np.shape[0]) - (Dmh @ A_np @ Dmh)  # normalized Laplacian
        evals, evecs = np.linalg.eigh(L.astype(np.float64))
        self.evals = evals.astype(np.float32)
        self.evecs = evecs.astype(np.float32)
        self._cache = {}

    def _get_kernel(self, tau: float, device: torch.device):
        key = (str(device), float(tau))
        if key in self._cache:
            return self._cache[key]
        U = _to_tensor(self.evecs, device)
        lam = _to_tensor(self.evals, device)
        exp_diag = torch.exp(-float(tau) * lam)
        H = (U * exp_diag.unsqueeze(0)) @ U.t()  # e^{-tau L}
        self._cache[key] = H
        return H


@torch.no_grad()
def tie_guided_to_anchors(x, cond_mask, A_np, guided_idx, weight=0.5):
    if weight <= 0:
        return x
    B, K, L = x.shape
    out = x.clone()
    A = torch.as_tensor(A_np, device=x.device, dtype=x.dtype)
    g = guided_idx.to(device=x.device, dtype=torch.bool)
    a = ~g
    if a.sum() == 0:
        return x
    A_anchor = A[:, a]
    denom = A_anchor.sum(dim=1) + 1e-8

    for t in range(L):
        fut = (cond_mask[:, :, t] < 0.5).float()
        if not fut.any():
            continue
        X = out[:, :, t]
        Xa = X[:, a]
        m_all = (Xa @ A_anchor.T) / denom.unsqueeze(0)
        wmask = fut * g.float()
        X = torch.where(wmask > 0, (1 - weight) * X + weight * m_all, X)
        out[:, :, t] = X
    return out


def build_cond_fn(
    graph_csv: str,
    graph_format: str,
    num_features: int,
    lambda_g: float = 0.5,
    eta: float = 0.5,
    tau: float = 2.0,
    schedule: str = "late90",
    snr_gate: float = 0.0,
    guided_features: List[int] | None = None,
    num_steps: int = 50,
    alpha_series: np.ndarray | None = None,
    passes: int = 2,
    ramp_last: float = 0.5,
    enable_pair_tie: bool = True,
    pair_tie_weight: float = 0.2,
    spill: float = 0.30,
    spill_mode: str = "heat",
    prox_floor: float = 0.20,
    prox_gamma: float = 0.80,
    broadcast_w: float = 0.0,
    broadcast_hops: int = 2,
    broadcast_clip: float = 3.0,
):
    A = load_adjacency(
        graph_csv,
        fmt=graph_format,
        D=num_features if graph_format == "edges" else None,
    )
    smoother = HeatKernelSmoother(A)
    A_torch = torch.as_tensor(A, dtype=torch.float32)

    lam_eff = float(np.clip(lambda_g, 0.0, 1.0))
    tau_eff = float(max(1e-4, eta * tau))

    guided_mask_np = None
    if guided_features:
        guided_mask_np = np.zeros((num_features,), dtype=bool)
        guided_mask_np[np.array(guided_features, dtype=int)] = True

    snr = None
    if alpha_series is not None:
        alpha_series = np.asarray(alpha_series, dtype=np.float64)
        snr = alpha_series / np.maximum(1e-12, (1.0 - alpha_series))

    ramp_last = float(np.clip(ramp_last, 0.0, 1.0))
    r0 = int((1.0 - ramp_last) * num_steps)

    def lam_t(t: int) -> float:
        t = int(t)
        r = num_steps - 1 - t
        if schedule == "late50" and r < int(0.5 * num_steps):
            return 0.0
        if schedule == "late90" and r < int(0.9 * num_steps):
            return 0.0
        if snr is not None and snr_gate > 0:
            if snr[min(max(t, 0), num_steps - 1)] < snr_gate:
                return 0.0
        if r < r0:
            return 0.0 if ramp_last > 0 else lam_eff
        u = (r - r0) / max(1, (num_steps - 1 - r0))
        return lam_eff * 0.5 * (1 - np.cos(np.pi * u))

    @torch.no_grad()
    def cond_fn(x: torch.Tensor, cond_mask: torch.Tensor, t: int) -> torch.Tensor:
        lt = lam_t(int(t))
        if lt <= 0:
            return x

        B, K, L = x.shape
        device = x.device
        H = smoother._get_kernel(tau_eff, device)

        if guided_mask_np is None:
            g = torch.ones(K, device=device, dtype=torch.bool)
        else:
            g = torch.from_numpy(guided_mask_np).to(device=device)
        gF = g.float()

        if spill_mode == "uniform":
            prox = torch.ones(K, device=device)
        elif spill_mode == "heat":
            prox = (H @ gF)
        elif spill_mode == "adj":
            prox = (A_torch.to(device) @ gF)
        else:
            raise ValueError(f"Unknown spill_mode: {spill_mode}")

        if prox.max() > 0:
            prox = prox / prox.max()
        else:
            prox = torch.zeros_like(prox)
        if prox_gamma != 1.0:
            prox = torch.clamp(prox, 0, 1) ** float(prox_gamma)
        if prox_floor > 0:
            prox = prox * (1.0 - float(prox_floor)) + float(prox_floor)

        alpha_feat = lt * (gF + (1.0 - gF) * float(spill) * prox)

        out = x.clone()
        for t_idx in range(L):
            fut = (cond_mask[:, :, t_idx] < 0.5).float()
            if not fut.any():
                continue

            X0 = out[:, :, t_idx]
            X = X0

            for _ in range(int(max(1, passes))):
                Y = (H @ X.T).T
                a = (alpha_feat.unsqueeze(0) * fut).to(X.dtype)
                X = (1.0 - a) * X + a * Y

            X_sm = X
            if enable_pair_tie and pair_tie_weight > 0 and guided_mask_np is not None:
                tie_w = float(pair_tie_weight) * float(lt / max(lam_eff, 1e-8))
                cm = cond_mask.clone()
                cm[:, :, :] = 0.0
                cm[:, :, t_idx] = fut
                X_sm = tie_guided_to_anchors(
                    X_sm.unsqueeze(2), cm.unsqueeze(2), A, g, weight=tie_w
                ).squeeze(2)

            if broadcast_w > 0:
                d_g = (X_sm - X0) * gF.unsqueeze(0) * fut
                if broadcast_clip > 0:
                    med = d_g.median(dim=1, keepdim=True).values
                    mad = (d_g - med).abs().median(dim=1, keepdim=True).values + 1e-6
                    d_g = torch.clamp(d_g, min=med - broadcast_clip * mad, max=med + broadcast_clip * mad)

                Dprop = d_g
                for _ in range(int(max(1, broadcast_hops))):
                    Dprop = (H @ Dprop.T).T

                X_sm = X_sm + float(broadcast_w) * Dprop

            out[:, :, t_idx] = X_sm

        return out

    if DEBUG_GUIDANCE:
        guided_list = (
            np.where(guided_mask_np)[0].tolist() if guided_mask_np is not None else "ALL"
        )
        print(
            f"[guidance] selective+spill+broadcast | guided={guided_list} | schedule={schedule} "
            f"| lam_eff={lam_eff:.3f} | tau_eff={tau_eff:.3f} | passes={passes} "
            f"| spill={spill:.2f} | spill_mode={spill_mode} "
            f"| prox_floor={prox_floor:.2f} | prox_gamma={prox_gamma:.2f} "
            f"| broadcast_w={broadcast_w:.2f} | hops={broadcast_hops} | clip={broadcast_clip:.1f} "
            f"| pair_tie={'ON' if enable_pair_tie and pair_tie_weight>0 else 'OFF'}"
        )

    return cond_fn


# =========================
# PAIRS-ONLY GUIDANCE (topology-free) + CORR SPILL
# =========================
def build_cond_fn_pairs_only(
    pairs: List[Tuple[int, int, float]],   # [(i, j, w_ij)], 0-based; w in [-1,1] allowed
    num_features: int,
    lambda_g: float = 0.5,
    schedule: str = "late90",              # "always" | "late50" | "late90"
    num_steps: int = 50,
    snr_gate: float = 0.0,
    alpha_series: np.ndarray | None = None,
    # Uniform broadcast (topology-free)
    broadcast_w: float = 0.0,              # 0 disables
    broadcast_clip: float = 3.0,           # robust clip of guided deltas (per batch/time)
    broadcast_mode: str = "mean",          # "mean" | "sum" | "sumabs" | "rms"
    broadcast_gain: float = 1.0,           # extra multiplier on aggregated delta
    broadcast_exclude_guided: bool = False,# if True, broadcast only to unguided
    broadcast_center: bool = False,        # if True, center d_agg to avoid DC offsets
    # Spill
    unguided_spill_alpha: float = 0.0,     # 0 disables; try 0.10–0.30
    unguided_spill_mode: str = "delta",    # "delta" | "value" | "corr"
    unguided_spill_center: bool = True,    # center the aggregate before applying
    # Correlation-weighted spill config:
    corr_csv: Optional[str] = None,        # path to KxK correlation CSV (optional)
    corr_dynamic: bool = False,            # if True, compute from current history window (no CSV)
    corr_abs: bool = False,                # if True, use |corr| (drop sign)
    corr_power: float = 1.0,               # raise |corr| to this power (e.g., 1.0–2.0)
    corr_norm: str = "colmax",             # "colmax" | "l1" | "none"
):
    """
    Pair-only guidance:
      - No graph, no proximity.
      - Only the listed pairs are directly nudged (negatives allowed).
      - Optional uniform broadcast of aggregated guided deltas.
      - Optional unguided_spill:
           * "delta"/"value": topology-free baseline
           * "corr": correlation-weighted, signed propagation
             - Either from corr_csv (static) or corr_dynamic (computed from current history).
    """
    lam_eff = float(np.clip(lambda_g, 0.0, 1.0))

    # schedule
    snr = None
    if alpha_series is not None:
        alpha_series = np.asarray(alpha_series, dtype=np.float64)
        snr = alpha_series / np.maximum(1e-12, (1.0 - alpha_series))

    def lam_t(t: int) -> float:
        t = int(t)
        r = num_steps - 1 - t
        if schedule == "late50" and r < int(0.5 * num_steps):
            return 0.0
        if schedule == "late90" and r < int(0.9 * num_steps):
            return 0.0
        if snr is not None and snr_gate > 0:
            if snr[min(max(t, 0), num_steps - 1)] < snr_gate:
                return 0.0
        ramp_start = 0
        if schedule in ("late50", "late90"):
            frac = 0.5 if schedule == "late50" else 0.9
            ramp_start = int(frac * (num_steps - 1))
        if r < ramp_start:
            return 0.0
        u = (r - ramp_start) / max(1, (num_steps - 1 - ramp_start))
        return lam_eff * 0.5 * (1 - np.cos(np.pi * u))

    # sanitize pairs
    clean_pairs: List[Tuple[int, int, float]] = []
    guided_set = set()
    for (i, j, w) in pairs or []:
        if 0 <= int(i) < num_features and 0 <= int(j) < num_features and int(i) != int(j):
            ww = float(np.clip(float(w), -1.0, 1.0))  # allow NEGATIVE (anti-align)
            if ww == 0.0:
                continue
            clean_pairs.append((int(i), int(j), ww))
            guided_set.add(int(i)); guided_set.add(int(j))

    if len(clean_pairs) == 0:
        print("[pairs-only] WARNING: no valid pairs; cond_fn will be a no-op.")

    guided_mask = torch.zeros(num_features, dtype=torch.float32)
    if len(guided_set) > 0:
        guided_mask[list(guided_set)] = 1.0

    # ---- correlation setup (static) ----
    C_static_np = None
    if unguided_spill_mode.lower() == "corr" and (not corr_dynamic) and corr_csv:
        C_static_np = pd.read_csv(corr_csv, header=None).values.astype(np.float32)
        if C_static_np.shape != (num_features, num_features):
            raise ValueError(f"corr_csv shape {C_static_np.shape} != ({num_features},{num_features})")
        np.fill_diagonal(C_static_np, 0.0)
        if corr_abs:
            C_static_np = np.abs(C_static_np)
        if corr_power != 1.0:
            C_static_np = np.sign(C_static_np) * (np.abs(C_static_np) ** float(corr_power))
        if corr_norm == "colmax":
            s = np.maximum(1e-6, np.max(np.abs(C_static_np), axis=0, keepdims=True))
            C_static_np = C_static_np / s
        elif corr_norm == "l1":
            s = np.maximum(1e-6, np.sum(np.abs(C_static_np), axis=0, keepdims=True))
            C_static_np = C_static_np / s
        elif corr_norm == "none":
            pass
        else:
            raise ValueError("corr_norm must be 'colmax', 'l1', or 'none'")

    @torch.no_grad()
    def _corr_from_history(x: torch.Tensor, cond_mask: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Build KxK correlation from observed history (cond_mask>0.5) of the current batch.
        Returns torch.Tensor [K,K] on x.device, or None if not enough data.
        """
        B, K, L = x.shape
        M = (cond_mask > 0.5).float()  # 1 on history, 0 on future
        feats = []
        for k in range(K):
            vals = x[:, k, :][M[:, k, :] > 0.5]
            if vals.numel() < 4:
                return None  # not enough data
            feats.append(vals[:vals.numel()])
        Lmin = min(v.numel() for v in feats)
        if Lmin < 4:
            return None
        A = torch.stack([v[:Lmin] for v in feats], dim=0)  # [K,Lmin]
        A = (A - A.mean(dim=1, keepdim=True)) / (A.std(dim=1, keepdim=True) + 1e-6)
        C = (A @ A.t()) / float(Lmin)  # [K,K]
        C.fill_diagonal_(0.0)
        if corr_abs:
            C = C.abs()
        if corr_power != 1.0:
            C = torch.sign(C) * (C.abs() ** float(corr_power))
        if corr_norm == "colmax":
            s = torch.clamp(C.abs().max(dim=0, keepdim=True).values, min=1e-6)
            C = C / s
        elif corr_norm == "l1":
            s = torch.clamp(C.abs().sum(dim=0, keepdim=True), min=1e-6)
            C = C / s
        elif corr_norm == "none":
            pass
        else:
            raise ValueError("corr_norm must be 'colmax', 'l1', or 'none'")
        return C

    # cache for dynamic corr per batch (lazily computed once)
    C_cache: Optional[torch.Tensor] = None

    @torch.no_grad()
    def cond_fn(x: torch.Tensor, cond_mask: torch.Tensor, t: int) -> torch.Tensor:
        nonlocal C_cache
        lt = lam_t(int(t))
        if lt <= 0 or len(clean_pairs) == 0:
            return x

        B, K, L = x.shape
        out = x.clone()
        gF = guided_mask.to(device=x.device, dtype=torch.float32)

        # pick correlation mapping if needed
        use_corr = (unguided_spill_alpha > 0) and (unguided_spill_mode.lower() == "corr")
        W = None  # [K, |G|]
        if use_corr:
            G_idx = torch.nonzero(gF > 0.5, as_tuple=False).flatten()
            if G_idx.numel() > 0:
                if corr_dynamic:
                    if C_cache is None:
                        C_dyn = _corr_from_history(out, cond_mask)
                        if C_dyn is not None:
                            C_cache = C_dyn.to(device=out.device, dtype=out.dtype)
                    C_mat = C_cache
                else:
                    C_mat = torch.as_tensor(C_static_np, device=out.device, dtype=out.dtype) if C_static_np is not None else None
                if C_mat is not None:
                    W = C_mat[:, G_idx]  # [K, |G|]

        for t_idx in range(L):
            fut = (cond_mask[:, :, t_idx] < 0.5)  # (B,K) bool
            if not fut.any():
                continue

            X = out[:, :, t_idx]
            X0 = X.clone()

            # --- symmetric pair pulls (order-invariant) ---
            for (i, j, w) in clean_pairs:
                beta = float(lt) * float(w)
                if beta == 0.0:
                    continue
                m_i = fut[:, i].float().unsqueeze(1)
                m_j = fut[:, j].float().unsqueeze(1)
                Xi0 = X[:, i:i+1].clone()
                Xj0 = X[:, j:j+1].clone()
                Xi_new = Xi0 + m_i * (beta * (Xj0 - Xi0))
                Xj_new = Xj0 + m_j * (beta * (Xi0 - Xj0))
                X[:, i:i+1] = Xi_new
                X[:, j:j+1] = Xj_new

            # --- spill guided → unguided ---
            if unguided_spill_alpha > 0:
                g_mask = gF.unsqueeze(0) * fut.float()        # (B,K)
                ug_mask = (1.0 - gF).unsqueeze(0) * fut.float()
                mode = unguided_spill_mode.lower()

                if mode == "delta":
                    guided_delta = (X - X0) * g_mask
                    agg = guided_delta.mean(dim=1, keepdim=True)  # (B,1)
                    if unguided_spill_center:
                        agg = agg - agg.median(dim=0, keepdim=True).values
                    X = X + float(unguided_spill_alpha) * (agg * ug_mask)

                elif mode == "value":
                    guided_vals = X * g_mask
                    denom = g_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    agg = guided_vals.sum(dim=1, keepdim=True) / denom
                    agg = agg - X.mean(dim=1, keepdim=True)  # avoid DC drift
                    if unguided_spill_center:
                        agg = agg - agg.median(dim=0, keepdim=True).values
                    X = X + float(unguided_spill_alpha) * (agg * ug_mask)

                elif mode == "corr" and (W is not None):
                    # SIGN-AWARE, correlation-weighted propagation
                    d = (X - X0) * g_mask        # (B,K)
                    G_idx = torch.nonzero(gF > 0.5, as_tuple=False).flatten()
                    dG = d[:, G_idx]             # (B,|G|)

                    # Robust clipping per batch
                    if broadcast_clip > 0:
                        med = dG.median(dim=1, keepdim=True).values
                        mad = (dG - med).abs().median(dim=1, keepdim=True).values + 1e-6
                        dG = torch.clamp(dG,
                                         min=med - broadcast_clip * mad,
                                         max=med + broadcast_clip * mad)

                    # Map guided deltas into all features via signed correlations
                    d_all = dG @ W.T             # (B,K)
                    if unguided_spill_center:
                        d_all = d_all - d_all.median(dim=1, keepdim=True).values
                    X = X + float(unguided_spill_alpha) * (d_all * ug_mask)
                # else: if no W available, skip spill

            # optional UNIFORM broadcast (unchanged)
            if broadcast_w > 0:
                d_g = (X - X0) * gF.unsqueeze(0) * fut.float()
                if broadcast_clip > 0:
                    med = d_g.median(dim=1, keepdim=True).values
                    mad = (d_g - med).abs().median(dim=1, keepdim=True).values + 1e-6
                    d_g = torch.clamp(d_g, min=med - broadcast_clip * mad, max=med + broadcast_clip * mad)

                mode_b = broadcast_mode.lower()
                if mode_b == "sum":
                    d_agg = d_g.sum(dim=1, keepdim=True)
                elif mode_b == "mean":
                    denom = gF.sum().clamp_min(1e-6)
                    d_agg = d_g.sum(dim=1, keepdim=True) / denom
                elif mode_b == "sumabs":
                    d_agg = d_g.abs().sum(dim=1, keepdim=True)
                elif mode_b == "rms":
                    d_agg = torch.sqrt((d_g**2).mean(dim=1, keepdim=True) + 1e-8)
                else:
                    raise ValueError(f"unknown broadcast_mode: {broadcast_mode}")

                if broadcast_center:
                    d_agg = d_agg - d_agg.median(dim=0, keepdim=True).values

                apply_mask = fut.float()
                if broadcast_exclude_guided:
                    apply_mask = apply_mask * (1.0 - gF.unsqueeze(0))

                X = X + float(broadcast_w) * float(broadcast_gain) * (d_agg * apply_mask)

            out[:, :, t_idx] = X

        return out

    if DEBUG_GUIDANCE:
        print(f"[pairs-only] pairs={clean_pairs} | schedule={schedule} | lam_eff={lam_eff:.3f} "
              f"| spill_alpha={unguided_spill_alpha:.2f} | spill_mode={unguided_spill_mode} "
              f"| corr_dynamic={corr_dynamic} | corr_csv={corr_csv if (unguided_spill_mode.lower()=='corr' and not corr_dynamic) else None} "
              f"| corr_abs={corr_abs} | corr_power={corr_power} | corr_norm={corr_norm} "
              f"| broadcast_w={broadcast_w:.2f} | mode={broadcast_mode} | gain={broadcast_gain:.2f} "
              f"| clip={broadcast_clip:.1f} | excl_guided={broadcast_exclude_guided} | center={broadcast_center} "
              f"| NOTE: negative weights enabled for anti-align")

    return cond_fn
