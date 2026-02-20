# exe_forecasting.py
# Unified training + guided/unguided evaluation for forecasting datasets.
# Supports GRAPH guidance and PAIRS-ONLY guidance (topology-free) with corr-weighted spill.

import argparse, os, json, yaml, datetime, torch, numpy as np
from dataset_forecasting import get_dataloader, Forecasting_Dataset
from main_model import CSDI_Forecasting
from utils import train, evaluate
from graph_guidance import build_cond_fn, build_cond_fn_pairs_only


def parse_guided_features(s: str | None):
    if s is None or s.strip() == "":
        return None
    if s.startswith("file:"):
        path = s.split("file:", 1)[1]
        with open(path, "r") as f:
            idx = [int(line.strip()) for line in f if line.strip()]
        return idx
    return [int(x) for x in s.split(",") if x.strip() != ""]


def parse_pairs_inline(s: str | None):
    if s is None or s.strip() == "":
        return []
    out = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) == 2:
            i, j = int(parts[0]), int(parts[1]); w = 1.0
        elif len(parts) == 3:
            i, j, w = int(parts[0]), int(parts[1]), float(parts[2])
        else:
            continue
        out.append((i, j, w))
    return out


def parse_pairs_file(path: str | None):
    if path is None or path.strip() == "" or not os.path.exists(path):
        return []
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.replace(",", " ").split()]
            if len(parts) == 2:
                i, j = int(parts[0]), int(parts[1]); w = 1.0
            elif len(parts) >= 3:
                i, j, w = int(parts[0]), int(parts[1]), float(parts[2])
            else:
                continue
            pairs.append((i, j, w))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="CSDI Forecasting (graph or pairs-only guidance)")

    # Config / dataset identity
    parser.add_argument("--config", type=str, default="base_forecasting.yaml")
    parser.add_argument("--datatype", type=str, default="electricity")

    # Runtime/device/seed
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unconditional", action="store_true")

    # Save/load
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--nsample", type=int, default=100)

    # Data / horizons
    parser.add_argument("--data_pkl_path", type=str, default=None)
    parser.add_argument("--meanstd_pkl_path", type=str, default=None)
    parser.add_argument("--history_len", type=int, default=168)
    parser.add_argument("--pred_len", type=int, default=24)

    # (These are ignored by your current dataset_forecasting.py; kept for compatibility)
    parser.add_argument("--valid_length", type=int, default=0, help="optional override; not used by this loader")
    parser.add_argument("--test_length", type=int, default=0, help="optional override; not used by this loader")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="not used by this loader")

    # -------- Training CLI overrides --------
    parser.add_argument("--epochs", type=int, default=None, help="override config.train.epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="override config.train.batch_size")
    parser.add_argument("--itr_per_epoch", type=int, default=None, help="override config.train.itr_per_epoch")

    # ----- GRAPH guidance args -----
    parser.add_argument("--graph_csv", type=str, default="")
    parser.add_argument("--graph_format", type=str, default="dense", choices=["dense", "edges"])
    parser.add_argument("--lambda_g", type=float, default=0.8)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--guidance_schedule", type=str, default="late90",
                        choices=["always", "late50", "late90"])
    parser.add_argument("--snr_gate", type=float, default=0.0)
    parser.add_argument("--guided_features", type=str, default="")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--ramp_last", type=float, default=0.5)
    parser.add_argument("--disable_pair_tie", action="store_true")
    parser.add_argument("--pair_tie_weight", type=float, default=0.5)

    parser.add_argument("--spill", type=float, default=0.30)
    parser.add_argument("--spill_mode", type=str, default="heat", choices=["uniform", "adj", "heat"])
    parser.add_argument("--prox_floor", type=float, default=0.20)
    parser.add_argument("--prox_gamma", type=float, default=0.80)

    parser.add_argument("--broadcast_w", type=float, default=0.0)
    parser.add_argument("--broadcast_hops", type=int, default=2)
    parser.add_argument("--broadcast_clip", type=float, default=3.0)

    # ----- PAIRS-ONLY guidance args -----
    parser.add_argument("--pairs", type=str, default="")
    parser.add_argument("--pairs_file", type=str, default="")
    parser.add_argument("--pairs_broadcast_w", type=float, default=0.0)
    parser.add_argument("--pairs_broadcast_clip", type=float, default=3.0)
    parser.add_argument("--pairs_broadcast_mode", type=str, default="mean",
                        choices=["mean", "sum", "sumabs", "rms"])
    parser.add_argument("--pairs_broadcast_gain", type=float, default=1.0)
    parser.add_argument("--pairs_broadcast_exclude_guided", action="store_true")
    parser.add_argument("--pairs_broadcast_center", action="store_true")

    # NEW: unguided spill (pairs-only) — supports "corr"
    parser.add_argument("--unguided_spill_alpha", type=float, default=0.0)
    parser.add_argument("--unguided_spill_mode", type=str, default="delta",
                        choices=["delta", "value", "corr"])
    parser.add_argument("--unguided_spill_center", action="store_true")

    # NEW: correlation-weighted spill knobs
    parser.add_argument("--corr_csv", type=str, default="")        # optional: KxK CSV
    parser.add_argument("--corr_dynamic", action="store_true")     # build from current history, no files
    parser.add_argument("--corr_abs", action="store_true")         # ignore sign if set
    parser.add_argument("--corr_power", type=float, default=1.0)   # emphasize strong links
    parser.add_argument("--corr_norm", type=str, default="colmax",
                        choices=["colmax", "l1", "none"])

    args = parser.parse_args()
    print(args)

    # -----------------------
    torch.manual_seed(args.seed)
    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Optional CLI overrides for training section
    if "train" not in config:
        config["train"] = {}
    config["train"].setdefault("batch_size", 8)
    config["train"].setdefault("epochs", 50)
    config["train"].setdefault("itr_per_epoch", 1000)

    if args.batch_size is not None:
        config["train"]["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.itr_per_epoch is not None:
        config["train"]["itr_per_epoch"] = int(args.itr_per_epoch)

    # Model conditionality
    config["model"]["is_unconditional"] = args.unconditional
    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_prefix = f"forecasting_{args.datatype}"
    if args.modelfolder.strip():
        base_prefix = args.modelfolder.strip()
    foldername = f"./save/{base_prefix}_{current_time}/"
    os.makedirs(foldername, exist_ok=True)
    with open(os.path.join(foldername, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print("model folder:", foldername)

    # -----------------------
    # Dataloaders (match your dataset_forecasting.py signature)
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        datatype=args.datatype,
        device=args.device,
        batch_size=config["train"]["batch_size"],
        data_pkl_path=args.data_pkl_path,
        meanstd_pkl_path=args.meanstd_pkl_path,
        history_length=args.history_len,   # <- fixed name
        pred_length=args.pred_len,         # <- fixed name
    )

    target_dim = train_loader.dataset.target_dim if hasattr(train_loader.dataset, "target_dim") else train_loader.dataset.main_data.shape[1]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[train] epochs={config['train']['epochs']} | batch_size={config['train']['batch_size']} "
          f"| itr_per_epoch={config['train']['itr_per_epoch']} | target_dim={target_dim}")

    # -----------------------
    model = CSDI_Forecasting(config, device, target_dim).to(device)
    model.target_dim = target_dim

    # Optional: load existing checkpoint from a prior run folder
    load_ckpt = False
    if args.modelfolder.strip():
        candidate = os.path.join("./save", args.modelfolder.strip(), "model.pth")
        if os.path.exists(candidate):
            print("Loading checkpoint:", candidate)
            model.load_state_dict(torch.load(candidate, map_location=device))
            load_ckpt = True

    if not load_ckpt:
        train(model, config["train"], train_loader, valid_loader, foldername=foldername)

    # -----------------------
    # Diffusion alphas for schedule/snr gating in cond_fns
    diff = config["diffusion"]
    num_steps = diff["num_steps"]
    if diff["schedule"] == "quad":
        beta = np.linspace(diff["beta_start"] ** 0.5, diff["beta_end"] ** 0.5, num_steps) ** 2
    elif diff["schedule"] == "linear":
        beta = np.linspace(diff["beta_start"], diff["beta_end"], num_steps)
    else:
        raise ValueError("Unknown diffusion schedule.")
    alpha_hat = 1.0 - beta
    alpha_series = np.cumprod(alpha_hat)

    # -----------------------
    # Choose guidance
    cond_fn = None

    pairs = []
    if args.pairs_file:
        pairs = parse_pairs_file(args.pairs_file)
    elif args.pairs:
        pairs = parse_pairs_inline(args.pairs)

    if len(pairs) > 0:
        print(f"[info] Using PAIRS-ONLY guidance with {len(pairs)} pair(s).")
        cond_fn = build_cond_fn_pairs_only(
            pairs=pairs,
            num_features=target_dim,
            lambda_g=args.lambda_g,
            schedule=args.guidance_schedule,
            num_steps=num_steps,
            snr_gate=args.snr_gate,
            alpha_series=alpha_series,
            broadcast_w=args.pairs_broadcast_w,
            broadcast_clip=args.pairs_broadcast_clip,
            broadcast_mode=args.pairs_broadcast_mode,
            broadcast_gain=args.pairs_broadcast_gain,
            broadcast_exclude_guided=args.pairs_broadcast_exclude_guided,
            broadcast_center=args.pairs_broadcast_center,
            unguided_spill_alpha=args.unguided_spill_alpha,
            unguided_spill_mode=args.unguided_spill_mode,
            unguided_spill_center=args.unguided_spill_center,
            corr_csv=(args.corr_csv if (args.corr_csv and os.path.exists(args.corr_csv)) else None),
            corr_dynamic=bool(args.corr_dynamic),
            corr_abs=bool(args.corr_abs),
            corr_power=float(args.corr_power),
            corr_norm=args.corr_norm,
        )
    elif args.graph_csv and os.path.exists(args.graph_csv):
        print("[info] Using GRAPH guidance.")
        guided_idx = parse_guided_features(args.guided_features)
        cond_fn = build_cond_fn(
            graph_csv=args.graph_csv,
            graph_format=args.graph_format,
            num_features=target_dim,
            lambda_g=args.lambda_g,
            eta=args.eta,
            tau=args.tau,
            schedule=args.guidance_schedule,
            snr_gate=args.snr_gate,
            guided_features=guided_idx,
            num_steps=num_steps,
            alpha_series=alpha_series,
            passes=args.passes,
            ramp_last=args.ramp_last,
            enable_pair_tie=(not args.disable_pair_tie),
            pair_tie_weight=args.pair_tie_weight,
            spill=args.spill,
            spill_mode=args.spill_mode,
            prox_floor=args.prox_floor,
            prox_gamma=args.prox_gamma,
            broadcast_w=args.broadcast_w,
            broadcast_hops=args.broadcast_hops,
            broadcast_clip=args.broadcast_clip,
        )
    elif args.graph_csv:
        print(f"[warn] Graph CSV not found at {args.graph_csv}, running unguided.")

    # -----------------------
    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
        cond_fn=cond_fn,
    )


if __name__ == "__main__":
    main()
