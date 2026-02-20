import os
import pickle
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm


# -----------------------------
# Training
# -----------------------------
def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = os.path.join(foldername, "model.pth")

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0.0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break
            lr_scheduler.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0.0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)


# -----------------------------
# CRPS helpers (CPU + device-safe)
# -----------------------------
def _quantile_loss_cpu(target, forecast_q, q: float, eval_points):
    return 2.0 * torch.sum(
        torch.abs((forecast_q - target) * eval_points * ((target <= forecast_q).float() - q))
    )


def _denom_cpu(target, eval_points):
    return torch.sum(torch.abs(target * eval_points)) + 1e-12


def calc_quantile_CRPS_cpu(target, forecast, eval_points, mean_scaler, scaler):
    """
    target   : (N, L, K) tensor
    forecast : (N, nsample, L, K) tensor
    eval_pts : (N, L, K) tensor
    """
    target = target.detach().cpu().float()
    forecast = forecast.detach().cpu().float()
    eval_points = eval_points.detach().cpu().float()

    mean_scaler = torch.as_tensor(mean_scaler).detach().cpu().float()
    scaler = torch.as_tensor(scaler).detach().cpu().float()

    # denormalize
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = torch.arange(0.05, 1.00, 0.05, dtype=torch.float32)
    denom = _denom_cpu(target, eval_points)

    crps = 0.0
    for q in quantiles:
        q_pred = torch.quantile(forecast, q.item(), dim=1)  # (N,L,K)
        q_loss = _quantile_loss_cpu(target, q_pred, q.item(), eval_points)
        crps += (q_loss / denom)
    crps = crps / len(quantiles)
    return float(crps)


def calc_quantile_CRPS_sum_cpu(target, forecast, eval_points, mean_scaler, scaler):
    """
    Sum over features version.
    """
    target = target.detach().cpu().float()
    forecast = forecast.detach().cpu().float()
    eval_points = eval_points.detach().cpu().float()

    mean_scaler = torch.as_tensor(mean_scaler).detach().cpu().float()
    scaler = torch.as_tensor(scaler).detach().cpu().float()

    # denormalize
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    # sum over features
    target_sum = target.sum(dim=-1)             # (N, L)
    forecast_sum = forecast.sum(dim=-1)         # (N, nsample, L)
    eval_points_sum = eval_points.mean(dim=-1)  # (N, L)

    quantiles = torch.arange(0.05, 1.00, 0.05, dtype=torch.float32)
    denom = torch.sum(torch.abs(target_sum * eval_points_sum)) + 1e-12

    crps = 0.0
    for q in quantiles:
        q_pred = torch.quantile(forecast_sum, q.item(), dim=1)  # (N, L)
        q_loss = 2.0 * torch.sum(
            torch.abs((q_pred - target_sum) * eval_points_sum * ((target_sum <= q_pred).float() - q.item()))
        )
        crps += (q_loss / denom)
    crps = crps / len(quantiles)
    return float(crps)


# -----------------------------
# Evaluation / Sampling (no-grad)
# -----------------------------
def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", cond_fn=None):
    """
    Run sampling on test_loader and save:
      - generated_outputs_nsample{nsample}.pk  (big bundle)
      - result_nsample{nsample}.pk             (RMSE/MAE/CRPS)
    Always under torch.no_grad() to avoid autograd memory.
    """
    os.makedirs(foldername, exist_ok=True)
    model.eval()

    scaler_cpu = torch.as_tensor(scaler).detach().cpu().float()

    with torch.no_grad():
        mse_total = 0.0
        mae_total = 0.0
        evalpoints_total = 0.0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                # model.evaluate internally also uses no_grad for sampling
                samples, c_target, eval_points, observed_points, observed_time = model.evaluate(
                    test_batch, nsample, cond_fn=cond_fn
                )

                # move to CPU early
                samples = samples.permute(0, 1, 3, 2).contiguous().cpu()      # (B,ns,L,K)
                c_target = c_target.permute(0, 2, 1).contiguous().cpu()       # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1).contiguous().cpu() # (B,L,K)
                observed_points = observed_points.permute(0, 2, 1).contiguous().cpu()
                observed_time = observed_time.cpu()

                # median for point metrics
                samples_median = samples.median(dim=1).values  # (B,L,K)

                mse_current = (((samples_median - c_target) * eval_points) ** 2)
                mse_current = mse_current * (scaler_cpu ** 2)
                mae_current = torch.abs((samples_median - c_target) * eval_points) * scaler_cpu

                mse_total += float(mse_current.sum().item())
                mae_total += float(mae_current.sum().item())
                evalpoints_total += float(eval_points.sum().item())

                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / max(evalpoints_total, 1.0)),
                        "mae_total": mae_total / max(evalpoints_total, 1.0),
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # concat on CPU
    all_target = torch.cat(all_target, dim=0)
    all_evalpoint = torch.cat(all_evalpoint, dim=0)
    all_observed_point = torch.cat(all_observed_point, dim=0)
    all_observed_time = torch.cat(all_observed_time, dim=0)
    all_generated_samples = torch.cat(all_generated_samples, dim=0)

    # save bundle
    with open(os.path.join(foldername, f"generated_outputs_nsample{nsample}.pk"), "wb") as f:
        pickle.dump(
            [
                all_generated_samples,
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                scaler,
                mean_scaler,
            ],
            f,
        )

    # scalars
    rmse = np.sqrt(mse_total / max(evalpoints_total, 1.0))
    mae = mae_total / max(evalpoints_total, 1.0)

    # CRPS safely on CPU
    try:
        CRPS = calc_quantile_CRPS_cpu(
            all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
        )
        CRPS_sum = calc_quantile_CRPS_sum_cpu(
            all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
        )
    except Exception as e:
        print(f"[warn] CRPS computation failed: {e}")
        CRPS = float("nan")
        CRPS_sum = float("nan")

    with open(os.path.join(foldername, f"result_nsample{nsample}.pk"), "wb") as f:
        pickle.dump([rmse, mae, CRPS], f)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("CRPS:", CRPS)
    print("CRPS_sum:", CRPS_sum)
