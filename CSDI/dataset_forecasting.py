# dataset_forecasting.py
# Sliding-window forecasting dataset for CSDI with configurable history/pred lengths.
# If custom pickle paths are provided, the loader accepts any --datatype string.

import pickle
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from typing import Optional


def _safe_arange(start: int, end: int, step: int) -> np.ndarray:
    """np.arange that returns an empty int array if end <= start."""
    if step == 0:
        raise ValueError("step must be non-zero")
    if end <= start:
        return np.asarray([], dtype=int)
    return np.arange(start, end, step, dtype=int)


class Forecasting_Dataset(Dataset):
    """
    Sliding-window forecasting dataset for CSDI.

    Expects pickles:
      - data.pkl:    (main_data, mask_data) with shapes (T, D), (T, D)
      - meanstd.pkl: (mean_data, std_data) with shapes (D,), (D,)

    If data_pkl_path & meanstd_pkl_path are provided, any --datatype is accepted.
    If they are NOT provided, --datatype must be 'electricity' to use defaults.
    """

    def __init__(
        self,
        datatype: str,
        mode: str = "train",
        data_pkl_path: Optional[str] = None,
        meanstd_pkl_path: Optional[str] = None,
        history_length: int = 168,
        pred_length: int = 24,
    ):
        self.history_length = int(history_length)
        self.pred_length = int(pred_length)
        self.seq_length = self.history_length + self.pred_length

        # --- evaluation spans (default to hourly electricity dataset) ---
        self.test_length = 24 * 7
        self.valid_length = 24 * 5

        # --- resolve paths ---
        if (data_pkl_path is None) or (meanstd_pkl_path is None):
            # No explicit paths: only electricity defaults are known
            if datatype != "electricity":
                raise ValueError(
                    f"Unsupported datatype without explicit paths: {datatype}. "
                    f"Provide --data_pkl_path and --meanstd_pkl_path."
                )
            datafolder = "./data/electricity_nips"
            data_pkl_path = data_pkl_path or (datafolder + "/data.pkl")
            meanstd_pkl_path = meanstd_pkl_path or (datafolder + "/meanstd.pkl")

        # --- load data and stats ---
        with open(data_pkl_path, "rb") as f:
            self.main_data, self.mask_data = pickle.load(f)
        with open(meanstd_pkl_path, "rb") as f:
            self.mean_data, self.std_data = pickle.load(f)

        # guard against zeros
        eps = 1e-6
        self.std_data = np.where(self.std_data < eps, 1.0, self.std_data)

        # normalize (broadcast over time)
        self.main_data = (self.main_data - self.mean_data) / self.std_data

        # expose dims
        self.target_dim = int(self.main_data.shape[1])

        # --- build index ranges ---
        total_length = int(len(self.main_data))
        latest_start = total_length - self.seq_length  # last valid start index (inclusive)

        if mode == "train":
            start = 0
            end = total_length - self.seq_length - self.valid_length - self.test_length + 1
            self.use_index = _safe_arange(start, end, 1)

        elif mode == "valid":
            start = total_length - self.seq_length - self.valid_length - self.test_length + self.pred_length
            end = total_length - self.seq_length - self.test_length + self.pred_length
            self.use_index = _safe_arange(start, end, self.pred_length)

        elif mode == "test":
            start = total_length - self.seq_length - self.test_length + self.pred_length
            end = total_length - self.seq_length + self.pred_length
            self.use_index = _safe_arange(start, end, self.pred_length)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # clamp to ensure exact-length slices
        if latest_start < 0:
            self.use_index = np.asarray([], dtype=int)
        else:
            self.use_index = self.use_index[self.use_index <= latest_start].astype(int)

        if len(self.use_index) == 0:
            raise ValueError(
                f"No valid windows for mode={mode}. "
                f"T={total_length}, seq={self.seq_length}, valid={self.valid_length}, test={self.test_length}."
            )

    def __getitem__(self, orgindex: int):
        index = int(self.use_index[orgindex])
        target_mask = self.mask_data[index : index + self.seq_length].copy()
        # hide prediction horizon
        target_mask[-self.pred_length :] = 0.0

        s = {
            "observed_data": self.main_data[index : index + self.seq_length],
            "observed_mask": self.mask_data[index : index + self.seq_length],
            "gt_mask": target_mask,
            "timepoints": np.arange(self.seq_length, dtype=float),
            "feature_id": np.arange(self.main_data.shape[1], dtype=float),
        }
        return s

    def __len__(self):
        return int(len(self.use_index))


def get_dataloader(
    datatype: str,
    device: str,
    batch_size: int = 8,
    data_pkl_path: Optional[str] = None,
    meanstd_pkl_path: Optional[str] = None,
    history_length: int = 168,
    pred_length: int = 24,
):
    """
    Construct train/valid/test loaders while allowing custom pickle paths
    and configurable history/prediction lengths.
    """
    common_kwargs = dict(
        datatype=datatype,
        data_pkl_path=data_pkl_path,
        meanstd_pkl_path=meanstd_pkl_path,
        history_length=history_length,
        pred_length=pred_length,
    )

    ds_train = Forecasting_Dataset(mode="train", **common_kwargs)
    ds_valid = Forecasting_Dataset(mode="valid", **common_kwargs)
    ds_test  = Forecasting_Dataset(mode="test", **common_kwargs)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, drop_last=False)

    scaler = torch.from_numpy(ds_train.std_data).to(device).float()
    mean_scaler = torch.from_numpy(ds_train.mean_data).to(device).float()

    # stash target_dim on the dataset for convenience
    return train_loader, valid_loader, test_loader, scaler, mean_scaler
