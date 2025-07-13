import pickle
import base64
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Optional

import torch
from torch import nn


class ScalerImputer:
    """
    Encapsulates a StandardScaler and SimpleImputer for tabular data preprocessing.
    Provides fit, transform, dump, and load methods with batching and serialization.
    """

    def __init__(
        self,
        scaler: Optional[StandardScaler] = None,
        imputer: Optional[SimpleImputer] = None,
    ):
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.imputer = (
            imputer if imputer is not None else SimpleImputer(strategy="mean")
        )

    def fit(self, X: pd.DataFrame) -> None:
        self.scaler.fit(X)
        self.imputer.fit(X)

    def transform(
        self, X: pd.DataFrame, batch_size: int = 50000, dtype=np.float32
    ) -> pd.DataFrame:
        X = X.astype(dtype, copy=False)
        X = X.copy()
        for i in range(0, len(X), batch_size):
            X.iloc[i : i + batch_size] = self.imputer.transform(
                X.iloc[i : i + batch_size]
            )
            X.iloc[i : i + batch_size] = self.scaler.transform(
                X.iloc[i : i + batch_size]
            )
        return X

    def dump(self) -> Tuple[str, str]:
        scaler_bytes = pickle.dumps(self.scaler)
        scaler_b64 = base64.b64encode(scaler_bytes).decode("utf-8")
        imputer_bytes = pickle.dumps(self.imputer)
        imputer_b64 = base64.b64encode(imputer_bytes).decode("utf-8")
        return scaler_b64, imputer_b64

    @classmethod
    def load(cls, scaler_b64: str, imputer_b64: str) -> "ScalerImputer":
        scaler = pickle.loads(base64.b64decode(scaler_b64))
        imputer = pickle.loads(base64.b64decode(imputer_b64))
        return cls(scaler, imputer)


def get_loss(loss_name):
    if loss_name == "spearman":
        return spearman_loss
    elif loss_name == "mse":
        return weighted_mse_loss
    elif loss_name == "huber":
        return weighted_huber_loss
    assert False


def spearman_loss(pred, target, factor=0.1):
    import torchsort

    pred = torchsort.soft_rank(pred.transpose(0, 1)).transpose(0, 1)
    target = torchsort.soft_rank(target.transpose(0, 1)).transpose(0, 1)
    eps = 1e-8
    pred = pred - pred.mean()
    pred = pred / (eps + pred.norm())
    num_extra_targets = 0 if len(pred.shape) == 1 else pred.shape[1] - 1
    target = target - target.mean()
    target = target / (eps + target.norm())
    weights = np.insert(
        (1 - factor) * np.ones(num_extra_targets) / num_extra_targets, 0, factor, axis=0
    )
    weights = torch.tensor(weights, device=pred.device)
    return -((pred * target).sum(dim=0) * weights).sum()
    return -(pred * target).sum()


def weighted_generic_loss(criterion, pred, target, factor=0.1):
    target = torch.nan_to_num(target, nan=0.0)
    pred = torch.nan_to_num(pred, nan=0.0)
    num_extra_targets = 0 if len(pred.shape) == 1 else pred.shape[1] - 1
    loss = criterion(pred, target)
    weights = np.insert(
        (1 - factor) * np.ones(num_extra_targets) / num_extra_targets, 0, factor, axis=0
    )
    weights = torch.tensor(weights, device=pred.device)
    weighted_loss = loss * weights
    return weighted_loss.mean()


def weighted_huber_loss(pred, target, factor=0.1):
    criterion = nn.HuberLoss(reduction="none", delta=0.1)
    return weighted_generic_loss(criterion, pred, target, factor)


def weighted_mse_loss(pred, target, factor=0.1):
    criterion = nn.MSELoss(reduction="none")
    return weighted_generic_loss(criterion, pred, target, factor)


def custom_loss(pred, target, factor=0.1):
    return 0.25 * spearman_loss(pred, target, factor) + 0.75 * weighted_mse_loss(
        pred, target, factor
    )
