import json
import zipfile

import pickle
import base64
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import torch
import torch.optim as optim

from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.metrics import Metric


from utils import log_timing, write_json
from model_wrapper import ModelWrapper
from nn_utils import get_loss

from typing import Any, Callable, Sequence, Optional, Tuple, Dict, List, Union

from nn_utils import ScalerImputer


def get_noise_augmentation_func(noise_std: float) -> Callable:

    def noise_augmentation(x: torch.Tensor, y: Any) -> Tuple[torch.Tensor, Any]:
        noise = torch.randn_like(x) * noise_std
        x_noisy = x + noise
        return x_noisy, y

    return noise_augmentation


def get_metric_wrapper(metric_function: Callable) -> Callable[[], "Metric"]:
    class MetricWrapper(Metric):
        def __init__(self) -> None:
            self._name, _, self._maximize = metric_function(None, None)

        def __call__(self, y_true: Any, y_pred: Any) -> Any:
            print("val", len(y_true), len(y_pred))
            return metric_function(y_pred, y_true)[1]

    return MetricWrapper


class TabNetWrapper(ModelWrapper):
    file_to_add: str = "tabnet_wrapper_params.json"
    features: Sequence[str]
    params: Dict[str, Any]
    scaler_imputer: ScalerImputer
    preprocess_fit: bool
    model: Any
    model_name: str

    def __init__(
        self,
        params: Dict[str, Any],
        features: Sequence[str],
        fpath: Optional[str],
        model_name: str = "tabnet",
    ) -> None:
        if fpath is None:
            self.features = features
            self.params = params
            self.scaler_imputer = ScalerImputer()
            self.preprocess_fit = False
        else:
            with zipfile.ZipFile(fpath, "r") as zipf:
                with zipf.open(self.file_to_add) as json_file:
                    data = json.load(json_file)

            self.features = data["features"]
            self.params = data["params"]
            self.scaler_imputer = ScalerImputer.load(data["scaler"], data["imputer"])
            self.preprocess_fit = True

        self.model = TabNetRegressor(
            n_d=self.params["n_ad"],
            n_a=self.params["n_ad"],
            n_steps=self.params["n_steps"],
            gamma=self.params["gamma"],
            lambda_sparse=self.params["lambda_sparse"],
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(lr=self.params["lr"], weight_decay=0.01),
            scheduler_fn=optim.lr_scheduler.CosineAnnealingLR,
            scheduler_params={"T_max": self.params["num_epochs"], "eta_min": 1e-6},
            clip_value=1.0,
            seed=self.params.get("seed", 0),
            verbose=1,
        )

        if fpath is not None:
            self.model.load_model(fpath)

        self.model_name = model_name

    def _prepare_data(
        self, data: pd.DataFrame, fit_preprocess: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X, y = (
            data[self.features],
            data[
                [self.params["target_name"]] + self.params.get("auxiliary_targets", [])
            ],
        )

        if fit_preprocess:
            self.scaler_imputer.fit(X[::10])
            self.preprocess_fit = True

        X = self.scaler_imputer.transform(X)
        return X, y

    def _split_train_val(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data["era"] = data["era"].astype(int)

        val_era_start = data.era.max() - self.params["num_val_eras"]
        train_era_end = val_era_start - 4

        print(f"Training on data before: {train_era_end}")
        train_data = data[data.era < train_era_end]
        print(f"Validating on data starting with: {val_era_start}")
        val_data = data[data.era >= val_era_start]

        val_meta = val_data[[self.params["target_name"], "era"]].copy()
        return train_data, val_data, val_meta

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        eval_metrics: Optional[Sequence[Callable]] = None,
    ) -> None:
        if eval_metrics is None:
            eval_metrics = []
        X_train, y_train = self._prepare_data(
            train_data, fit_preprocess=not self.preprocess_fit
        )
        del train_data
        X_val, y_val = self._prepare_data(val_data, fit_preprocess=False)

        self.model.augmentation = get_noise_augmentation_func(
            self.params.get("input_noise", 0.0)
        )
        self.model.fit(
            X_train=X_train.values,
            y_train=y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            eval_metric=[get_metric_wrapper(metric_fn) for metric_fn in eval_metrics],
            max_epochs=self.params["num_epochs"],
            patience=self.params.get("patience", 10000),
            batch_size=self.params["batch_size"],
            virtual_batch_size=self.params["virtual_batch_size"],
            loss_fn=get_loss(self.params["loss_fn"]),
            compute_importance=False,
            num_workers=self.params.get("num_workers", 1),
        )

    @log_timing()
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        test_data = test_data[self.features]
        test_data = self.scaler_imputer.transform(test_data)
        return self.model.predict(test_data.values)[:, 0]

    def dump(self) -> Dict[str, Any]:
        scaler_b64, imputer_b64 = self.scaler_imputer.dump()
        return {
            "scaler": scaler_b64,
            "imputer": imputer_b64,
            "params": self.params,
            "features": self.features,
        }

    def save(self, fpath: str) -> None:
        self.model.save_model(fpath)
        os.rename(f"{fpath}.zip", fpath)

        write_json(self.dump(), os.path.join("/tmp", self.file_to_add))
        with zipfile.ZipFile(fpath, "a") as zipf:
            zipf.write(os.path.join("/tmp", self.file_to_add), arcname=self.file_to_add)

    def feature_names(self) -> Sequence[str]:
        return self.features

    def target_names(self) -> Sequence[str]:
        return [self.params["target_name"]]
