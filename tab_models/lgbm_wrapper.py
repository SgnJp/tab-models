import os
import joblib
import gc
from typing import Any, Callable, List, Optional, Sequence, Union

import pandas as pd
import numpy as np
from lightgbm.callback import CallbackEnv
import lightgbm as lgb
import torch

from tab_models.model_wrapper import ModelWrapper, ModelCallback


class LGBMCallbackWrapper:
    def __init__(self, model_callback: Any, model_wrapper: "LGBMWrapper") -> None:
        self.model_callback = model_callback
        self.model_wrapper = model_wrapper
        self.order: int = 100  # ensure it runs after built-in callbacks
        self.before_iteration: bool = False

    def __call__(self, env: CallbackEnv) -> None:
        self.model_wrapper.model = (
            env.model
        )  # Ensure wrapper always has the latest model
        self.model_callback.after_iteration(env.iteration, self.model_wrapper)


class LGBMWrapper(ModelWrapper):
    def __init__(
        self,
        params: Optional[dict],
        features: Optional[Sequence[str]],
        fpath: Optional[str],
        model_name: str = "lgbm_model",
    ) -> None:
        self.fpath: Optional[str] = fpath
        self.features: Optional[Sequence[str]] = features
        self.model_name: str = model_name
        self.params: Optional[dict] = params
        self.model: Optional[lgb.Booster] = None
        self.target_name: Optional[str] = None

        if fpath is not None:
            self.model = joblib.load(fpath)
            self.features = self.model.feature_name()
            self.target_name = self.model.target_name

        if self.params is not None:
            self.params["num_leaves"] = 2 ** self.params["max_depth"] - 1
            if torch.cuda.is_available():
                self.params["device"] = "gpu"
            self.params["verbosity"] = -1
            self.target_name = self.params["target_name"]

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        eval_metrics: Optional[Callable] = None,
        callbacks: Optional[List[ModelCallback]] = None,
    ) -> None:
        if callbacks is None:
            callbacks = []
        train_set = lgb.Dataset(
            train_data[self.features],
            train_data[self.params["target_name"]],
            free_raw_data=True,
        )
        valid_sets = (
            [lgb.Dataset(val_data[self.features], val_data[self.params["target_name"]])]
            if len(val_data) > 0
            else []
        )

        del train_data
        gc.collect()

        self.model = lgb.train(
            self.params,
            train_set,
            valid_sets=valid_sets,
            valid_names=["valid"],
            feval=eval_metrics,
            init_model=self.model,
            callbacks=[LGBMCallbackWrapper(cb, self) for cb in callbacks],
        )

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        return self.model.predict(test_data[self.features])

    def save(self, fpath: str) -> str:
        assert self.model is not None
        self.model.target_name = self.params["target_name"]
        if os.path.isdir(fpath):
            joblib.dump(self.model, os.path.join(fpath, f"{self.model_name}.bin"))
            return os.path.join(fpath, f"{self.model_name}.bin")
        else:
            joblib.dump(self.model, fpath)
            return fpath

    def feature_names(self) -> Sequence[str]:
        return self.features

    def target_names(self):
        return [self.target_name]
