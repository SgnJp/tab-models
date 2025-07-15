import os
import joblib
from typing import Any, Callable, List, Optional, Sequence

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool

from tab_models.model_wrapper import ModelWrapper


class CatBoostWrapper(ModelWrapper):
    def __init__(
        self,
        params: Optional[dict],
        features: Optional[Sequence[str]],
        fpath: Optional[str],
        model_name: str = "catboost_model",
    ) -> None:
        self.fpath: Optional[str] = fpath
        self.features: Optional[Sequence[str]] = features
        self.model_name: str = model_name
        self.params: Optional[dict] = params
        self.model: Optional[CatBoostRegressor] = None
        self.target_name: Optional[str] = None

        if fpath is not None:
            loaded_obj = joblib.load(fpath)
            if isinstance(loaded_obj, dict) and "model" in loaded_obj:
                self.model = loaded_obj["model"]
                self.features = loaded_obj.get("features", None)
                self.target_name = loaded_obj.get("target_name", None)
            else:
                self.model = loaded_obj
                self.features = self.model.feature_names_
                self.target_name = getattr(self.model, "target_name_", None)
        else:
            if self.params is not None:
                self.target_name = self.params["target_name"]
                allowed_params = [
                    "iterations",
                    "learning_rate",
                    "depth",
                    "loss_function",
                    "subsample",
                    "rsm",
                    "min_data_in_leaf",
                    "max_bin",
                    "grow_policy",
                    # 'seed' will be mapped to 'random_seed'
                ]
                cat_params = {
                    k: v
                    for k, v in self.params.items()
                    if k in allowed_params and k != "seed"
                }
                # Map 'seed' to 'random_seed' if present
                if "seed" in self.params:
                    cat_params["random_seed"] = self.params["seed"]
                if cat_params.get("task_type") is None:
                    # Use GPU if available
                    try:
                        import catboost

                        if catboost.config.GPU_DEVICE_COUNT > 0:
                            cat_params["task_type"] = "GPU"
                    except Exception:
                        pass
                self.model = CatBoostRegressor(**cat_params)

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        eval_metrics: Optional[Callable] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> None:
        X_train = train_data[self.features]
        y_train = train_data[self.params["target_name"]]
        eval_set = None
        if val_data is not None:
            X_val = val_data[self.features]
            y_val = val_data[self.params["target_name"]]
            eval_set = Pool(X_val, y_val)
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            use_best_model=True if eval_set is not None else False,
            verbose=self.params.get("verbose", 100),
        )

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        return self.model.predict(test_data[self.features])

    def save(self, fpath: str) -> str:
        assert self.model is not None
        obj_to_save = {
            "model": self.model,
            "target_name": self.target_name,
            "features": self.features,
        }
        if os.path.isdir(fpath):
            joblib.dump(obj_to_save, os.path.join(fpath, f"{self.model_name}.cbm"))
            return os.path.join(fpath, f"{self.model_name}.cbm")
        else:
            joblib.dump(obj_to_save, fpath)
            return fpath

    def feature_names(self) -> Sequence[str]:
        return self.features

    def target_names(self):
        return [self.target_name]
