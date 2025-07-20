from xgboost import XGBRegressor
from xgboost.callback import TrainingCallback
from tab_models.model_wrapper import ModelWrapper, ModelCallback
from typing import Callable, Optional, Sequence, Any, List
import pandas as pd
import numpy as np


class XGBoostCallbackWrapper(TrainingCallback):
    def __init__(self, model_callback: Any, model_wrapper: "XGBoostWrapper") -> None:
        self.model_callback = model_callback
        self.model_wrapper = model_wrapper

    def after_iteration(self, model, epoch, evals_log):
        self.model_wrapper.model = model
        self.model_callback.after_iteration(epoch, self.model_wrapper)
        return False


class XgbMetric:
    def __init__(self, metric_fn, eval_frequency):
        self.metric_fn = metric_fn
        self.eval_frequency = eval_frequency
        self.iteration = 0

    def __call__(self):
        def custom_eval(dtrain, predt):
            self.iteration += 1
            if (
                self.eval_frequency == 0
                or (self.iteration - 1) % self.eval_frequency != 0
            ):
                return 0.0

            name, score, _ = self.metric_fn(predt, dtrain)
            return score

        return custom_eval


class XGBoostWrapper(ModelWrapper):
    def __init__(self, params, features, fpath, model_name="xgb_model"):
        self.model_name = model_name
        self.fpath = fpath

        if self.fpath is not None and str(self.fpath).endswith(".pkl"):
            import pickle

            with open(self.fpath, "rb") as f:
                info = pickle.load(f)
            self.model = info["model"]
            self.features = info["feature_names"]
            self.params = info["params"]
        else:
            self.features = features
            self.params = params

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        eval_metrics: Sequence[Callable] = [],
        eval_frequency: int = 100,
        callbacks: List[ModelCallback] = [],
    ) -> None:

        self.model = XGBRegressor(
            objective=self.params["objective"],
            n_estimators=self.params["n_estimators"],
            colsample_bytree=self.params["colsample_bytree"],
            learning_rate=self.params["learning_rate"],
            max_depth=self.params["max_depth"],
            device="cuda:0",
            seed=self.params.get("seed", 0),
            subsample=self.params["subsample"],
            min_child_weight=self.params["min_child_weight"],
            eval_metric=(
                XgbMetric(eval_metrics[0], eval_frequency)()
                if len(eval_metrics) > 0
                else None
            ),
            callbacks=[
                XGBoostCallbackWrapper(callback, self) for callback in callbacks
            ],
        )

        all_target_names = [self.params["target_name"]] + self.params.get(
            "auxiliary_targets", []
        )
        self.model.fit(
            train_data[self.features],
            train_data[all_target_names],
            eval_set=(
                [(val_data[self.features], val_data[all_target_names])]
                if val_data is not None and len(val_data) > 0
                else None
            ),
            verbose=eval_frequency,
        )

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(test_data[self.features]))

    def save(self, fpath: str) -> None:
        import pickle
        import os

        # Check if fpath is a directory
        if os.path.isdir(fpath):
            filename = f"{self.model_name}.pkl"
            save_path = os.path.join(fpath, filename)
        else:
            save_path = fpath

        extra_info = {
            "model": self.model,
            "feature_names": self.features,
            "target_name": self.target_names()[0],
            "params": self.params,
        }
        with open(save_path, "wb") as f:
            pickle.dump(extra_info, f)

    def feature_names(self) -> Sequence[str]:
        return self.features

    def target_names(self) -> Sequence[str]:
        return [self.params["target_name"]]
