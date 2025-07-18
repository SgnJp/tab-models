from xgboost import XGBRegressor
from xgboost.callback import TrainingCallback
from tab_models.model_wrapper import ModelWrapper
from tab_models.utils import log_timing
import os
import time
from typing import Callable, Optional, Sequence
import pandas as pd
import numpy as np


class CheckpointCallback(TrainingCallback):
    def __init__(self, save_period, save_path_prefix):
        self.save_period = save_period
        self.save_path_prefix = save_path_prefix

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % self.save_period == 0:
            filename = f"{self.save_path_prefix}_iter{epoch + 1}.json"
            model.save_model(os.path.join("checkpoints", filename))
            print(f"[Checkpoint] Saved model at iteration {epoch + 1} to {filename}")
        return False


class TimeCallback(TrainingCallback):
    def __init__(self):
        self.times = []

    def before_training(self, model):
        self.start_time = time.time()
        return model

    def after_iteration(self, model, epoch, evals_log):
        now = time.time()
        if (epoch + 1) % 50 == 0:
            elapsed = now - self.start_time
            self.times.append(elapsed)
            self.start_time = now
            print(f"Iteration {epoch}, elapsed: {elapsed:.4f} sec")

        return False


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
        eval_metrics: Optional[Callable] = None,
        callbacks: Optional[list] = None,
    ) -> None:
        def custom_eval(dtrain, predt):
            name, score, _ = eval_metrics(predt, dtrain)
            return score

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
            eval_metric=custom_eval if eval_metrics is not None else None,
            #            tree_method="hist",
            #            multi_strategy="multi_output_tree",
            callbacks=[TimeCallback()],
        )

        all_target_names = [self.params["target_name"]] + self.params.get("auxiliary_targets", [])
        self.model.fit(
            train_data[self.features],
            train_data[all_target_names],
            eval_set=(
                [(val_data[self.features], val_data[all_target_names])]
                if val_data is not None
                else None
            ),
            verbose=100,
        )

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        import numpy as np

        result = self.model.predict(test_data[self.features])
        return np.asarray(result)

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

        # No return, as per abstract method

    def feature_names(self) -> Sequence[str]:
        return self.features

    def target_names(self) -> Sequence[str]:
        return [self.params["target_name"]]
