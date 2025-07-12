import os
import time
import joblib
import gc
from utils import log_timing

from model_wrapper import ModelWrapper
from lightgbm.callback import CallbackEnv
import lightgbm as lgb


def checkpoint_callback(model_name="lgbm_model", save_dir="checkpoints", interval=100):
    os.makedirs(save_dir, exist_ok=True)

    def _callback(env: CallbackEnv):
        if env.iteration % interval == 0 and env.iteration > 0:
            filename = f"{model_name}_iter_{env.iteration}.pkl"
            path = os.path.join(save_dir, filename)
            joblib.dump(env.model, path)
            print(f"[{env.iteration}] Saved checkpoint to: {path}")

    _callback.order = 10
    return _callback


def time_callback(interval=100):
    start_time = time.time()

    def _callback(env: CallbackEnv):
        elapsed = time.time() - start_time
        if env.iteration % interval != 0:
            return

        print(f"[{env.iteration}] Time elapsed: {elapsed:.2f} seconds")

    _callback.order = 30  # order of execution, lower runs earlier
    return _callback


class LGBMWrapper(ModelWrapper):
    def __init__(self, params, features, fpath, model_name="lgbm_model"):
        self.fpath = fpath
        self.features = features
        self.model_name = model_name
        self.params = params
        self.model = None

        if fpath is not None:
            self.model = joblib.load(fpath)
            self.features = self.model.feature_name()

        if self.params is not None:
            self.params["num_leaves"] = 2 ** params["max_depth"] - 1
            self.params["device"] = "gpu"
            self.params["verbosity"] = -1

    def fit(self, train_data):
        train_set = lgb.Dataset(train_data[self.features], train_data["target"], free_raw_data=True)
        del train_data
        gc.collect()

        self.model = lgb.train(
            self.params,
            train_set,
            init_model=self.model,
            callbacks=[time_callback(), checkpoint_callback(model_name=self.model_name, interval=1000)])


    @log_timing()
    def predict(self, test_data):
        return self.model.predict(test_data[self.features])

    def save(self, fpath):
        pass

    def feature_names(self):
        return self.features
