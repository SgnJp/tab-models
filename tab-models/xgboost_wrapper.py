from xgboost import XGBRegressor
from xgboost.callback import TrainingCallback
from model_wrapper import ModelWrapper
from utils import log_timing
import os
import time


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

        if self.fpath != None:
            self.model = XGBRegressor()
            self.model.load_model(fpath)
            self.features = self.model.feature_names_in_.tolist()
        else:
            self.model = XGBRegressor(
                objective=params['objective'],
                n_estimators=params['n_estimators'],
                colsample_bynode=params['colsample_bynode'],
                learning_rate=params['learning_rate'],
                max_depth=params['max_depth'],
                device='cuda:0',
                subsample=params['subsample'],
                min_child_weight=params['min_child_weight'],
                gamma=params['gamma'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                callbacks=[TimeCallback(), CheckpointCallback(1000, model_name)] 
            )
            self.features = features

    def fit(self, train_data):
        if self.fpath is not None:
            self.model.fit(train_data[self.features], train_data["target"], xgb_model=os.path.join("checkpoints", self.fpath))
        else:
            self.model.fit(train_data[self.features], train_data["target"])

    @log_timing()
    def predict(self, test_data):
        return self.model.predict(test_data[self.features])

    def save(self, fpath):
        pass

    def feature_names(self):
        return self.features