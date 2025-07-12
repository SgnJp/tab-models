import json
import zipfile

import pickle
import base64

import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
import torch.optim as optim

from train import calculate_metrics
from utils import log_timing, write_json
from model_wrapper import ModelWrapper
from const import NUM_WORKERS
from pytorch_tabnet.tab_model import TabNetRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
from pytorch_tabnet.metrics import Metric
from nn_wrapper import spearman_loss, weighted_mse_loss, custom_loss


def noise_augmentation(x: torch.Tensor, y):
    noise = torch.randn_like(x) * 0.1  # mean=0, std=0.1
    x_noisy = x + noise
    return x_noisy, y


def get_numerai_metric(val_data):
    class NumeraiCustomMetric(Metric):
        def __init__(self):
            #self.val_meta = load_data([], TRAIN_PATH, VALIDATION_PATH, 1000, 1113, target_name="target")

    #        self.meta_model = pd.read_parquet(META_MODEL_PATH)
    #        self.val_meta["meta_model"] = self.meta_model["numerai_meta_model"]
    #        self.val_meta = load_data([], TRAIN_PATH, VALIDATION_PATH, 625, 780, target_name="target")

            self.val_meta = val_data.copy()
            self._name = "corr"
            self._maximize = True

        def __call__(self, y_true, y_pred):
            self.val_meta["prediction"] = y_pred[:,0]
            metrics = calculate_metrics(self.val_meta, False)
            print (metrics)
            return 100*metrics['per_era_corr']

    return NumeraiCustomMetric

class TabNetWrapper(ModelWrapper):
    file_to_add = 'tabnet_wrapper_params.json'  # File you want to add

    def __init__(self):
        pass

    def __init__(self, params, features, fpath, model_name="tabnet"):
        """
        if fpath is not None:
            saved_model = torch.load(fpath)
            self.features = saved_model["features"]
            self.params = saved_model["params"] if params is None else params

            scaler_bytes = base64.b64decode(saved_model["scaler"])
            self.scaler = pickle.loads(scaler_bytes)
            self.scaler_fit = True
        else:
        """
        if fpath is None:
            self.features = features
            self.params = params
            self.scaler = StandardScaler()
            self.scaler_fit = False
        else:
            with zipfile.ZipFile(fpath, 'r') as zipf:
                with zipf.open(self.file_to_add) as json_file:
                    data = json.load(json_file)
            self.features = data["features"]
            self.params = data["params"]
            scaler_bytes = base64.b64decode(data["scaler"])
            self.scaler = pickle.loads(scaler_bytes)
            self.scaler_fit = True

        self.model = TabNetRegressor(
            n_d=self.params["n_ad"], n_a=self.params["n_ad"],
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
#        self.model.to(device)

    def load(self, model_dump):
        pass

    def _prepare_data(self, data, fit_scaler):
        X, y = data[self.features], data[["target"] + self.params["auxiliary_targets"]]

        if fit_scaler:
            self.scaler.fit(X[::10])
            self.scaler_fit = True

        X = X.astype(np.float16, copy=False)
        batch_size = 50000
        for i in range(0, len(X), batch_size):
            X[i:i+batch_size] = self.scaler.transform(X[i:i+batch_size]).astype(np.float16, copy=False)

#        X[:] = self.scaler.transform(X).astype(np.float32)

        return X, y 

    def _split_train_val(self, data):
        data["era"] = data["era"].astype(int)

        val_era_start = data.era.max() - self.params["num_val_eras"]
        train_era_end = val_era_start - 4

        print (f"Training on data before: {train_era_end}")
        train_data = data[data.era < train_era_end]
        print (f"Validating on data starting with: {val_era_start}")
        val_data = data[data.era >= val_era_start]

        val_meta = val_data[[self.params["target_name"], "era"]].copy()
        return train_data, val_data, val_meta

    def fit(self, train_data):
        train_data, val_data, val_meta = self._split_train_val(train_data)

        X_train, y_train = self._prepare_data(train_data, fit_scaler=not self.scaler_fit)
        del train_data
        X_val, y_val = self._prepare_data(val_data, fit_scaler=False)

        self.model.augmentation = noise_augmentation
        self.model.fit(
            X_train=X_train.values, y_train=y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            eval_metric=[get_numerai_metric(val_data[["era", "target"]])],
            max_epochs=self.params['num_epochs'],
            patience=10,
            batch_size=self.params["batch_size"], virtual_batch_size=self.params["virtual_batch_size"],
            loss_fn=spearman_loss,
            compute_importance=False,
            num_workers=NUM_WORKERS,
        )
        self.save(f"checkpoints/{self.model_name}")

    @log_timing()
    def predict(self, test_data):
        test_data = test_data[self.features]
        test_data = test_data.astype(np.float32, copy=False)
        test_data[:] = self.scaler.transform(test_data).astype(np.float32, copy=False)

        return self.model.predict(test_data.values)[:, 0]


    def dump(self):
        scaler_bytes = pickle.dumps(self.scaler)
        scaler_b64 = base64.b64encode(scaler_bytes).decode("utf-8")
        return {
                "scaler": scaler_b64,
                "params": self.params,
                "features": self.features,
            }

    def save(self, fpath):
        self.model.save_model(fpath)
        zip_path = f'{fpath}.zip'  # Path to your existing zip file

        write_json(self.dump(), self.file_to_add)
        with zipfile.ZipFile(zip_path, 'a') as zipf:
            zipf.write(self.file_to_add, arcname=self.file_to_add)


    def feature_names(self):
        return self.features
