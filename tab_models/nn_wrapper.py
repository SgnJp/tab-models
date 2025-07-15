from typing import Callable, List, Optional
import tqdm
import pickle
import base64

import numpy as np
from sklearn.preprocessing import StandardScaler
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tab_models.model_wrapper import ModelCallback, ModelWrapper
from tab_models.nn_utils import ScalerImputer, get_loss

import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


def mixup_data(x, y, alpha=1.0):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class LinearDropConnect(nn.Module):
    def __init__(self, in_features, out_features, drop_prob=0.5, bias=True):
        super(LinearDropConnect, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.drop_prob = drop_prob

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.training:
            # Apply DropConnect to weights during training
            mask = (torch.rand_like(self.weight) > self.drop_prob).float()
            weight = self.weight * mask
        else:
            # Scale the weights during evaluation (just like in Dropout)
            weight = self.weight * (1.0 - self.drop_prob)

        return F.linear(x, weight, self.bias)


class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=0.1):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        return x


class FeedforwardNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        dropout_rates,
        input_noise_std,
        input_dropout,
        output_dim=1,
    ):
        super().__init__()
        assert len(hidden_dims) == len(
            dropout_rates
        ), "Each hidden layer must have a corresponding dropout rate"

        layers = [AddGaussianNoise(std=input_noise_std)]
        current_dim = input_dim
        if input_dropout != 0:
            layers.append(nn.Dropout(input_dropout))

        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


class EmptyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise RuntimeError("Should not be called")


class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float16)
        del X
        gc.collect()
        self.y = (
            torch.tensor(y.values, dtype=torch.float16).squeeze(1)
            if y is not None
            else None
        )
        del y
        gc.collect()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].float(), (
            self.y[idx].float().squeeze() if self.y is not None else 0.0
        )


def get_model_outputs(model, data_loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            all_preds.append(outputs.cpu())

    preds = torch.cat(all_preds).numpy()
    return preds


class NNWrapper(ModelWrapper):
    def __init__(self):
        pass

    def __init__(self, params, features, fpath, model_name="nn"):
        if fpath is not None:
            saved_model = torch.load(fpath, map_location=torch.device("cpu"))
            self.features = saved_model["features"]
            self.params = saved_model["params"] if params is None else params
            self.scaler_imputer = ScalerImputer.load(
                saved_model["scaler"], saved_model.get("imputer", None)
            )
            self.scaler_fit = True
        else:
            self.features = features
            self.params = params
            self.scaler_imputer = ScalerImputer()
            self.scaler_fit = False

        self.model = FeedforwardNN(
            input_dim=len(self.features),
            hidden_dims=self.params["hidden_dims"],
            dropout_rates=self.params["dropout_rates"],
            input_noise_std=self.params["input_noise"],
            input_dropout=self.params.get("input_dropout", 0),
            output_dim=len(self.params["auxiliary_targets"]) + 1,
        )

        if fpath is not None:
            self.model.load_state_dict(saved_model["model_state_dict"])
        else:
            self.model.apply(init_weights)

        self.model_name = model_name
        self.model.to(device)

    def load(self, model_dump):
        self.features = model_dump["features"]
        self.params = model_dump["params"]

        scaler_bytes = base64.b64decode(model_dump["scaler"])
        self.scaler_imputer = ScalerImputer.load(
            scaler_bytes, model_dump.get("imputer", None)
        )
        self.model = FeedforwardNN(
            input_dim=len(self.features),
            hidden_dims=self.params["hidden_dims"],
            dropout_rates=self.params["dropout_rates"],
            input_noise_std=self.params.get("input_noise", 0),
            output_dim=len(self.params["auxiliary_targets"]) + 1,
        )
        self.model.load_state_dict(model_dump["model_state_dict"])
        self.mode.to(device)

    def _prepare_data(self, data, fit_scaler):
        X, y = (
            data[self.features],
            data[[self.params["target_name"]] + self.params["auxiliary_targets"]],
        )
        del data

        if fit_scaler:
            self.scaler_imputer.fit(X)
            self.scaler_fit = True

        X = self.scaler_imputer.transform(X, dtype=np.float32)
        ds = TabularDataset(X, y)
        return ds

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        eval_metrics: Optional[Callable] = None,
        callbacks: Optional[List[ModelCallback]] = None,
    ) -> None:

        train_ds = self._prepare_data(train_data, fit_scaler=not self.scaler_fit)
        val_ds = (
            self._prepare_data(val_data, fit_scaler=False)
            if val_data is not None
            else EmptyDataset()
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.params["batch_size"],
            num_workers=self.params.get("num_workers", 1),
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=4 * 4096,
            shuffle=False,
            num_workers=self.params.get("num_workers", 1),
            drop_last=False,
            pin_memory=True,
        )

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=4 * self.params["num_epochs"], T_mult=1
        )

        loss_fn = get_loss(self.params.get("loss_fn", "mse"))
        val_per_era_corr = None
        val_sharpe = None
        best_sharpe = 0.0

        steps_without_improvement = 0
        main_target_loss_factor = 1.0

        for epoch in range(self.params["num_epochs"]):
            self.model.train()
            running_loss = 0.0
            NUM_TOTAL = len(train_loader) // 4

            use_mixup = self.params.get("mixup_alpha", 0.0) > 0.0

            for data_loader_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(
                    device, non_blocking=True
                )

                if use_mixup:
                    inputs, y_a, y_b, lam = mixup_data(
                        inputs, targets, alpha=self.params["mixup_alpha"]
                    )

                optimizer.zero_grad()
                outputs = self.model(inputs)

                if use_mixup:
                    loss = lam * loss_fn(outputs, y_a, main_target_loss_factor) + (
                        1 - lam
                    ) * loss_fn(outputs, y_b, main_target_loss_factor)
                else:
                    loss = loss_fn(outputs, targets, main_target_loss_factor)

                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                if (data_loader_idx + 1) % NUM_TOTAL == 0:
                    val_metrics = {}
                    if len(val_ds) != 0:
                        self.model.eval()
                        val_output = get_model_outputs(self.model, val_loader, "cuda")

                        y_true = val_data[self.params["target_name"]]
                        y_pred = val_output

                        for metric_fn in eval_metrics:
                            name, score, _ = metric_fn(y_pred, y_true)
                            val_metrics[name] = score

                        scheduler.step()

                    print(
                        f"Epoch {epoch+1}/{self.params['num_epochs']}, Train Loss: {running_loss/len(train_ds):.7f}, LR = {optimizer.param_groups[0]['lr']:.5f}, Val metrics: {{'"
                        + ", ".join(
                            [f"{k}: {float(v):.4f}" for k, v in val_metrics.items()]
                        )
                        + "'}}"
                    )

                    running_loss = 0.0
                    self.model.train()

        return val_data

    def _inference_full(self, test_data):
        test_data = test_data[self.features]
        test_data = self.scaler_imputer.transform(test_data)
        loader = DataLoader(
            TabularDataset(test_data),
            batch_size=4 * 4096,
            shuffle=False,
            num_workers=self.params.get("num_workers", 1),
            drop_last=False,
            pin_memory=True,
        )
        return get_model_outputs(self.model, loader, device)

    def predict(self, test_data):
        output = self._inference_full(test_data)
        if output.ndim == 1:
            return output
        elif output.ndim == 2:
            return output[:, 0]
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")

    def dump(self):
        scaler_b64, imputer_b64 = self.scaler_imputer.dump()
        return {
            "scaler": scaler_b64,
            "imputer": imputer_b64,
            "model_state_dict": self.model.state_dict(),
            "params": self.params,
            "features": self.features,
        }

    def save(self, fpath):
        torch.save(self.dump(), fpath)

    def feature_names(self):
        return self.features

    def target_names(self):
        return [self.params["target_name"]] + self.params.get("auxiliary_targets", [])
