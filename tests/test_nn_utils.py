import pytest
import numpy as np
import torch
import sys
import os

# Add the tab-models directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tab_models"))

from nn_utils import (
    spearman_loss,
    weighted_mse_loss,
    weighted_huber_loss,
    custom_loss,
    get_loss,
    ScalerImputer,
)


def test_spearman_loss_basic():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    loss = spearman_loss(pred, target)
    assert isinstance(loss, torch.Tensor) or isinstance(loss, float)


def test_weighted_mse_loss_basic():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    loss = weighted_mse_loss(pred, target)
    assert torch.isfinite(loss)


def test_weighted_huber_loss_basic():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    loss = weighted_huber_loss(pred, target)
    assert torch.isfinite(loss)


def test_custom_loss_basic():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    loss = custom_loss(pred, target)
    assert torch.isfinite(loss)


def test_get_loss():
    assert get_loss("spearman") == spearman_loss
    assert get_loss("mse") == weighted_mse_loss
    assert get_loss("huber") == weighted_huber_loss
    with pytest.raises(AssertionError):
        get_loss("unknown")


def test_losses_with_nans():
    pred = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
    target = torch.tensor([[1.0, 2.0], [float("nan"), 4.0]])
    # Should not raise
    weighted_mse_loss(pred, target)
    weighted_huber_loss(pred, target)
    custom_loss(pred, target)


def test_spearman_loss_weights_sum_to_one():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    factor = 0.2
    import torchsort

    # Check that weights sum to 1
    num_extra_targets = 0 if len(pred.shape) == 1 else pred.shape[1] - 1
    weights = np.insert(
        (1 - factor) * np.ones(num_extra_targets) / num_extra_targets, 0, factor, axis=0
    )
    assert np.isclose(weights.sum(), 1.0)


def test_scaler_imputer_fit_transform():
    import pandas as pd

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [4.0, 5.0, np.nan, 7.0]})
    scaler_imputer = ScalerImputer()
    scaler_imputer.fit(X)
    X_trans = scaler_imputer.transform(X)
    assert not np.isnan(X_trans.values).any()
    assert X_trans.shape == X.shape


def test_scaler_imputer_dump_load():
    import pandas as pd

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [4.0, 5.0, np.nan, 7.0]})
    scaler_imputer = ScalerImputer()
    scaler_imputer.fit(X)
    X_trans1 = scaler_imputer.transform(X)
    scaler_b64, imputer_b64 = scaler_imputer.dump()
    loaded = ScalerImputer.load(scaler_b64, imputer_b64)
    X_trans2 = loaded.transform(X)
    np.testing.assert_allclose(X_trans1, X_trans2)
    assert X_trans2.shape == X.shape
