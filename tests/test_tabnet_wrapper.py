import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add the tab-models directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tab_models"))

from tabnet_wrapper import TabNetWrapper
from tab_models.model_utils import load_model


class TestTabNetWrapper:
    """Simple unit tests for TabNetWrapper"""

    def setup_method(self):
        """Set up test data before each test"""
        # Create simple test data
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        self.features = [f"feature_{i}" for i in range(n_features)]
        self.data = pd.DataFrame(
            {
                "target": np.random.randn(n_samples),
                **{
                    f"feature_{i}": np.random.randn(n_samples)
                    for i in range(n_features)
                },
            }
        )

        self.params = {
            "n_ad": 8,
            "n_steps": 3,
            "gamma": 1.3,
            "lambda_sparse": 1e-3,
            "lr": 0.02,
            "num_epochs": 5,
            "batch_size": 16,
            "virtual_batch_size": 8,
            "target_name": "target",
            "num_val_eras": 1,
            "loss_fn": "mse",
        }

    def test_initialization(self):
        """Test that TabNetWrapper can be initialized"""
        model = TabNetWrapper(
            self.params, self.features, fpath=None, model_name="test_model"
        )
        assert model is not None
        assert model.features == self.features
        assert model.model_name == "test_model"

    def test_feature_names(self):
        """Test that feature_names method returns correct features"""
        model = TabNetWrapper(self.params, self.features, fpath=None)
        assert model.feature_names() == self.features

    def test_fit(self):
        """Test that the model can be fitted"""
        model = TabNetWrapper(self.params, self.features, fpath=None)
        train_data = self.data.iloc[:80]
        val_data = self.data.iloc[80:]
        model.fit(train_data, val_data, eval_metrics=[])
        assert hasattr(model.model, "predict")

    def test_predict(self):
        """Test that the model can make predictions"""
        model = TabNetWrapper(self.params, self.features, fpath=None)
        train_data = self.data.iloc[:80]
        val_data = self.data.iloc[80:]
        model.fit(train_data, val_data, eval_metrics=[])
        predictions = model.predict(self.data)
        assert len(predictions) == len(self.data)
        assert isinstance(predictions, np.ndarray)

    def test_save_method_exists(self):
        """Test that save method exists (even if not implemented)"""
        model = TabNetWrapper(self.params, self.features, fpath=None)
        assert hasattr(model, "save")
        assert callable(model.save)

    def test_save_and_load_model(self):
        """Test saving and loading TabNet model produces identical predictions"""
        model = TabNetWrapper(self.params, self.features, fpath=None)
        train_data = self.data.iloc[:80]
        val_data = self.data.iloc[80:]
        model.fit(train_data, val_data, eval_metrics=[])
        test_data = self.data
        preds_before = model.predict(test_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "tabnet_model_test.bin")
            model.save(fpath)
            loaded_model = load_model(fpath)
            preds_after = loaded_model.predict(test_data)
            assert np.allclose(preds_before, preds_after, atol=1e-5)
