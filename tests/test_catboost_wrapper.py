import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add the tab-models directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tab_models"))

from catboost_wrapper import CatBoostWrapper
from tab_models.model_utils import load_model


class TestCatBoostWrapper:
    """Simple unit tests for CatBoostWrapper"""

    def setup_method(self):
        """Set up test data before each test"""
        # Create simple test data
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        self.features = [f"feature_{i}" for i in range(n_features)]
        self.data = pd.DataFrame(
            {
                "target1": np.random.randn(n_samples),
                "era": np.random.randint(0, 5, n_samples),
                **{
                    f"feature_{i}": np.random.randn(n_samples)
                    for i in range(n_features)
                },
            }
        )

        self.params = {
            "loss_function": "RMSE",
            "iterations": 10,
            "learning_rate": 0.1,
            "depth": 3,
            "verbose": 0,
            "random_seed": 42,
            "target_name": "target1",
        }

    def test_initialization(self):
        """Test that CatBoostWrapper can be initialized"""
        model = CatBoostWrapper(
            self.params, self.features, fpath=None, model_name="test_model"
        )
        assert model is not None
        assert model.features == self.features
        assert model.model_name == "test_model"

    def test_feature_names(self):
        """Test that feature_names method returns correct features"""
        model = CatBoostWrapper(self.params, self.features, fpath=None)
        assert model.feature_names() == self.features

    def test_fit(self):
        """Test that the model can be fitted"""
        model = CatBoostWrapper(self.params, self.features, fpath=None)
        train_data = self.data.iloc[:80]
        val_data = self.data.iloc[80:]
        model.fit(train_data, val_data, eval_metrics=None, callbacks=None)
        assert hasattr(model.model, "feature_names_")

    def test_predict(self):
        """Test that the model can make predictions"""
        model = CatBoostWrapper(self.params, self.features, fpath=None)
        train_data = self.data.iloc[:80]
        val_data = self.data.iloc[80:]
        model.fit(train_data, val_data, eval_metrics=None, callbacks=None)
        predictions = model.predict(self.data)
        assert len(predictions) == len(self.data)
        assert isinstance(predictions, np.ndarray)

    def test_save_method_exists(self):
        """Test that save method exists (even if not implemented)"""
        model = CatBoostWrapper(self.params, self.features, fpath=None)
        assert hasattr(model, "save")
        assert callable(model.save)

    def test_save_and_load_model(self):
        """Test saving and loading CatBoost model produces identical predictions"""
        model = CatBoostWrapper(self.params, self.features, fpath=None)
        train_data = self.data.iloc[:80]
        val_data = self.data.iloc[80:]
        model.fit(train_data, val_data, eval_metrics=None, callbacks=None)
        test_data = self.data
        preds_before = model.predict(test_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "catboost_model_test.cbm")
            model.save(fpath)
            loaded_model = load_model(fpath)
            preds_after = loaded_model.predict(test_data)
            assert np.allclose(preds_before, preds_after)
