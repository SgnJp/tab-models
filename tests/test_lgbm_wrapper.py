import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add the tab-models directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tab_models"))

from lgbm_wrapper import LGBMWrapper
from tab_models.callbacks import CheckpointCallback
from tab_models.model_utils import load_model


class TestLGBMWrapper:
    """Simple unit tests for LGBMWrapper"""

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
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "max_depth": 3,
            "num_iterations": 10,
            "target_name": "target1",
        }

    def test_initialization(self):
        """Test that LGBMWrapper can be initialized"""
        model = LGBMWrapper(
            self.params, self.features, fpath=None, model_name="test_model"
        )
        assert model is not None
        assert model.features == self.features
        assert model.model_name == "test_model"

    def test_feature_names(self):
        """Test that feature_names method returns correct features"""
        model = LGBMWrapper(self.params, self.features, fpath=None)
        assert model.feature_names() == self.features

    def test_fit(self):
        """Test that the model can be fitted"""
        model = LGBMWrapper(self.params, self.features, fpath=None)
        train_data = self.data.iloc[:80]
        val_data = self.data.iloc[80:]
        model.fit(train_data, val_data, eval_metrics=None, callbacks=[])
        assert hasattr(model.model, "feature_importance")

    def test_predict(self):
        """Test that the model can make predictions"""
        model = LGBMWrapper(self.params, self.features, fpath=None)
        train_data = self.data.iloc[:80]
        val_data = self.data.iloc[80:]
        model.fit(train_data, val_data, eval_metrics=None, callbacks=[])
        predictions = model.predict(self.data)
        assert len(predictions) == len(self.data)
        assert isinstance(predictions, np.ndarray)

    def test_save_method_exists(self):
        """Test that save method exists (even if not implemented)"""
        model = LGBMWrapper(self.params, self.features, fpath=None)
        assert hasattr(model, "save")
        assert callable(model.save)

    def test_checkpoint_callback_with_fit(self):
        """Test that CheckpointCallback saves checkpoints when used in model.fit"""
        model = LGBMWrapper(self.params, self.features, fpath=None)
        train_data = self.data.iloc[:80]
        val_data = self.data.iloc[80:]
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CheckpointCallback(
                path_to_checkpoints=tmpdir, n_iterations=5, base_name="test_model"
            )
            model.fit(train_data, val_data, eval_metrics=None, callbacks=[callback])
            files = os.listdir(tmpdir)
            assert any("test_model_4.bin" in f for f in files)
            assert any("test_model_9.bin" in f for f in files)

    def test_save_and_load_model(self):
        """Test saving and loading LGBM model produces identical predictions"""
        model = LGBMWrapper(self.params, self.features, fpath=None)
        train_data = self.data.iloc[:80]
        val_data = self.data.iloc[80:]
        model.fit(train_data, val_data, eval_metrics=None, callbacks=[])
        test_data = self.data
        preds_before = model.predict(test_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "lgbm_model_test.bin")
            model.save(fpath)
            loaded_model = load_model(fpath)
            preds_after = loaded_model.predict(test_data)
            assert (preds_before == preds_after).all()
