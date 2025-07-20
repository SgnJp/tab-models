import pandas as pd
import numpy as np
import tempfile
import os
import sys
import pytest

# Add the tab-models directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tab_models"))

from tab_models.xgboost_wrapper import XGBoostWrapper


class TestXGBoostWrapper:
    """Simple unit tests for XGBoostWrapper"""

    @pytest.fixture(autouse=True)
    def setup_method(self, sample_data, xgboost_params):
        self.data, self.features = sample_data
        self.params = xgboost_params

    def test_initialization(self):
        model = XGBoostWrapper(
            self.params, self.features, fpath=None, model_name="test_model"
        )
        assert model is not None
        assert model.features == self.features
        assert model.model_name == "test_model"

    def test_feature_names(self):
        model = XGBoostWrapper(self.params, self.features, fpath=None)
        assert model.feature_names() == self.features

    def test_fit(self):
        model = XGBoostWrapper(self.params, self.features, fpath=None)
        model.fit(self.data)
        assert hasattr(model.model, "feature_names_in_")

    def test_predict(self):
        model = XGBoostWrapper(self.params, self.features, fpath=None)
        model.fit(self.data)
        predictions = model.predict(self.data)
        assert len(predictions) == len(self.data)
        assert isinstance(predictions, np.ndarray)

    def test_save_method_exists(self):
        model = XGBoostWrapper(self.params, self.features, fpath=None)
        assert hasattr(model, "save")
        assert callable(model.save)

    def test_save_and_load_model(self):
        model = XGBoostWrapper(self.params, self.features, fpath=None)
        model.fit(self.data)
        test_data = self.data
        preds_before = model.predict(test_data)
        from tab_models.model_utils import load_model
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "xgb_model_test.pkl")
            model.save(fpath)
            loaded_model = load_model(fpath)
            preds_after = loaded_model.predict(test_data)
            assert np.allclose(preds_before, preds_after, atol=1e-5)

    def test_fit_with_mock_metric_and_eval_frequency(self):
        """Test that a mock metric is called with eval_frequency during fit"""
        from unittest.mock import MagicMock

        model = XGBoostWrapper(self.params, self.features, fpath=None)
        # Use all data for training for simplicity, as in other tests
        train_data = self.data
        val_data = self.data  # XGBoostWrapper.fit expects both train and val

        # Create a mock metric
        mock_metric = MagicMock(return_value=("Mock", 0.0, False))

        # Fit the model with the mock metric and eval_frequency
        model.fit(
            train_data,
            val_data=val_data,
            eval_metrics=[mock_metric],
            eval_frequency=2,
            callbacks=[],
        )

        # Check that the mock metric was called at least once
        assert mock_metric.call_count > 0

    def test_checkpoint_callback_with_fit(self):
        """Test that CheckpointCallback saves checkpoints when used in model.fit"""
        from tab_models.callbacks import CheckpointCallback
        import tempfile
        import os

        model = XGBoostWrapper(self.params, self.features, fpath=None)
        train_data = self.data
        val_data = self.data
        with tempfile.TemporaryDirectory() as tmpdir:
            # XGBoost CheckpointCallback saves to 'checkpoints' dir, so create it inside tmpdir
            checkpoints_dir = os.path.join(tmpdir, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
            callback = CheckpointCallback(
                path_to_checkpoints=checkpoints_dir,
                n_iterations=5,
                base_name="test_model",
            )
            model.fit(
                train_data,
                val_data=val_data,
                eval_metrics=[],
                callbacks=[callback],
                eval_frequency=1,
            )
            # Check that at least one checkpoint file was created
            files = os.listdir(checkpoints_dir)
            assert any("test_model_4.bin" in f for f in files)
            assert any("test_model_9.bin" in f for f in files)
