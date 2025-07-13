import pandas as pd
import numpy as np
import tempfile
import os
import sys
import pytest

# Add the tab-models directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tab_models"))

from xgboost_wrapper import XGBoostWrapper


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
