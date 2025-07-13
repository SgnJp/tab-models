import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add the tab-models directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tab_models"))

from nn_wrapper import NNWrapper
from tab_models.model_utils import load_model

# Avoid multiprocessing in tests
NUM_WORKERS = 0


class TestNNWrapper:
    """Simple unit tests for NNWrapper"""

    def setup_method(self):
        """Set up test data before each test"""
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
            "hidden_dims": [8, 8],
            "dropout_rates": [0.1, 0.1],
            "input_noise_std": 0.0,
            "input_dropout": 0.0,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "batch_size": 16,
            "num_epochs": 2,
            "auxiliary_targets": [],
            "mixup_alpha": 0.0,
            "loss": {
                "name": "mse",
                "main_target_init_factor": 1.0,
                "main_target_factor_increment": 1.0,
                "main_target_max_factor": 1.0,
            },
            "target_name": "target",
        }

    def test_initialization(self):
        model = NNWrapper(
            self.params, self.features, fpath=None, model_name="test_model"
        )
        assert model is not None
        assert model.features == self.features
        assert model.model_name == "test_model"

    def test_feature_names(self):
        model = NNWrapper(self.params, self.features, fpath=None)
        assert model.feature_names() == self.features

    def test_fit(self):
        model = NNWrapper(self.params, self.features, fpath=None)
        model.fit(self.data)
        assert hasattr(model.model, "forward")

    def test_predict(self):
        model = NNWrapper(self.params, self.features, fpath=None)
        model.fit(self.data)
        predictions = model.predict(self.data)
        assert len(predictions) == len(self.data)
        assert isinstance(predictions, np.ndarray)

    def test_save_method_exists(self):
        model = NNWrapper(self.params, self.features, fpath=None)
        assert hasattr(model, "save")
        assert callable(model.save)

    def test_save_and_load_model(self):
        model = NNWrapper(self.params, self.features, fpath=None)
        model.fit(self.data)
        test_data = self.data
        preds_before = model.predict(test_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "nn_model_test.pth")
            model.save(fpath)
            loaded_model = load_model(fpath)
            preds_after = loaded_model.predict(test_data)
            assert np.allclose(preds_before, preds_after, atol=1e-5)
