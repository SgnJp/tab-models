import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add the tab-models directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tab-models'))

from lgbm_wrapper import LGBMWrapper


class TestLGBMWrapper:
    """Simple unit tests for LGBMWrapper"""
    
    def setup_method(self):
        """Set up test data before each test"""
        # Create simple test data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        self.features = [f'feature_{i}' for i in range(n_features)]
        self.data = pd.DataFrame({
            'target': np.random.randn(n_samples),
            'era': np.random.randint(0, 5, n_samples),
            **{f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)}
        })
        
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 3,
            'num_iterations': 10  # Small number for quick tests
        }
    
    def test_initialization(self):
        """Test that LGBMWrapper can be initialized"""
        model = LGBMWrapper(self.params, self.features, fpath=None, model_name="test_model")
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
        # This should not raise an exception
        model.fit(self.data)
        assert hasattr(model.model, 'feature_importance')
    
    def test_predict(self):
        """Test that the model can make predictions"""
        model = LGBMWrapper(self.params, self.features, fpath=None)
        model.fit(self.data)
        
        # Test predictions
        predictions = model.predict(self.data)
        assert len(predictions) == len(self.data)
        assert isinstance(predictions, np.ndarray)
    
    def test_save_method_exists(self):
        """Test that save method exists (even if not implemented)"""
        model = LGBMWrapper(self.params, self.features, fpath=None)
        assert hasattr(model, 'save')
        assert callable(model.save) 