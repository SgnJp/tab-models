import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add the tab-models directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tab-models'))

from xgboost_wrapper import XGBoostWrapper


class TestXGBoostWrapper:
    """Simple unit tests for XGBoostWrapper"""
    
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
            'objective': 'reg:squarederror',
            'n_estimators': 10,  # Small number for quick tests
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bynode': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    
    def test_initialization(self):
        """Test that XGBoostWrapper can be initialized"""
        model = XGBoostWrapper(self.params, self.features, fpath=None, model_name="test_model")
        assert model is not None
        assert model.features == self.features
        assert model.model_name == "test_model"
    
    def test_feature_names(self):
        """Test that feature_names method returns correct features"""
        model = XGBoostWrapper(self.params, self.features, fpath=None)
        assert model.feature_names() == self.features
    
    def test_fit(self):
        """Test that the model can be fitted"""
        model = XGBoostWrapper(self.params, self.features, fpath=None)
        # This should not raise an exception
        model.fit(self.data)
        assert hasattr(model.model, 'feature_importances_')
    
    def test_predict(self):
        """Test that the model can make predictions"""
        model = XGBoostWrapper(self.params, self.features, fpath=None)
        model.fit(self.data)
        
        # Test predictions
        predictions = model.predict(self.data)
        assert len(predictions) == len(self.data)
        assert isinstance(predictions, np.ndarray)
    
    def test_save_method_exists(self):
        """Test that save method exists (even if not implemented)"""
        model = XGBoostWrapper(self.params, self.features, fpath=None)
        assert hasattr(model, 'save')
        assert callable(model.save) 