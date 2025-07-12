import pytest
import sys
import os

# Add the tab-models directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tab-models'))

from model_wrapper import ModelWrapper


class TestModelWrapper:
    """Simple unit tests for the base ModelWrapper class"""
    
    def test_model_wrapper_is_abstract(self):
        """Test that ModelWrapper is an abstract base class"""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            ModelWrapper()
    
    def test_abstract_methods_exist(self):
        """Test that all required abstract methods are defined"""
        # Check that the abstract methods exist in the class
        assert hasattr(ModelWrapper, 'save')
        assert hasattr(ModelWrapper, 'fit')
        assert hasattr(ModelWrapper, 'predict')
        assert hasattr(ModelWrapper, 'feature_names')
        
        # Check that they are abstract methods
        assert ModelWrapper.save.__isabstractmethod__
        assert ModelWrapper.fit.__isabstractmethod__
        assert ModelWrapper.predict.__isabstractmethod__
        assert ModelWrapper.feature_names.__isabstractmethod__ 