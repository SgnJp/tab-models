import pytest
import tempfile
import json
import time
import sys
import os

# Add the tab-models directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tab-models'))

from utils import log_timing, write_json


class TestUtils:
    """Simple unit tests for utility functions"""
    
    def test_log_timing_decorator(self):
        """Test that log_timing decorator works correctly"""
        @log_timing()
        def test_function():
            time.sleep(0.01)  # Small delay to measure
            return "test_result"
        
        # Should not raise an exception and should return the result
        result = test_function()
        assert result == "test_result"
    
    def test_write_json(self):
        """Test that write_json function works correctly"""
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            write_json(test_data, temp_file)
            
            # Read back the file and verify content
            with open(temp_file, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == test_data
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_write_json_with_nested_data(self):
        """Test write_json with more complex nested data"""
        test_data = {
            "model_params": {
                "learning_rate": 0.1,
                "max_depth": 6,
                "n_estimators": 100
            },
            "features": ["feature_1", "feature_2", "feature_3"],
            "metrics": [0.85, 0.87, 0.89]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            write_json(test_data, temp_file)
            
            # Read back the file and verify content
            with open(temp_file, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == test_data
            assert loaded_data["model_params"]["learning_rate"] == 0.1
            assert len(loaded_data["features"]) == 3
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file) 