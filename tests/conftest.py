import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the tab-models directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tab-models"))


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 50  # Small dataset for quick tests
    n_features = 3

    features = [f"feature_{i}" for i in range(n_features)]
    data = pd.DataFrame(
        {
            "target": np.random.randn(n_samples),
            **{f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)},
        }
    )

    return data, features


@pytest.fixture
def xgboost_params():
    """XGBoost parameters for testing"""
    return {
        "objective": "reg:squarederror",
        "n_estimators": 5,  # Very small for quick tests
        "learning_rate": 0.1,
        "max_depth": 2,
        "subsample": 0.8,
        "colsample_bynode": 0.8,
        "min_child_weight": 1,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "target_name": "target",
    }


@pytest.fixture
def lgbm_params():
    """LightGBM parameters for testing"""
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 7,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "max_depth": 2,
        "num_iterations": 5,  # Very small for quick tests
    }
