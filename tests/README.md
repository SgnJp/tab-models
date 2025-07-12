# Tests for Tab-Models

This directory contains unit tests for the tab-models library.

## Running Tests

### Using the test runner script
```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py -v

# Run a specific test file
python run_tests.py -t tests/test_xgboost_wrapper.py
```

### Using pytest directly
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_xgboost_wrapper.py

# Run tests matching a pattern
pytest -k "test_fit"
```

## Test Structure

- `test_xgboost_wrapper.py` - Tests for XGBoostWrapper
- `test_lgbm_wrapper.py` - Tests for LGBMWrapper  
- `test_model_wrapper.py` - Tests for the base ModelWrapper class
- `test_utils.py` - Tests for utility functions
- `conftest.py` - Shared test fixtures and configuration

## Test Features

- **Simple and Fast**: Tests use small datasets and minimal iterations for quick execution
- **Isolated**: Each test is independent and doesn't rely on external data
- **Comprehensive**: Tests cover initialization, fitting, prediction, and basic functionality
- **No Dependencies**: Tests don't require external data files or complex setup

## Adding New Tests

1. Create a new test file following the naming convention `test_*.py`
2. Import the module you want to test
3. Create test classes that inherit from `unittest.TestCase` or use pytest-style functions
4. Use the fixtures from `conftest.py` for common test data
5. Run the tests to ensure they pass 