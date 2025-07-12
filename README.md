# Tab-Models

A comprehensive Python library providing unified wrappers for popular machine learning models designed specifically for tabular data. This library simplifies the process of training and deploying various ML models with a consistent interface.

## Features

- **Unified Interface**: Consistent API across different model types
- **Multiple Model Support**: XGBoost, LightGBM, Neural Networks, and TabNet
- **GPU Acceleration**: Support for CUDA-enabled training
- **Checkpointing**: Automatic model checkpointing during training
- **Custom Loss Functions**: Built-in support for Spearman correlation loss
- **Easy Deployment**: Simple save/load functionality for model persistence

## Supported Models

- **XGBoost**: Gradient boosting with GPU support
- **LightGBM**: Fast gradient boosting framework
- **Neural Networks**: Custom PyTorch implementations with advanced features
- **TabNet**: Interpretable deep learning for tabular data

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### Development Installation

```bash
pip install -e .[dev]
```

### GPU Support (Optional)

```bash
pip install -e .[gpu]
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from xgboost_wrapper import XGBoostWrapper
from lgbm_wrapper import LGBMWrapper
from nn_wrapper import NNWrapper
from tabnet_wrapper import TabNetWrapper

# Load your data
data = pd.read_csv('your_data.csv')
features = [col for col in data.columns if col not in ['target', 'era']]

# XGBoost Example
xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 1000,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bynode': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1
}

xgb_model = XGBoostWrapper(xgb_params, features, fpath=None, model_name="xgb_model")
xgb_model.fit(data)
predictions = xgb_model.predict(test_data)
```

### LightGBM Example

```python
lgbm_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'max_depth': 6
}

lgbm_model = LGBMWrapper(lgbm_params, features, fpath=None, model_name="lgbm_model")
lgbm_model.fit(data)
predictions = lgbm_model.predict(test_data)
```

### Neural Network Example

```python
nn_params = {
    'hidden_dims': [512, 256, 128, 64],
    'dropout_rates': [0.3, 0.3, 0.2, 0.1],
    'input_noise_std': 0.1,
    'input_dropout': 0.1,
    'learning_rate': 1e-3,
    'batch_size': 1024,
    'num_epochs': 100,
    'loss_name': 'spearman'
}

nn_model = NNWrapper(nn_params, features, fpath=None, model_name="nn_model")
nn_model.fit(data)
predictions = nn_model.predict(test_data)
```

### TabNet Example

```python
tabnet_params = {
    'n_ad': 64,
    'n_steps': 3,
    'gamma': 1.3,
    'lambda_sparse': 1e-4,
    'lr': 2e-2,
    'num_epochs': 100,
    'batch_size': 1024,
    'virtual_batch_size': 128,
    'auxiliary_targets': ['target_aux1', 'target_aux2'],
    'target_name': 'target',
    'num_val_eras': 10
}

tabnet_model = TabNetWrapper(tabnet_params, features, fpath=None, model_name="tabnet_model")
tabnet_model.fit(data)
predictions = tabnet_model.predict(test_data)
```

## Model Wrapper Interface

All model wrappers inherit from `ModelWrapper` and provide a consistent interface:

```python
class ModelWrapper(ABC):
    @abstractmethod
    def fit(self, train_data):
        """Train the model on the provided data"""
        pass
    
    @abstractmethod
    def predict(self, test_data):
        """Generate predictions on test data"""
        pass
    
    @abstractmethod
    def save(self, fpath):
        """Save the model to disk"""
        pass
    
    @abstractmethod
    def feature_names(self):
        """Return the feature names used by the model"""
        pass
```

## Advanced Features

### Custom Loss Functions

The library includes custom loss functions optimized for financial prediction tasks:

```python
# Spearman correlation loss
spearman_loss(pred, target, factor=0.1)

# Weighted MSE loss
weighted_mse_loss(pred, target, factor=0.1)

# Custom combined loss
custom_loss(pred, target, factor=0.1)  # 25% Spearman + 75% MSE
```

### Checkpointing

Models automatically save checkpoints during training:

```python
# Checkpoints are saved to the 'checkpoints' directory
# Format: {model_name}_iter{iteration}.{extension}
```

### GPU Support

All models support GPU acceleration when available:

```python
# XGBoost GPU
xgb_params['device'] = 'cuda:0'

# LightGBM GPU
lgbm_params['device'] = 'gpu'

# PyTorch models automatically detect CUDA
```

## Data Format

The library expects tabular data in pandas DataFrame format with the following structure:

```python
# Required columns
data = pd.DataFrame({
    'target': [...],      # Target variable
    'era': [...],         # Era/period identifier
    'feature_1': [...],   # Feature columns
    'feature_2': [...],
    # ... more features
})
```

## Model Persistence

### Saving Models

```python
# Save trained model
model.save("models/my_model")
```

### Loading Models

```python
# Load pre-trained model
model = XGBoostWrapper(None, None, fpath="models/my_model.json")
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
```

### Linting

```bash
flake8 .
```

### Type Checking

```bash
mypy .
```

## Requirements

- Python >= 3.8
- CUDA support (optional, for GPU acceleration)
- See `requirements.txt` for full dependency list

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Support

For issues and questions, please open an issue on GitHub. 