from .model_wrapper import ModelWrapper


def load_model(fpath) -> ModelWrapper:
    if "lgbm" in fpath:
        from .lgbm_wrapper import LGBMWrapper

        return LGBMWrapper(None, None, fpath)
    elif "xgb" in fpath:
        from .xgboost_wrapper import XGBoostWrapper

        return XGBoostWrapper(None, None, fpath)
    elif "nn" in fpath:
        from .nn_wrapper import NNWrapper

        return NNWrapper(None, None, fpath)
    elif "tabnet" in fpath:
        from .tabnet_wrapper import TabNetWrapper

        return TabNetWrapper(None, None, fpath)
    else:
        raise RuntimeError(f"Not supported model type: {fpath}")


def get_model(params, features, fpath=None, model_name="") -> ModelWrapper:
    if params["model"] == "xgb":
        from .xgboost_wrapper import XGBoostWrapper

        return XGBoostWrapper(params, features, fpath, f"xgb_{model_name}")
    elif params["model"] == "lgbm":
        from .lgbm_wrapper import LGBMWrapper

        return LGBMWrapper(params, features, fpath, f"lgbm_{model_name}")
    elif params["model"] == "nn":
        from .nn_wrapper import NNWrapper

        return NNWrapper(params, features, fpath, f"nn_{model_name}")
    elif params["model"] == "tabnet":
        from .tabnet_wrapper import TabNetWrapper

        model = TabNetWrapper(params, features, fpath, f"tabnet_{model_name}")
        print("mid", id(model))
        return model

    raise RuntimeError(f"Not known model: {params['model']}")
