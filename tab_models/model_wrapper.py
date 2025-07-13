from abc import ABC, abstractmethod
from typing import List, Sequence, Any, Optional, Callable
import pandas as pd
import numpy as np


class ModelCallback(ABC):
    @abstractmethod
    def after_iteration(self, iter_num: int, model: "ModelWrapper") -> None:
        pass


class ModelWrapper(ABC):
    @abstractmethod
    def save(self, fpath: str) -> None:
        pass

    @abstractmethod
    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        eval_metrics: Optional[Callable] = None,
        callbacks: Optional[List[ModelCallback]] = None,
    ) -> None:
        pass

    @abstractmethod
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def feature_names(self) -> Sequence[str]:
        pass

    @abstractmethod
    def target_names(self) -> Sequence[str]:
        pass
