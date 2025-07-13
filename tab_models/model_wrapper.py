from abc import ABC, abstractmethod
from typing import List, Sequence, Any
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
        val_data: pd.DataFrame = None,
        eval_metrics: Any = None,
        callbacks: List[Any] = [],
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
