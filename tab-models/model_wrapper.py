from abc import ABC, abstractmethod

class ModelWrapper(ABC):
    @abstractmethod
    def save(self, fpath):
        pass

    @abstractmethod
    def fit(self, train_data):
        pass

    @abstractmethod
    def predict(self, test_data):
        pass

    @abstractmethod
    def feature_names(self):
        pass
