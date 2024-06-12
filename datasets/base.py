from abc import ABC, abstractmethod

class BaseDataLoader(ABC):
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.transform = None
        self.train_loader = self.test_loader = None

    @abstractmethod
    def set_transform(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    def get_train_loader(self):
        if self.train_loader is None:
            self.load_data()
        return self.train_loader
    
    def get_test_loader(self):
        if self.test_loader is None:
            self.load_data()
        return self.test_loader


