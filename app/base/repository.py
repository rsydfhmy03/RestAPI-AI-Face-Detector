from abc import ABC, abstractmethod

class BaseRepository(ABC):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, image):
        pass
