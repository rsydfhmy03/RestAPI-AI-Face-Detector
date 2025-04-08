from abc import ABC, abstractmethod

class BaseService(ABC):
    def __init__(self, repository):
        self.repository = repository

    @abstractmethod
    def process_and_predict(self, image):
        pass
