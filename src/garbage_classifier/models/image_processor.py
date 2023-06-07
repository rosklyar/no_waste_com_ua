from abc import ABC, abstractmethod

class ImageProcessor(ABC):
    
    @abstractmethod
    def process(self, image):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass