from abc import ABC, abstractmethod

class DomainWatermark(ABC):

    @abstractmethod
    def apply_watermark(self, watermark_data):
        pass

    @abstractmethod
    def show_watermark(self):
        pass