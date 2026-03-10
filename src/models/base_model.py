from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseDetector(ABC):
    @abstractmethod
    def predict(self, image) -> List[Dict[str, Any]]:
        """Return list of detections."""
        raise NotImplementedError
