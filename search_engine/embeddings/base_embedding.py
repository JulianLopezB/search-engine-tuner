from abc import ABC, abstractmethod
from typing import List, Dict

class BaseEmbedding(ABC):
    @property
    @abstractmethod
    def vector_size(self) -> int:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def train(self, texts: List[str]):
        pass

    @abstractmethod
    def load_model(self, model_path: str):
        pass

    @abstractmethod
    def save_model(self, model_path: str):
        pass
