import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
from tqdm import tqdm
from .base_embedding import BaseEmbedding

class TransformerEmbedding(BaseEmbedding):
    def __init__(self, config, training_data=None):
        self.config = config
        self.training_data = training_data
        self.model = None
        self.tokenizer = None
        self.load_or_train_model()

    def load_or_train_model(self):
        model_name = self.config.get('model_name')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.model = AutoModel.from_pretrained(model_name)

    @property
    def vector_size(self) -> int:
        return self.model.config.hidden_size

    def train(self, texts: List[str]):
        # Transformer models are usually pre-trained, so we don't need to train them here
        # However, we could implement fine-tuning logic if needed
        pass

    def load_model(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

    def save_model(self, model_path: str):
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in tqdm()]

    def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.config.get("max_length", 512))
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
