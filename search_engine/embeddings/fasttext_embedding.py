import os
import fasttext
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from .base_embedding import BaseEmbedding

class FastTextEmbedding(BaseEmbedding):
    def __init__(self, config, training_data=None, model_path=None):
        self.config = config
        self.training_data = training_data
        self.model_path = model_path
        self.load_or_train_model()

    @property
    def vector_size(self) -> int:
        return self.model.get_dimension()

    def load_or_train_model(self):
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
        elif self.training_data:
            self.model = self.train(self.training_data)
            if self.model_path:
                self.save_model(self.model_path)
        else:
            raise ValueError("No model path or training data provided")

    def train(self, articles: List[str]):
        # Prepare training data
        temp_file = "temp_training_data.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            for text in articles:
                f.write(f"{text['text']}\n")

        # Train the model
        model = fasttext.train_unsupervised(
            temp_file,
            model=self.config.get('model', 'skipgram'),
            dim=self.config.get('dim', 100),
            lr=self.config.get('lr', 0.05),
            epoch=self.config.get('epoch', 5),
            wordNgrams=self.config.get('wordNgrams', 1),
            minn=self.config.get('minn', 3),
            maxn=self.config.get('maxn', 6),
            thread=self.config.get('thread', 4)
        )

        # Remove temporary file
        os.remove(temp_file)

        return model

    def load_model(self, model_path: str):
        self.model = fasttext.load_model(model_path)

    def save_model(self, model_path: str):
        self.model.save_model(model_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in tqdm(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.model.get_sentence_vector(text).tolist()
