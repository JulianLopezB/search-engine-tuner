import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pandas as pd
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
import numpy as np
import fasttext
from langchain.embeddings.base import Embeddings

load_dotenv()

import logging
logger = logging.getLogger(__name__)

class FastTextEmbeddings(Embeddings):
    def __init__(self, model_path, get_preprocessed_texts_func):
        self.model_path = model_path
        self.get_preprocessed_texts = get_preprocessed_texts_func
        self.model = self.load_or_train_model()

    def load_or_train_model(self):
        if not os.path.exists(self.model_path):
            logger.info(f"FastText model not found at {self.model_path}. Training new model.")
            return self.train_fasttext()
        else:
            try:
                return fasttext.load_model(self.model_path)
            except Exception as e:
                logger.error(f"Failed to load FastText model from {self.model_path}: {str(e)}")
                raise ValueError(f"Failed to load FastText model: {str(e)}")

    def embed_documents(self, texts):
        return [self.model.get_sentence_vector(text).tolist() for text in texts]

    def embed_query(self, text):
        return self.model.get_sentence_vector(text).tolist()

    def train_fasttext(self):
        texts = self.get_preprocessed_texts()
        with open('temp_training_data.txt', 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(f"{text}\n")
        
        model = fasttext.train_unsupervised(
            'temp_training_data.txt',
            model='skipgram',
            dim=300,
            epoch=int(os.getenv("FASTTEXT_EPOCH", "50")),
            lr=float(os.getenv("FASTTEXT_LR", "0.05")),
            wordNgrams=int(os.getenv("FASTTEXT_WORD_NGRAMS", "2")),
            minn=int(os.getenv("FASTTEXT_MINN", "2")),
            maxn=int(os.getenv("FASTTEXT_MAXN", "5")),
            thread=int(os.getenv("FASTTEXT_THREAD", "4"))
        )
        
        model.save_model(self.model_path)
        os.remove('temp_training_data.txt')
        return model

class DataIngestionService:
    def __init__(self, file_path):
        logger.info("Initializing DataIngestionService")
        self.file_path = file_path
        self.path = os.getenv("VECTOR_DB_PATH")
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.client = QdrantClient(path=self.path)
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME")
        
        embedding_type = os.getenv("EMBEDDING_TYPE", "openai").lower()
        if embedding_type == "openai":
            self.embeddings = OpenAIEmbeddings(
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            self.vector_size = len(self.embeddings.embed_query("test"))
        elif embedding_type == "fasttext":
            fasttext_model_path = os.getenv("FASTTEXT_MODEL_PATH")
            if not fasttext_model_path:
                raise ValueError("FASTTEXT_MODEL_PATH must be set when using FastText embeddings")
            self.embeddings = FastTextEmbeddings(fasttext_model_path, self.get_preprocessed_texts)
            self.vector_size = 300  # FastText embeddings size
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
        
        self.ensure_collection()
        
        vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )

    def ensure_collection(self):
        logger.info(f"Ensuring collection: {self.collection_name}")
        collections = self.client.get_collections().collections
        if any(collection.name == self.collection_name for collection in collections):
            logger.info(f"Removing existing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
        
        logger.info(f"Creating new collection: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE)
        )
        logger.info(f"Collection {self.collection_name} created successfully")

    def preprocess_text(self, html_content):
        logger.debug("Preprocessing text")
        # Remove HTML tags
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()
        
        # Remove special characters but keep some punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
        
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text

    def _load_and_filter_data(self):
        df = pd.read_excel(self.file_path)
        df = df[df['obsoleto'].isna()]
        df = df[df['revisado'] == 's']
        return df.reset_index(drop=True)

    def _process_row(self, row):
        pregunta_html = str(row['pregunta'])
        respuesta_html = str(row['respuesta'])
        
        pregunta_text = self.preprocess_text(pregunta_html)
        respuesta_text = self.preprocess_text(respuesta_html)
        
        full_text = pregunta_text + ' ' + respuesta_text
        
        metadata = {
            'id': int(row['id']),
            'pregunta': pregunta_html,
            'respuesta': respuesta_html,
            'grupo': row['grupo'],
            'tema': row['tema']
        }
        
        return full_text, metadata

    def get_preprocessed_texts(self):
        df = self._load_and_filter_data()
        return [self._process_row(row)[0] for _, row in df.iterrows()]

    def ingest_data(self):
        logger.info(f"Starting data ingestion from file: {self.file_path}")
        df = self._load_and_filter_data()
        total_rows = len(df)
        logger.info(f"Total rows to process: {total_rows}")
        
        vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )

        for _, row in tqdm(df.iterrows(), total=total_rows):
            full_text, metadata = self._process_row(row)
            vector_store.add_texts(texts=[full_text], metadatas=[metadata])

        logger.info("Data ingestion completed.")