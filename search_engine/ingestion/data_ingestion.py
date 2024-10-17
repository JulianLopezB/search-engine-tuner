import os
import logging
from typing import List, Dict
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from ..embeddings.base_embedding import BaseEmbedding
from ..embeddings.fasttext_embedding import FastTextEmbedding  # Add this line
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from ..utils.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DataIngestionService:
    def __init__(self, embedding: BaseEmbedding, search_service, qdrant_client: QdrantClient, config: dict, artifacts_path: str):
        self.embedding = embedding
        self.search_service = search_service
        self.qdrant_path = os.getenv("VECTOR_DB_PATH")
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME")
        self.config = config
        self.artifacts_path = artifacts_path
        
        self.client = qdrant_client
        chunk_size = config['ingestion'].get('chunk_size', 1000)
        chunk_overlap = config['ingestion'].get('chunk_overlap', 200)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.recreate_collection()

    def recreate_collection(self):
        # Check if collection exists
        collections = self.client.get_collections().collections
        if any(collection.name == self.collection_name for collection in collections):
            logger.info(f"Collection {self.collection_name} already exists. Skipping recreation.")
            return

        # Create a new collection
        logger.info(f"Creating new collection: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.embedding.vector_size, distance=models.Distance.COSINE),
        )

    def ingest_data(self):
        logger.info("Starting data ingestion")
        
        logger.info(f"Loaded {len(self.embedding.training_data)} records")
        
        writer = self.search_service.ix.writer()
        
        total_points = 0
        batch_size = 100  # Adjust this value based on your needs and memory constraints

        for item in tqdm(self.embedding.training_data):
            full_text = item['text']
            chunks = self.text_splitter.split_text(full_text)
            
            points = []
            for i, chunk in enumerate(chunks):
                point_id = str(uuid.uuid4())
                chunk_embedding = self.embedding.embed_query(chunk)
                
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=chunk_embedding,
                        payload={
                            'text': chunk,
                            'pregunta': item['pregunta'],
                            'respuesta': item['respuesta'],
                            'grupo': str(item['grupo']),  # Convert to string to avoid nan
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'article_id': str(item['id'])
                        }
                    )
                )
                
                writer.add_document(
                    id=point_id,
                    article_id=str(item['id']),
                    pregunta=str(item['pregunta']),  # Convert to string to avoid nan
                    respuesta=str(item['respuesta']),  # Convert to string to avoid nan
                    grupo=str(item['grupo']),  # Convert to string to avoid nan
                    tema=str(item.get('tema', '')),  # Convert to string to avoid nan
                    text=chunk,
                    chunk_index=str(i),
                    total_chunks=str(len(chunks))
                )
                logger.debug(f"Added document with id: {point_id}")
            
            # Upsert points for this item
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            total_points += len(points)
        
        # Final commit for any remaining documents
        writer.commit()
        logger.info(f"Committed a total of {total_points} points to Whoosh index")
        logger.info(f"Upserted a total of {total_points} points to Qdrant")

        # # Save FastText model if it's a FastText embedding
        # if isinstance(self.embedding, FastTextEmbedding):
        #     self.embedding.save_model(os.path.join(self.artifacts_path, "fasttext_model.bin"))
