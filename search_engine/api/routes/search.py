import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict
from search_engine.search.search_service import SearchService
from search_engine.embeddings.fasttext_embedding import FastTextEmbedding
from search_engine.embeddings.transformer_embedding import TransformerEmbedding
from qdrant_client import QdrantClient
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()

# Load configuration
config_path = os.getenv("CONFIG_FILE_PATH")
if not config_path:
    raise ValueError("CONFIG_FILE_PATH environment variable is not set")

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Set up artifacts path
qdrant_path = os.getenv("VECTOR_DB_PATH")
if not qdrant_path:
    raise ValueError("VECTOR_DB_PATH environment variable is not set")

collection_name = os.getenv("QDRANT_COLLECTION_NAME")
if not collection_name:
    raise ValueError("QDRANT_COLLECTION_NAME environment variable is not set")

artifacts_path = os.path.join(qdrant_path, "collection", collection_name)

# Initialize embedding model
embedding_config = config['embedding']['config']
training_data_path = config['embedding'].get('training_data_path')
model_path = os.path.join(artifacts_path, "fasttext_model.bin")

if os.path.exists(model_path):
    embedding_model = FastTextEmbedding(config=embedding_config, model_path=model_path)
else:
    training_data = load_training_data(training_data_path)
    embedding_model = FastTextEmbedding(config=embedding_config, training_data=training_data, model_path=model_path)

# Initialize QdrantClient
qdrant_client = QdrantClient(path=qdrant_path)

# Initialize SearchService
whoosh_index_dir = os.path.join(artifacts_path, "whoosh_index")
search_service = SearchService(embedding_model, config, qdrant_client, whoosh_index_dir)

@router.get("/search")
async def semantic_search(
    query: str,
    category: str = None,
    limit: int = None,
    threshold: float = None,
    search_type: str = None
):
    try:
        logger.info(f"Received search request - query: {query}, category: {category}, limit: {limit}, threshold: {threshold}, search_type: {search_type}")
        results = search_service.search(query, category, limit, threshold, search_type)
        logger.info(f"Search completed, found {len(results)} results")
        response = [
            {
                "id": str(r['id']),
                "score": r['score'],
                "pregunta": str(r.get("pregunta", "")),
                "respuesta": str(r.get("respuesta", "")),
                "grupo": str(r.get("grupo", "")),
                "tema": str(r.get("tema", "") or "")
            } for r in results
        ]
        return response
    except Exception as e:
        logger.error(f"Error in semantic_search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during the search process")

@router.get("/article/{article_id}")
async def get_article(article_id: str):
    try:
        logger.info(f"Received request for article with id: {article_id}")
        article = search_service.get_article(article_id)
        if article:
            logger.info(f"Article found: {article['id']}")
            return article
        logger.warning(f"Article not found: {article_id}")
        raise HTTPException(status_code=404, detail="Article not found")
    except Exception as e:
        logger.error(f"Error in get_article: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while retrieving the article")

@router.get("/categories")
async def get_categories():
    try:
        logger.info("Received request for categories")
        categories = search_service.get_categories()
        logger.info(f"Retrieved {len(categories)} categories")
        return categories
    except Exception as e:
        logger.error(f"Error in get_categories: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while retrieving categories")