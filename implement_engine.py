import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from search_engine.ingestion.data_ingestion import DataIngestionService
from search_engine.search.search_service import SearchService
from search_engine.utils.config import load_config
from search_engine.utils.data_preprocessor import load_training_data
from search_engine.embeddings.fasttext_embedding import FastTextEmbedding
from search_engine.embeddings.transformer_embedding import TransformerEmbedding
from whoosh import index
from whoosh.fields import Schema, TEXT, ID

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def get_embedding_model(config, artifacts_path):
    embedding_type = config['embedding']['type']
    embedding_config = config['embedding']['config']
    training_data_path = config['embedding'].get('training_data_path')

    logger.info(f"Loading training data from {training_data_path}")
    training_data = load_training_data(training_data_path)
    logger.info(f"Loaded {len(training_data)} training samples")

    if embedding_type == 'fasttext':
        logger.info("Initializing FastText embedding model")
        model_path = os.path.join(artifacts_path, "fasttext_model.bin")
        return FastTextEmbedding(config=embedding_config, training_data=training_data, model_path=model_path)
    elif embedding_type == 'transformer':
        logger.info("Initializing Transformer embedding model")
        return TransformerEmbedding(config=embedding_config, training_data=training_data)
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

def create_whoosh_index(config, artifacts_path):
    whoosh_index_dir = os.path.join(artifacts_path, "whoosh_index")
    if not os.path.exists(whoosh_index_dir):
        os.makedirs(whoosh_index_dir, exist_ok=True)
        logger.info(f"Created Whoosh index directory: {whoosh_index_dir}")
    
    schema = Schema(
        id=ID(stored=True),
        article_id=ID(stored=True),
        pregunta=TEXT(stored=True),
        respuesta=TEXT(stored=True),
        grupo=ID(stored=True),
        tema=TEXT(stored=True),
        text=TEXT(stored=True),
        chunk_index=ID(stored=True),
        total_chunks=ID(stored=True)
    )
    
    if not index.exists_in(whoosh_index_dir):
        ix = index.create_in(whoosh_index_dir, schema)
        logger.info(f"Created new Whoosh index in {whoosh_index_dir}")
    else:
        logger.info(f"Whoosh index already exists in {whoosh_index_dir}")
    return whoosh_index_dir

def implement_engine():
    logger.info("Starting engine implementation")

    # Load configuration
    config_path = os.getenv("CONFIG_FILE_PATH")
    if not config_path:
        raise ValueError("CONFIG_FILE_PATH environment variable is not set")
    
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    logger.info("Configuration loaded successfully")

    # Set up artifacts path
    qdrant_path = os.getenv("VECTOR_DB_PATH")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    artifacts_path = os.path.join(qdrant_path, "collection", collection_name)
    os.makedirs(artifacts_path, exist_ok=True)
    logger.info(f"Artifacts will be stored in: {artifacts_path}")

    # Initialize and train (if needed) the embedding model
    logger.info("Initializing embedding model")
    embedding_model = get_embedding_model(config, artifacts_path)
    logger.info("Embedding model initialized successfully")

    # Initialize QdrantClient
    logger.info(f"Initializing QdrantClient with path: {qdrant_path}")
    qdrant_client = QdrantClient(path=qdrant_path)
    logger.info("QdrantClient initialized successfully")

    # Create Whoosh index structure (if it doesn't exist)
    whoosh_index_dir = create_whoosh_index(config, artifacts_path)

    # Initialize SearchService
    logger.info("Initializing SearchService")
    search_service = SearchService(embedding_model, config, qdrant_client, whoosh_index_dir)
    logger.info("SearchService initialized successfully")

    # Ingest data
    logger.info("Starting data ingestion process")
    data_ingestion = DataIngestionService(embedding_model, search_service, qdrant_client, config, artifacts_path)
    data_ingestion.ingest_data()
    logger.info("Data ingestion completed")

    logger.info("Engine implementation completed successfully")
    return search_service

if __name__ == "__main__":
    try:
        implement_engine()
    except Exception as e:
        logger.exception("An error occurred during engine implementation")
        raise