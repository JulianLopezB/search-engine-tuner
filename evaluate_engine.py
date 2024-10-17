import logging
from dotenv import load_dotenv
from search_engine.embeddings.fasttext_embedding import FastTextEmbedding
from search_engine.embeddings.transformer_embedding import TransformerEmbedding
from search_engine.evaluation.evaluator import Evaluator
from search_engine.utils.config import load_config
from search_engine.utils.data_preprocessor import load_training_data
from search_engine.search.search_service import SearchService
from qdrant_client import QdrantClient
import mlflow
import os
import uuid

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embedding_model(config, artifacts_path):
    embedding_type = config['embedding']['type']
    logger.info(f"Loading training data from {config['embedding'].get('training_data_path')}")
    training_data = load_training_data(config['embedding'].get('training_data_path'))
    logger.info(f"Loaded {len(training_data)} training samples")

    if embedding_type == 'fasttext':
        logger.info("Initializing FastText embedding model")
        model_path = os.path.join(artifacts_path, "fasttext_model.bin")
        return FastTextEmbedding(config=config['embedding']['config'], training_data=training_data, model_path=model_path)
    elif embedding_type == 'transformer':
        logger.info("Initializing Transformer embedding model")
        return TransformerEmbedding(config=config['embedding']['config'], training_data=training_data)
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

def sanitize_param_name(name):
    # Replace invalid characters with underscores
    return ''.join(char if char.isalnum() or char in ['_', '-', '.', ' ', '/'] else '_' for char in name)

def evaluate_engine():
    # Load configuration from the path specified in the environment variable
    config_path = os.getenv("CONFIG_FILE_PATH")
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Set up artifacts path
    qdrant_path = os.getenv("VECTOR_DB_PATH")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    artifacts_path = os.path.join(qdrant_path, "collection", collection_name)
    os.makedirs(artifacts_path, exist_ok=True)
    logger.info(f"Artifacts will be stored in: {artifacts_path}")

    logger.info("Initializing embedding model")
    embedding = get_embedding_model(config, artifacts_path)

    # Initialize QdrantClient
    logger.info(f"Initializing QdrantClient with path: {qdrant_path}")
    qdrant_client = QdrantClient(path=qdrant_path)

    # Initialize SearchService
    whoosh_index_dir = os.path.join(artifacts_path, "whoosh_index")
    logger.info("Initializing SearchService")
    search_service = SearchService(embedding, config, qdrant_client, whoosh_index_dir)

    env_vars = {sanitize_param_name(key): os.getenv(key) for key in os.environ.keys() if key.isidentifier()}
    
    search_types = config['search'].get('types_to_evaluate', ['vector', 'keyword', 'hybrid'])

    # Generate a unique identifier for this run
    run_id = str(uuid.uuid4())[:8]  # Use the first 8 characters of the UUID
    
    for search_type in search_types:

        # End any active run before starting a new one
        mlflow.end_run()  # End the previous run if it exists

        run_name = f"{run_id}_{config['embedding']['type']}_{search_type}_search"

        with mlflow.start_run(run_name=run_name):

            logger.info(f"Starting run {run_name}")
            
            logger.info(f"Setting up MLflow experiment: {config['mlflow']['experiment_name']}")
            mlflow.set_experiment(config['mlflow']['experiment_name'])

            # Log all parameters from the config
            mlflow.log_params(config)

            # Log all environment variables
            mlflow.log_params(env_vars)
            
            logger.info(f"Starting evaluation for {search_type} search (Run ID: {run_id})")
            evaluator = Evaluator(search_service)
            logger.info(f"Evaluating with articles file: {config['evaluation']['articles_file']}")
            logger.info(f"Evaluating with rankings file: {config['evaluation']['rankings_file']}")
            results = evaluator.evaluate(config['evaluation']['articles_file'], config['evaluation']['rankings_file'], search_type)

            logger.info("Logging parameters and metrics to MLflow")
            mlflow.log_params({
                "embedding_type": config['embedding']['type'],
                "search_type": search_type,
                "run_id": run_id,
                **config['embedding']['config'],
                **config['search'].get(search_type, {})
            })
            
            for metric_name, metric_value in results.items():
                logger.info(f"Logging metric: {metric_name} = {metric_value}")
                mlflow.log_metric(metric_name, metric_value)

            logger.info(f"{config['embedding']['type'].capitalize()} {search_type.capitalize()} Search Evaluation Results (Run ID: {run_id}):")
            for metric_name, metric_value in results.items():
                logger.info(f"{metric_name}: {metric_value}")

                

        # End any active run before starting a new one
        mlflow.end_run()  # End the previous run if it exists

    return embedding

if __name__ == "__main__":
    logger.info("Starting evaluation")
    evaluate_engine()
    logger.info("Evaluation completed")