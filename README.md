# Search Engine Tuner POC - Backend

This is the backend component of the Search Engine Tuner POC, built with FastAPI.

## Overview

Search Engine Tuner is a comprehensive platform for calibrating and evaluating search engines using different embeddings and approaches. It supports various vector-based methods, including pretrained transformers from HuggingFace and OpenAI, as well as embeddings trained with algorithms like FastText.

The platform allows the measurement of key metrics in information retrieval, such as topK precision, recall, MAP, MRR, NDCG, and more, using a customizable evaluation dataset.

The project leverages synthetic data generation via generative AI to create evaluation datasets and employs mlflow for tracking results, adhering to MLOps best practices. Search Engine Tuner is scalable, language-agnostic, customizable, and performant, making it suitable for any document corpus.

## Key Features

- Vector-based search using Qdrant
- Multiple embedding models support (FastText and Transformer-based)
- Data ingestion and preprocessing
- Evaluation of embedding models
- Dockerized application for easy deployment
- Supports vector, keyword, and hybrid engines
- Embedding customization through pretrained models or FastText
- Scalable and performant, adaptable to any language or corpus
- MLOps best practices using mlflow
- Tested on articles in Spanish and Galician
- Serves as both a demo and a production environment for a client, including a frontend with a user-friendly UX

## Prerequisites

- Python 3.11+
- Docker (optional)

## Setup and Running

### Evaluating and Implementing the Engine

1. Evaluate the engine using different configurations:
   ```
   python evaluate_engine.py config/fasttext_config.yaml
   python evaluate_engine.py config/transformer_config.yaml
   ```

2. Implement the engine (uses environment variables for configuration):
   ```
   python implement_engine.py
   ```

### Running the API

#### Using Python

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the FastAPI server:
   ```
   python run.py
   ```

3. The API will be available at `http://localhost:8001`.

#### Using Docker

1. Build and run using Docker Compose:
   ```
   docker-compose up --build
   ```

2. The API will be available at `http://localhost:8000`.

## API Endpoints

- `/search`: Perform a vector search
- `/categories`: Get available categories
- `/article/{article_id}`: Get a specific article

For detailed API documentation, visit `http://localhost:8001/docs` after starting the application.

## Development

- Main application code is in the `search_engine` directory
- API routes are defined in `search_engine/api/routes/search.py`
- Core search functionality is in `search_engine/search/search_service.py`
- Evaluation logic is in `search_engine/evaluation/evaluator.py`

## Evaluation and Metrics

The platform allows measurement of key metrics in information retrieval, including:
- Top-K Precision
- Recall
- Mean Average Precision (MAP)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

## Data Generation

Search Engine Tuner uses synthetic data generation via generative AI to create customizable evaluation datasets, ensuring a robust and diverse testing environment.

## MLOps Integration

The project adheres to MLOps best practices by using mlflow for experiment tracking and result logging, enabling better reproducibility and model management.

## Language Support

While primarily tested on articles in Spanish and Galician, Search Engine Tuner is designed to be language-agnostic and can be adapted to any document corpus.

## License

This project is licensed under the MIT License.
