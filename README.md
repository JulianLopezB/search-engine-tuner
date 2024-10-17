# ContactNova Search Engine POC - Backend

This is the backend component of the ContactNova Search Engine POC, built with FastAPI.

## Features

- Vector-based search using Qdrant
- Multiple embedding models support (FastText and Transformer-based)
- Data ingestion and preprocessing
- Evaluation of embedding models
- Dockerized application for easy deployment

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

## License

This project is licensed under the MIT License.