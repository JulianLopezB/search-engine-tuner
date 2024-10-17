import os
import logging
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from ..embeddings.base_embedding import BaseEmbedding
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, NUMERIC
from whoosh.query import Term, And, Or
import numpy as np
from ..utils.text_preprocessing import preprocess_text
from whoosh import qparser

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self, embedding: BaseEmbedding, config: dict, qdrant_client: QdrantClient, whoosh_index_dir: str):
        self.embedding = embedding
        self.config = config
        self.qdrant_path = os.getenv("VECTOR_DB_PATH")
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME")
        self.default_search_limit = int(os.getenv("SEARCH_LIMIT", 15))
        self.default_search_threshold = float(os.getenv("SEARCH_THRESHOLD", 0.0))
        self.hnsw_ef = int(os.getenv("HNSW_EF", 128))
        self.default_search_type = os.getenv("DEFAULT_SEARCH_TYPE", "hybrid")
        self.vector_weight = float(os.getenv("VECTOR_WEIGHT", 0.7))
        self.keyword_weight = float(os.getenv("KEYWORD_WEIGHT", 0.3))
        
        self.client = qdrant_client
        
        # Initialize Whoosh index
        self.whoosh_index_dir = whoosh_index_dir
        if not os.path.exists(self.whoosh_index_dir):
            raise ValueError(f"Whoosh index directory does not exist: {self.whoosh_index_dir}")
        
        if index.exists_in(self.whoosh_index_dir):
            self.ix = index.open_dir(self.whoosh_index_dir)
            logger.info(f"Opened existing Whoosh index in {self.whoosh_index_dir}")
        else:
            raise ValueError(f"Whoosh index does not exist in the specified directory: {self.whoosh_index_dir}")

    def search(self, query: str, category: str = None, limit: int = None, threshold: float = None, search_type: str = None) -> List[Dict]:
        limit = limit if limit is not None else self.default_search_limit
        threshold = threshold if threshold is not None else self.default_search_threshold
        search_type = search_type if search_type is not None else self.default_search_type
        processed_query = preprocess_text(query)
        
        if search_type == "vector":
            return self._vector_search(processed_query, category, limit, threshold)
        elif search_type == "keyword":
            return self._keyword_search(processed_query, category, limit)
        elif search_type == "hybrid":
            return self._hybrid_search(processed_query, category, limit, threshold)
        else:
            raise ValueError(f"Unsupported search type: {search_type}")

    def _vector_search(self, query: str, category: str = None, limit: int = None, threshold: float = None) -> List[Dict]:
        query_vector = self.embedding.embed_query(query)
        
        filter_conditions = []
        if category:
            filter_conditions.append(
                models.FieldCondition(key="grupo", match=models.MatchValue(value=str(category)))
            )
            
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=models.Filter(must=filter_conditions) if filter_conditions else None,
            limit=limit * 5,  # Retrieve more results initially
            score_threshold=threshold,
            search_params=models.SearchParams(hnsw_ef=self.hnsw_ef),
        )
        
        # Group results by original article ID
        grouped_results = {}
        for result in search_result:
            article_id = result.payload.get('article_id')
            pregunta = result.payload.get('pregunta')
            if pregunta is None or (isinstance(pregunta, float) and np.isnan(pregunta)):
                pregunta = ""
            if article_id not in grouped_results or result.score > grouped_results[article_id]['score']:
                grouped_results[article_id] = {
                    "id": result.id,
                    "article_id": article_id,
                    "score": result.score,
                    "pregunta": pregunta,
                    "respuesta": result.payload.get('respuesta'),
                    "grupo": result.payload.get('grupo'),
                    "tema": result.payload.get('tema', ''),
                }
        
        # Filter out results with NaN or infinite scores
        filtered_results = [
            result for result in grouped_results.values()
            if not (np.isnan(result['score']) or np.isinf(result['score']))
        ]
        
        # Sort the best chunks and return the top results
        sorted_results = sorted(filtered_results, key=lambda x: x['score'], reverse=True)
        
        return sorted_results[:limit]

    def _keyword_search(self, query: str, category: str = None, limit: int = None) -> List[Dict]:
        with self.ix.searcher() as searcher:
            print(f"Document count: {searcher.doc_count()}")

            # Modify the query string to force OR behavior
            terms = [Term("text", term) for term in query.split()]
            parsed_query = Or(terms)

            if category:
                category_query = Term("grupo", str(category))

                logger.debug(f"Executing keyword search with query: {parsed_query}")
                
                results = searcher.search(parsed_query, filter=category_query, limit=limit * 5 if limit else None)  # Retrieve more results initially
                logger.debug(f"Keyword search returned {len(results)} results")
            else:
                
                logger.debug(f"Executing keyword search with query: {parsed_query}")
                
                results = searcher.search(parsed_query, limit=limit * 5 if limit else None)  # Retrieve more results initially
                logger.debug(f"Keyword search returned {len(results)} results")
            # Group results by original article ID
            grouped_results = {}
            for result in results:
                article_id = result.get('article_id')
                pregunta = result.get('pregunta', '')
                if pregunta is None or (isinstance(pregunta, float) and np.isnan(pregunta)):
                    pregunta = ""
                if article_id not in grouped_results or result.score > grouped_results[article_id]['score']:
                    grouped_results[article_id] = {
                        "id": result['id'],
                        "article_id": article_id,
                        "score": result.score,
                        "pregunta": pregunta,
                        "respuesta": result.get('respuesta', ''),
                        "grupo": result.get('grupo', ''),
                        "tema": result.get('tema', ''),
                    }

            # Filter out results with NaN or infinite scores
            filtered_results = [
                result for result in grouped_results.values()
                if not (np.isnan(result['score']) or np.isinf(result['score']))
            ]

            # Sort the best chunks and return the top results
            sorted_results = sorted(filtered_results, key=lambda x: x['score'], reverse=True)
            
            return sorted_results[:limit] if limit else sorted_results

    def _hybrid_search(self, query: str, category: str = None, limit: int = None, threshold: float = None) -> List[Dict]:
        vector_results = self._vector_search(query, category, limit * 2, threshold)
        keyword_results = self._keyword_search(query, category, limit * 2)
        
        # Combine and re-rank results
        combined_results = {}
        for result in vector_results:
            combined_results[result['article_id']] = {
                **result,
                'final_score': self.vector_weight * result['score']
            }
        
        for result in keyword_results:
            if result['article_id'] in combined_results:
                combined_results[result['article_id']]['final_score'] += self.keyword_weight * result['score']
            else:
                combined_results[result['article_id']] = {
                    **result,
                    'final_score': self.keyword_weight * result['score']
                }
                
        sorted_results = sorted(combined_results.values(), key=lambda x: x['final_score'], reverse=True)
        return sorted_results[:limit]

    def get_categories(self):
        logger.info("Fetching categories")
        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(),
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        categories = set()
        for point in result[0]:
            category = point.payload.get("grupo")
            if category:
                categories.add(category)
        
        # Convert to list of integers, sort, then convert back to strings
        sorted_categories = sorted(map(int, categories))
        return [str(cat) for cat in sorted_categories]

    def get_article(self, article_id: str) -> Optional[Dict]:
        try:
            result = self.client.retrieve(collection_name=self.collection_name,ids=[str(article_id)],with_payload=True,with_vectors=False)
            
            if result:
                article = result[0]
                return {
                    "id": str(article.id),
                    "pregunta": str(article.payload.get("pregunta", "")),
                    "grupo": str(article.payload.get("grupo", "")),
                    "tema": str(article.payload.get("tema", "") or ""),
                    "respuesta": str(article.payload.get("respuesta", "")).replace("_x000d_", "")
                }
            else:
                logger.warning(f"Article not found: {article_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving article {article_id}: {str(e)}")
            return None