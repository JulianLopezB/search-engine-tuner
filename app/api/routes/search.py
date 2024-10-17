import logging
logger = logging.getLogger(__name__)

from fastapi import APIRouter
from app.services.search_service import SearchService

router = APIRouter()
search_service = SearchService()

@router.get("/search")
def semantic_search(query: str, category: str = None, limit: int = 15, embedding_type: str = "fasttext"):
    logger.info(f"Received search request - query: {query}, category: {category}, limit: {limit}, embedding_type: {embedding_type}")
    results = search_service.search(query, category, limit, embedding_type)
    logger.info(f"Search completed, found {len(results)} results")
    response = [
        {
            "id": r.id,
            "score": r.score,
            "pregunta": str(r.payload.get("metadata", {}).get("pregunta", "")),
            "grupo": str(r.payload.get("metadata", {}).get("grupo", "")),
            "tema": str(r.payload.get("metadata", {}).get("tema", "") or "")
        } for r in results
    ]

    return response

@router.get("/article/{article_id}")
def get_article(article_id: str, embedding_type: str = "openai"):
    logger.info(f"Received request for article with id: {article_id}, embedding_type: {embedding_type}")
    article = search_service.get_article(article_id, embedding_type)
    if article:
        logger.info(f"Article found: {article.id}")
        return {
            "id": article.id,
            "pregunta": str(article.payload.get("metadata", {}).get("pregunta", "")),
            "grupo": str(article.payload.get("metadata", {}).get("grupo", "")),
            "tema": str(article.payload.get("metadata", {}).get("tema", "") or ""),
            "respuesta": str(article.payload.get("metadata", {}).get("respuesta", "")).replace("_x000d_", "")
        }
    return {"error": "Article not found"}

@router.get("/categories")
def get_categories():
    logger.info("Received request for categories")
    categories = search_service.get_categories()
    logger.info(f"Retrieved {len(categories)} categories")
    return categories

@router.get("/rag_search")
def rag_search(query: str, category: str = None):
    logger.info(f"Received RAG search request - query: {query}, category: {category}")
    results = search_service.rag_search(query, category)
    response = [
            {
                "id": source.get("_id"),
                "score": 1.0,  # RAG doesn't provide a score, so we use a default value
                "pregunta": source.get("pregunta", ""),
                "grupo": str(source.get("grupo", "")),
                "tema": str(source.get("tema", "") or "")
            } for source in results["sources"]
        ]
    logger.info(f"RAG search completed, answer generated with {len(response)} source documents")
    return response

@router.get("/search-with-ai-validation")
def search_with_ai_validation(query: str, category: str = None, limit: int = 15, threshold: float = 0.0, embedding_type: str = "openai"):
    logger.info(f"Received AI-validated search request - query: {query}, category: {category}, limit: {limit}, threshold: {threshold}, embedding_type: {embedding_type}")
    results = search_service.search_with_ai_validation(query, category, limit, threshold, embedding_type)
    response = [
        {
            "id": r['result'].id,
            "score": r['result'].score,
            "pregunta": str(r['result'].payload.get("metadata", {}).get("pregunta", "")),
            "grupo": str(r['result'].payload.get("metadata", {}).get("grupo", "")),
            "tema": str(r['result'].payload.get("metadata", {}).get("tema", "") or ""),
            "reason": r['reason']
        } for r in results
    ]
    logger.info(f"AI-validated search completed, found {len(response)} validated results")
    return response