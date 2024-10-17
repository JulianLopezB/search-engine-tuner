import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict
from tqdm import tqdm
from ..search.search_service import SearchService
from .metrics import evaluate_rankings
from ..utils.data_preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service

    def load_articles_and_rankings(self, articles_file: str, rankings_file: str) -> Tuple[Dict[int, str], List[Dict]]:
        logger.info(f"Loading and preprocessing articles from {articles_file}")
        preprocessor = DataPreprocessor(articles_file)
        df = preprocessor._load_and_filter_data()
        all_articles = {row['id']: preprocessor._process_row(row) for _, row in df.iterrows()}
        logger.info(f"Loaded {len(all_articles)} articles")

        logger.info(f"Loading rankings from {rankings_file}")
        with open(rankings_file, 'r', encoding='utf-8') as f:
            content = f.read()

        queries = content.split('### Consulta')[1:]  # Split by queries, ignore the first empty part
        rankings = []
        relevant_article_ids = set()

        for query in queries:
            lines = query.strip().split('\n')
            query_id = int(lines[0].split(':')[0].strip())
            query_text = lines[1].split('**Consulta:**')[1].strip()
            article_ranking = [int(line.split('.')[1].strip()) for line in lines[3:8]]  # Get the article IDs
            relevant_article_ids.update(article_ranking)

            rankings.append({
                'query_id': query_id,
                'query_text': query_text,
                'articles_ranking': article_ranking
            })

        logger.info(f"Loaded {len(rankings)} queries with rankings")
        relevant_articles = {article_id: all_articles[article_id] for article_id in relevant_article_ids}
        logger.info(f"Identified {len(relevant_articles)} relevant articles for evaluation")
        return relevant_articles, rankings


    def evaluate(self, articles_file: str, rankings_file: str, search_type: str) -> dict:
        logger.info("Starting evaluation")
        articles, rankings = self.load_articles_and_rankings(articles_file, rankings_file)
        
        predictions = []
        ground_truth = []

        logger.info("Computing rankings for queries")
        for i, query in enumerate(tqdm(rankings, desc="Processing queries")):
            query_text = query['query_text']
            
            search_results = self.search_service.search(query_text, limit=5, search_type=search_type)

            # Use article_id instead of id, and convert to int
            ranked_articles = [
                int(result.get('article_id'))
                for result in search_results
                if result.get('article_id') or result['article_id'].isdigit()
            ]
            
            if not ranked_articles:
                logger.warning(f"Empty prediction for query {i+1}: '{query_text}'")
                continue  # Skip this query if we got no valid results
            
            predictions.append(ranked_articles)
            ground_truth.append(query['articles_ranking'])

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(rankings)} queries")

        logger.info(f"Final number of valid predictions: {len(predictions)}")
        logger.info(f"Final number of ground truth items: {len(ground_truth)}")

        if len(predictions) != len(ground_truth):
            logger.error("Mismatch between number of predictions and ground truth items")
            return {}

        logger.info("Computing evaluation metrics")
        results = evaluate_rankings(predictions, ground_truth)
        logger.info("Evaluation completed")
        return results