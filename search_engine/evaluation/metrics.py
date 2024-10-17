import numpy as np
from typing import List, Dict
from scipy import stats

def mean_reciprocal_rank(rankings: List[List[int]], ground_truth: List[List[int]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR)
    """
    reciprocal_ranks = []
    for pred, true in zip(rankings, ground_truth):
        for rank, article_id in enumerate(pred, 1):
            if article_id in true:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks)

def precision_at_k(rankings: List[List[int]], ground_truth: List[List[int]], k: int) -> float:
    """
    Calculate Precision@k
    """
    precisions = []
    for pred, true in zip(rankings, ground_truth):
        relevant = set(true)
        pred_at_k = pred[:k]
        precisions.append(len(set(pred_at_k) & relevant) / k)
    return np.mean(precisions)

def recall_at_k(rankings: List[List[int]], ground_truth: List[List[int]], k: int) -> float:
    """
    Calculate Recall@k
    """
    recalls = []
    for pred, true in zip(rankings, ground_truth):
        relevant = set(true)
        pred_at_k = set(pred[:k])
        recalls.append(len(pred_at_k & relevant) / len(relevant))
    return np.mean(recalls)

def average_precision(ranking: List[int], ground_truth: List[int]) -> float:
    """
    Calculate Average Precision for a single query
    """
    relevant_items = set(ground_truth)
    precisions = []
    relevant_count = 0

    for rank, article_id in enumerate(ranking, 1):
        if article_id in relevant_items:
            relevant_count += 1
            precisions.append(relevant_count / rank)

    if not precisions:
        return 0.0
    return sum(precisions) / len(relevant_items)

def mean_average_precision(rankings: List[List[int]], ground_truth: List[List[int]]) -> float:
    """
    Calculate Mean Average Precision (MAP)
    """
    aps = [average_precision(ranking, truth) for ranking, truth in zip(rankings, ground_truth)]
    return np.mean(aps)

def dcg_at_k(ranking: List[int], ground_truth: List[int], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at k
    """
    dcg = 0
    for i, article_id in enumerate(ranking[:k]):
        if article_id in ground_truth:
            dcg += 1 / np.log2(i + 2)
    return dcg

def ndcg_at_k(rankings: List[List[int]], ground_truth: List[List[int]], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k
    """
    ndcgs = []
    for ranking, truth in zip(rankings, ground_truth):
        dcg = dcg_at_k(ranking, truth, k)
        idcg = dcg_at_k(sorted(ranking, key=lambda x: truth.index(x) if x in truth else len(truth)), truth, k)
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    return np.mean(ndcgs)

def kendalls_tau(rankings: List[List[int]], ground_truth: List[List[int]]) -> float:
    """
    Calculate Kendall's Tau
    """
    taus = []
    for pred, true in zip(rankings, ground_truth):
        tau, _ = stats.kendalltau(pred, true)
        taus.append(tau)
    return np.mean(taus)

def spearmans_rank_correlation(rankings: List[List[int]], ground_truth: List[List[int]]) -> float:
    """
    Calculate Spearman's Rank Correlation Coefficient
    """
    correlations = []
    for pred, true in zip(rankings, ground_truth):
        corr, _ = stats.spearmanr(pred, true)
        correlations.append(corr)
    return np.mean(correlations)

def evaluate_rankings(predictions: List[List[int]], ground_truth: List[List[int]]) -> Dict[str, float]:
    """
    Evaluate rankings using multiple metrics
    """
    return {
        "MRR": mean_reciprocal_rank(predictions, ground_truth),
        "P_at_1": precision_at_k(predictions, ground_truth, 1),
        "P_at_3": precision_at_k(predictions, ground_truth, 3),
        "P_at_5": precision_at_k(predictions, ground_truth, 5),
        "R_at_1": recall_at_k(predictions, ground_truth, 1),
        "R_at_3": recall_at_k(predictions, ground_truth, 3),
        "R_at_5": recall_at_k(predictions, ground_truth, 5),
        "MAP": mean_average_precision(predictions, ground_truth),
        "NDCG_at_3": ndcg_at_k(predictions, ground_truth, 3),
        "NDCG_at_5": ndcg_at_k(predictions, ground_truth, 5),
        "Kendalls_Tau": kendalls_tau(predictions, ground_truth),
        "Spearmans_Rank_Correlation": spearmans_rank_correlation(predictions, ground_truth)
    }
