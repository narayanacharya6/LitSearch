import numpy as np

def aggregate_recall(queries):
    """
    Calculate aggregate recall (mean and micro-averaged) for a set of queries.

    Args:
        queries (list of dict): A list of dictionaries where each dictionary represents a query.
                                Each dictionary should have two keys:
                                - 'corpusids': Set of relevant document ids
                                - 'retrieved': List of retrieved document ids

    Returns:
        dict: A dictionary with two keys, 'mean_recall' and 'micro_avg_recall'.
    """
    recall_scores = []
    total_relevant_retrieved = 0
    total_relevant = 0

    for query in queries:
        true_ids = query['corpusids']
        retrieved_ids = query['retrieved']
        
        # Calculate Recall for the current query
        retrieved_relevant_count = len([doc_id for doc_id in retrieved_ids if doc_id in true_ids])
        total_relevant_count = len(true_ids)
        
        # Update totals for micro-averaged recall
        total_relevant_retrieved += retrieved_relevant_count
        total_relevant += total_relevant_count
        
        # Calculate recall and store for mean recall calculation
        recall = retrieved_relevant_count / total_relevant_count if total_relevant_count > 0 else 0
        recall_scores.append(recall)

    # Mean Recall (Macro-Averaged Recall)
    mean_recall = np.mean(recall_scores) if recall_scores else 0

    # Micro-Averaged Recall
    micro_avg_recall = total_relevant_retrieved / total_relevant if total_relevant > 0 else 0

    return {
        'mean_recall': mean_recall,
        'micro_avg_recall': micro_avg_recall
    }

def aggregate_ndcg(queries):
    """
    Calculate aggregate NDCG (mean and micro-averaged) for a set of queries.

    Args:
        queries (list of dict): A list of dictionaries where each dictionary represents a query.
                                Each dictionary should have two keys:
                                - 'corpusids': Set of relevant document ids
                                - 'retrieved': List of retrieved document ids

    Returns:
        dict: A dictionary with 'mean_ndcg' key.
    """
    ndcg_scores = []

    for query in queries:
        true_ids = query['corpusids']
        retrieved_ids = query['retrieved']
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in true_ids:
                dcg += 1 / np.log2(i + 2)  # i+2 because we want to start from log_2(2)
        
        # Calculate IDCG
        ideal_retrieved = sorted(true_ids, key=lambda x: retrieved_ids.index(x) if x in retrieved_ids else float('inf'))
        idcg = 0.0
        for i in range(min(len(ideal_retrieved), len(retrieved_ids))):
            idcg += 1 / np.log2(i + 2)  # i+2 because we want to start from log_2(2)
        
        # Calculate NDCG for the current query
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

    # Mean NDCG (Macro-Averaged NDCG)
    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0

    return {
        'mean_ndcg': mean_ndcg
    }


# Example usage
# queries = [
#     {
#         'corpusids': {1, 2, 3},
#         'retrieved': [3, 4, 5]
#     },
#     {
#         'corpusids': {2, 4},
#         'retrieved': [3, 2, 1, 4]
#     }
# ]

# results = aggregate_ndcg(queries)
# print("Mean NDCG:", results['mean_ndcg'])

# results = aggregate_recall(queries)
# print("Mean Recall:", results['mean_recall'])
# print("Micro-Averaged Recall:", results['micro_avg_recall'])
