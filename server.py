# ------------------------------------------------- #
# Author: xietao                					#
# Repo: https://github.com/xietao02/Federated-NCF/  #
# ------------------------------------------------- #


import config 
import flwr as fl
import numpy as np
import torch
import logging
from typing import List, Tuple, Dict

fl.common.logger.FLOWER_LOGGER.setLevel(logging.CRITICAL)

rounds = 0
best_hr = 0
best_ndcg = 0
best_round = 0

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    # Initialize dictionaries to store aggregated metric values and total examples
    aggregated_metrics = {metric_name: 0.0 for metric_name in metrics[0][1].keys()}
    total_examples = 0

    # Iterate through client metrics
    for num_examples, client_metrics in metrics:
        total_examples += num_examples
        # Update the aggregated metrics for each metric name
        for metric_name, metric_value in client_metrics.items():
            aggregated_metrics[metric_name] += num_examples * metric_value

    
    # Calculate the weighted average for each metric
    weighted_avg_metrics = {metric_name: aggregated_metrics[metric_name] / total_examples for metric_name in aggregated_metrics}

    global rounds
    rounds = rounds + 1
    print(f'[Rounds {rounds}]')
    print(f'Num of Federated Clients: {len(metrics)}')
    print("Weighted Average Metrics:")
    for metric_name, metric_value in weighted_avg_metrics.items():
        print(f" - {metric_name}: {metric_value}")
    print("")

    global best_hr, best_ndcg, best_round
    HR = weighted_avg_metrics['HR']
    NDCG = weighted_avg_metrics['NDCG']

    if HR > best_hr:
        best_hr = HR
        best_ndcg = NDCG
        best_round = rounds

    if rounds == config.FEDERATION_ROUNDS:
        print("[Federated Learning Finished]")
        print(f'Best Round {best_round}: HR = {best_hr}, NDCG = {best_ndcg}')
        print("")

    return weighted_avg_metrics


# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# Start server
print("Starting server...")
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=config.FEDERATION_ROUNDS),
    strategy=strategy,
)