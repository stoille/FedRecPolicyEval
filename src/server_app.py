"""FedRecPolicyEval: A Flower / PyTorch package for evaluating recommender system policies."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from src.task import VAE, get_weights
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from typing import List, Tuple
from flwr.server import ServerApp, Driver
from flwr.server.history import History
import torch

def aggregate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    aggregated = {}
    total_examples = sum([num_examples for num_examples, _ in metrics])
    for _, m in metrics:
        for key in m:
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(m[key])
    averaged_metrics = {key: np.mean(aggregated[key]) for key in aggregated}
    return averaged_metrics

class CustomFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = {}

    def aggregate_fit(self, rnd, results, failures):
        return super().aggregate_fit(rnd, results, failures)

    def aggregate_evaluate(self, rnd, results, failures):
        metrics = aggregate_metrics([(res.num_examples, res.metrics) for _, res in results])
        self.history[rnd] = metrics
        return super().aggregate_evaluate(rnd, results, failures)

def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = context.run_config["num-server-rounds"]
    num_items = context.run_config["num-items"]
    
    # Initialize model on CPU since server doesn't need GPU
    device = torch.device("cpu")
    net = VAE(num_items=num_items).to(device)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)
    
    strategy = CustomFedAvg(
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)

import atexit

def cleanup():
    print("Performing cleanup operations")
    plot_history(app._strategy.history)

# Register the cleanup function
atexit.register(cleanup)

# Run the server and capture the history
def plot_history(history: History):

    # Save history to 'history.json'
    with open('history.json', 'w') as f:
        json.dump(history, f)

    print("Plotting history")
    # Generate plots
    rounds = sorted(history.keys())

    # Prepare data
    train_losses = [history[r].get('train_loss') for r in rounds]
    val_losses = [history[r].get('val_loss') for r in rounds]
    rmse = [history[r].get('rmse') for r in rounds]
    precision_at_k = [history[r].get('precision_at_k') for r in rounds]
    recall_at_k = [history[r].get('recall_at_k') for r in rounds]
    ndcg_at_k = [history[r].get('ndcg_at_k') for r in rounds]
    coverage = [history[r].get('coverage') for r in rounds]
    roc_auc = [history[r].get('roc_auc') for r in rounds]

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    axs[0, 0].plot(rounds, train_losses, label='Training Loss')
    axs[0, 0].plot(rounds, val_losses, label='Validation Loss')
    axs[0, 0].set_xlabel('Rounds')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Training and Validation Loss')
    axs[0, 0].legend()

    axs[0, 1].plot(rounds, rmse, label='RMSE', color='orange')
    axs[0, 1].set_xlabel('Rounds')
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].set_title('Root Mean Squared Error')
    axs[0, 1].legend()

    axs[0, 2].plot(rounds, precision_at_k, label='Precision@K')
    axs[0, 2].plot(rounds, recall_at_k, label='Recall@K')
    axs[0, 2].set_xlabel('Rounds')
    axs[0, 2].set_ylabel('Score')
    axs[0, 2].set_title('Precision@K and Recall@K')
    axs[0, 2].legend()

    axs[1, 0].plot(rounds, ndcg_at_k, label='NDCG@K', color='green')
    axs[1, 0].set_xlabel('Rounds')
    axs[1, 0].set_ylabel('NDCG')
    axs[1, 0].set_title('Normalized Discounted Cumulative Gain')
    axs[1, 0].legend()

    axs[1, 1].plot(rounds, roc_auc, label='ROC AUC', color='blue')
    axs[1, 1].set_xlabel('Rounds')
    axs[1, 1].set_ylabel('ROC AUC')
    axs[1, 1].set_title('ROC AUC')
    axs[1, 1].legend()

    axs[1, 2].plot(rounds, coverage, label='Coverage', color='brown')
    axs[1, 2].set_xlabel('Rounds')
    axs[1, 2].set_ylabel('Coverage')
    axs[1, 2].set_title('Coverage')
    axs[1, 2].legend()

    plt.tight_layout()
    plt.savefig('metrics_plots.png')
    plt.close()
