"""FedRecPolicyEval: A Flower / PyTorch package for evaluating recommender system policies."""

import matplotlib.pyplot as plt  # Add this import
import json
import os
from src.task import VAE, get_weights
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from typing import List, Tuple
from flwr.common import Metrics
import numpy as np

def val_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    val_losses = []
    examples = []
    for num_examples, m in metrics:
        loss = m.get("val_loss", float('inf'))
        if not np.isnan(loss):
            val_losses.append(num_examples * loss)
            examples.append(num_examples)
    if sum(examples) == 0:
        return {"val_loss": float('nan')}
    return {"val_loss": sum(val_losses) / sum(examples)}

def fit_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    train_losses = []
    examples = []
    for num_examples, m in metrics:
        loss = m.get("train_loss", float('inf'))
        if not np.isnan(loss):
            train_losses.append(num_examples * loss)
            examples.append(num_examples)
    if sum(examples) == 0:
        return {"train_loss": float('nan')}
    return {"train_loss": sum(train_losses) / sum(examples)}

class CustomFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_losses = []
        self.val_losses = []

    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        train_loss = self._compute_train_loss(results)
        self.train_losses.append(train_loss)
        return aggregated_weights

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        val_loss = self._compute_val_loss(results)
        self.val_losses.append(val_loss)
        self._save_history(rnd, val_loss=val_loss, train_loss=self.train_losses[rnd-1])
        return aggregated_metrics

    def _compute_train_loss(self, results):
        train_losses = []
        examples = []
        for client_proxy, fit_res in results:
            num_examples = fit_res.num_examples
            metrics = fit_res.metrics
            loss = metrics.get("train_loss", float('inf'))
            if not np.isnan(loss):
                train_losses.append(num_examples * loss)
                examples.append(num_examples)
        if sum(examples) == 0:
            return float('nan')
        return sum(train_losses) / sum(examples)

    def _compute_val_loss(self, results):
        val_losses = []
        examples = []
        for client_proxy, eval_res in results:
            num_examples = eval_res.num_examples
            metrics = eval_res.metrics
            loss = metrics.get("val_loss", float('inf'))
            if not np.isnan(loss):
                val_losses.append(num_examples * loss)
                examples.append(num_examples)
        if sum(examples) == 0:
            return float('nan')
        return sum(val_losses) / sum(examples)

    def _save_history(self, rnd, train_loss=None, val_loss=None):
        history = {}
        if os.path.exists('history.json'):
            with open('history.json', 'r') as f:
                history = json.load(f)
        
        history[str(rnd)] = {}
        if train_loss is not None:
            history[str(rnd)]['train_loss'] = train_loss
        if val_loss is not None:
            history[str(rnd)]['val_loss'] = val_loss
        
        with open('history.json', 'w') as f:
            json.dump(history, f)

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""
    num_rounds = context.run_config["num-server-rounds"]
    num_items = 9724  # Number of items in MovieLens
    net = VAE(num_items=num_items)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)
    strategy = CustomFedAvg(
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=val_weighted_average,
        fit_metrics_aggregation_fn=fit_weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)