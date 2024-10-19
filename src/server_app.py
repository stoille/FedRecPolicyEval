"""FedRecPolicyEval: A Flower / PyTorch package for evaluating recommender system policies."""

import json
import os
import numpy as np
from src.task import VAE, get_weights
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from typing import List, Tuple

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
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        return aggregated_weights

    def aggregate_evaluate(self, rnd, results, failures):
        metrics = aggregate_metrics([(res.num_examples, res.metrics) for _, res in results])
        self.history[rnd] = metrics
        self._save_history()
        return super().aggregate_evaluate(rnd, results, failures)

    def _save_history(self):
        with open('history.json', 'w') as f:
            json.dump(self.history, f)

def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = context.run_config["num-server-rounds"]
    num_items = 9724
    net = VAE(num_items=num_items)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)
    strategy = CustomFedAvg(
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
