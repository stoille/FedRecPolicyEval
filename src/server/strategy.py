from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import json
import os

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
import logging

from src.utils import calculate_model_divergence

logger = logging.getLogger("Strategy")

class CustomFedAvg(FedAvg):
    """Customized Federated Averaging strategy with metrics tracking."""
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[bool] = None,
        on_fit_config_fn: Optional[bool] = None,
        on_evaluate_config_fn: Optional[bool] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        model_type: str = "vae",
        learning_rate: float = 0.0001,
        beta: float = 0.01,
        gamma: float = 0.01,
        temperature: float = 0.5,
        negative_penalty: float = 0.2,
        popularity_penalty: float = 0.1,
        learning_rate_schedule: str = "constant",
        num_rounds: int = 1,
        num_nodes: int = 1,
        local_epochs: int = 1
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.model_type = model_type
        self.current_round = 0
        self.num_clients = 0
        self.learning_rate = learning_rate
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.negative_penalty = negative_penalty
        self.popularity_penalty = popularity_penalty
        self.learning_rate_schedule = learning_rate_schedule
        self.num_rounds = num_rounds
        self.num_nodes = num_nodes
        self.local_epochs = local_epochs
        self.metrics_prefix = (
            f"num_nodes={self.num_nodes}_"
            f"rounds={self.num_rounds}_"
            f"epochs={self.local_epochs}_"
            f"lr={self.learning_rate}_"
            f"beta={self.beta}_"
            f"gamma={self.gamma}_"
            f"temp={self.temperature}_"
            f"negpen={self.negative_penalty}_"
            f"poppen={self.popularity_penalty}_"
            f"lrsched={self.learning_rate_schedule}"
        )
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted averaging and custom metrics."""
        if not results:
            return None, {}

        # Extract local models and calculate global model
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Calculate global model through aggregation
        total_examples = sum([num_examples for _, num_examples in weights_results])
        aggregated_weights = []
        for weights_list_tuple in zip(*[weights for weights, _ in weights_results]):
            aggregated_layer = sum(
                [(num_examples / total_examples) * layer
                 for layer, (_, num_examples) in zip(weights_list_tuple, weights_results)]
            )
            aggregated_weights.append(aggregated_layer)

        # Calculate divergence metrics
        local_models = [weights[0] for weights, _ in weights_results]  # First layer contains user preferences
        global_model = aggregated_weights[0]  # First layer of aggregated model
        
        divergence_metrics = calculate_model_divergence(local_models, global_model)
        
        logger.info(f"Divergence metrics: {divergence_metrics}")
        # Add divergence metrics to metrics_aggregated
        metrics_aggregated = {
            'local_global_divergence': divergence_metrics['local_global_divergence'],
            'personalization_degree': divergence_metrics['personalization_degree'],
            'max_local_divergence': divergence_metrics['max_local_divergence']
        }

        # Add logging before aggregation
        logger.info("Raw metrics from clients:")
        for _, fit_res in results:
            logger.info(f"Client metrics: {fit_res.metrics}")

        # Aggregate parameters
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        total_examples = sum([num_examples for _, num_examples in weights_results])

        # Weighted averaging of parameters
        aggregated_weights = []
        for weights_list_tuple in zip(*[weights for weights, _ in weights_results]):
            aggregated_layer = sum(
                [
                    (num_examples / total_examples) * layer
                    for layer, (_, num_examples) in zip(weights_list_tuple, weights_results)
                ]
            )
            aggregated_weights.append(aggregated_layer)

        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)

        # Aggregate metrics
        num_clients = len(results)
        
        for _, fit_res in results:
            for metric_name, value in fit_res.metrics.items():
                # Skip non-numeric metrics
                if isinstance(value, (int, float)):
                    if metric_name not in metrics_aggregated:
                        metrics_aggregated[metric_name] = 0.0
                    metrics_aggregated[metric_name] += float(value)

        # Average the metrics
        for name in metrics_aggregated:
            metrics_aggregated[name] /= num_clients

        # Add logging after aggregation
        logger.info(f"Aggregated metrics before saving: {metrics_aggregated}")
        
        metrics_file = f'metrics/rounds_{self.metrics_prefix}.json'
        os.makedirs('metrics', exist_ok=True)
        
        # Load existing metrics file
        try:
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_metrics = {
                'config': {
                    'num_nodes': self.num_nodes,
                    'rounds': self.num_rounds,
                    'epochs': self.local_epochs,
                    'lr': self.learning_rate,
                    'beta': self.beta,
                    'gamma': self.gamma,
                    'temp': self.temperature,
                    'neg_pen': self.negative_penalty,
                    'pop_pen': self.popularity_penalty,
                    'lr_schedule': self.learning_rate_schedule
                },
                'metrics': {
                    'train_loss': [],
                    'train_rmse': [],
                    'eval_loss': [],
                    'eval_rmse': [],
                    'precision_at_k': [],
                    'recall_at_k': [],
                    'ndcg_at_k': [],
                    'coverage': [],
                    'eval_ut_norm': [],
                    'eval_likable_prob': [],
                    'eval_nonlikable_prob': [],
                    'eval_correlated_mass': [],
                    'local_global_divergence': [],
                    'personalization_degree': [],
                    'max_local_divergence': []
                }
            }
        
        # Append new metrics to lists
        for metric_name, value in metrics_aggregated.items():
            if metric_name in all_metrics['metrics']:
                all_metrics['metrics'][metric_name].append(float(value))
        
        # Update metrics file with divergence metrics
        all_metrics['metrics'].update({
            'local_global_divergence': [],
            'personalization_degree': [],
            'max_local_divergence': []
        })
        
        for metric_name in ['local_global_divergence', 'personalization_degree', 'max_local_divergence']:
            if metric_name in metrics_aggregated:
                all_metrics['metrics'][metric_name].append(float(metrics_aggregated[metric_name]))
        
        # Add logging after loading/before saving
        logger.info(f"Current metrics state: {all_metrics}")
        logger.info(f"Saving metrics to: {metrics_file}")
        
        # Save updated metrics
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f)

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}

        metrics_aggregated = {
            'eval_loss': 0.0,
            'eval_rmse': 0.0,
            'precision_at_k': 0.0,
            'recall_at_k': 0.0,
            'ndcg_at_k': 0.0,
            'coverage': 0.0,
            'eval_ut_norm': 0.0,
            'eval_likable_prob': 0.0,
            'eval_nonlikable_prob': 0.0,
            'eval_correlated_mass': 0.0
        }
        num_clients = len(results)
        
        for _, eval_res in results:
            for name, value in eval_res.metrics.items():
                metrics_aggregated[name] += float(value)
        
        # Average the metrics
        for name in metrics_aggregated:
            metrics_aggregated[name] /= num_clients

        # Add logging for evaluation metrics
        logger.info("Raw evaluation metrics from clients:")
        for _, eval_res in results:
            logger.info(f"Client eval metrics: {eval_res.metrics}")

        logger.info(f"Aggregated evaluation metrics: {metrics_aggregated}")
        
        metrics_file = f'metrics/rounds_{self.metrics_prefix}.json'
        
        try:
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # File should exist from aggregate_fit, but just in case
            return float(metrics_aggregated['eval_loss']), metrics_aggregated
        
        # Append evaluation metrics to existing lists
        for name, value in metrics_aggregated.items():
            if name in all_metrics['metrics']:
                all_metrics['metrics'][name].append(float(value))
        
        # Add logging before saving
        logger.info(f"Final evaluation metrics state: {all_metrics}")
        
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f)
        
        return float(metrics_aggregated['eval_loss']), metrics_aggregated

    def aggregate_mf_parameters(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Tuple[Parameters, Dict[str, Scalar]]:
        """Special aggregation for Matrix Factorization parameters."""
        # Extract weights and number of examples
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Aggregate parameters (you might need special handling for sparse matrices)
        aggregated_ndarrays = [
            np.sum([weights[i] * num_examples for weights, num_examples in weights_results], axis=0)
            / sum(num_examples for _, num_examples in weights_results)
            for i in range(len(weights_results[0][0]))
        ]
        
        return ndarrays_to_parameters(aggregated_ndarrays), {}

    @staticmethod
    def aggregate_metrics(
        metrics: List[Tuple[int, Dict[str, Scalar]]]
    ) -> Dict[str, float]:
        """Aggregate metrics from multiple clients."""
        # Initialize aggregated metrics
        aggregated: Dict[str, List[Tuple[int, float]]] = {}
        
        # Collect metrics from all clients
        for num_examples, client_metrics in metrics:
            for metric_name, value in client_metrics.items():
                if metric_name not in aggregated:
                    aggregated[metric_name] = []
                aggregated[metric_name].append((num_examples, float(value)))
        
        # Calculate weighted averages
        return {
            metric_name: weighted_average(metric_values)
            for metric_name, metric_values in aggregated.items()
        }

def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Calculate weighted average of losses."""
    num_total_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_examples

def weighted_average(metrics: List[Tuple[int, float]]) -> float:
    """Calculate weighted average of metrics."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_sum = sum(num_examples * metric for num_examples, metric in metrics)
    return weighted_sum / total_examples if total_examples > 0 else 0.0