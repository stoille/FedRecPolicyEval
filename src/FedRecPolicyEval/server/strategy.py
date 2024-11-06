from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    EvaluateRes
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

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
        model_type: str = "vae"
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
        self.history: Dict[int, Dict[str, float]] = {}
        self.model_type = model_type

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted averaging."""
        if not results:
            return None, {}
        
        if self.model_type == "mf":
            # Matrix Factorization might need special handling for sparse gradients
            parameters_aggregated, metrics_aggregated = self.aggregate_mf_parameters(
                results
            )
        else:
            # Use default FedAvg for VAE
            parameters_aggregated, metrics_aggregated = super().aggregate_fit(
                rnd, results, failures
            )

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics across all clients."""
        if not results:
            return None, {}

        # Aggregate metrics
        metrics = self.aggregate_metrics([(res.num_examples, res.metrics) 
                                        for _, res in results])
        
        # Store metrics in history
        self.history[rnd] = metrics

        # Calculate weighted loss average
        loss_aggregated = weighted_loss_avg(
            [(res.num_examples, res.loss) for _, res in results]
        )

        return loss_aggregated, metrics

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