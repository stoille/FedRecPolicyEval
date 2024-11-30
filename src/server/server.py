from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import ndarrays_to_parameters, Context
from .strategy import CustomFedAvg
from src.models.vae import VAE
from src.models.matrix_factorization import MatrixFactorization
from src.utils.model_utils import get_weights
import atexit
from src.utils.visualization import plot_metrics_from_file
from typing import Dict, Any
import atexit
import torch
from src.data import load_data
import json
import os
import glob

def server_fn(context: Context) -> ServerApp:
    """Create server instance with initial parameters."""
    # Get config values
    model_type = context.run_config["model-type"]
    num_rounds = context.run_config["num-server-rounds"]
    local_epochs = context.run_config["local-epochs"]
    top_k = context.run_config["top-k"]
    learning_rate = str(context.run_config["learning-rate"])
    temperature = str(context.run_config["temperature"])
    negative_penalty = str(context.run_config["negative-penalty"])
    popularity_penalty = str(context.run_config["popularity-penalty"])
    beta = str(context.run_config["beta"])
    gamma = str(context.run_config["gamma"])
    learning_rate_schedule = context.run_config["learning-rate-schedule"]
    num_nodes = context.run_config.get("num-nodes", 2)
    
    # Create metrics filename prefix
    metrics_prefix = (
        f"lr={learning_rate}_"
        f"beta={beta}_"
        f"gamma={gamma}_"
        f"temp={temperature}_"
        f"negpen={negative_penalty}_"
        f"poppen={popularity_penalty}_"
        f"lrsched={learning_rate_schedule}"
    )
    
    # Clear existing metric files
    for file_type in ['metrics', 'rounds', 'epochs']:
        file_path = f'metrics/{file_type}_{metrics_prefix}.json'
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleared existing {file_path}")
    
    def on_exit():
        # Create metrics filename based on parameters
        metrics_prefix = (
            f"lr={learning_rate}_"
            f"beta={beta}_"
            f"gamma={gamma}_"
            f"temp={temperature}_"
            f"negpen={negative_penalty}_"
            f"poppen={popularity_penalty}_"
            f"lrsched={learning_rate_schedule}"
        )
        
        print(f"Starting consolidation for prefix: {metrics_prefix}")
        
        # Ensure metrics directory exists
        os.makedirs('metrics', exist_ok=True)
        
        # Initialize consolidated metrics
        consolidated_metrics = {
            'config': {
                'model': model_type,
                'lr': float(context.run_config["learning-rate"]),
                'beta': float(context.run_config["beta"]),
                'gamma': float(context.run_config["gamma"]),
                'temp': temperature,
                'neg_pen': float(negative_penalty),
                'pop_pen': float(popularity_penalty),
                'lr_schedule': learning_rate_schedule
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
                'train_ut_norm': [],
                'train_likable_prob': [],
                'train_nonlikable_prob': [],
                'train_correlated_mass': []
            }
        }
        
        # Load rounds file
        rounds_file = f'metrics/rounds_{metrics_prefix}.json'
        print(f"Looking for rounds file: {rounds_file}")
        if os.path.exists(rounds_file):
            print(f"Found rounds file")
            with open(rounds_file, 'r') as f:
                rounds_data = json.load(f)
                print(f"Rounds data: {rounds_data}")
                if 'metrics' in rounds_data:
                    for metric_name, values in rounds_data['metrics'].items():
                        if metric_name in consolidated_metrics['metrics']:
                            # Handle both list and float values
                            if isinstance(values, (list, tuple)):
                                consolidated_metrics['metrics'][metric_name].extend(values)
                            else:
                                consolidated_metrics['metrics'][metric_name].append(values)
                            print(f"Added values to {metric_name}")
        else:
            print("Rounds file not found")
            
        # Load epochs file
        epochs_file = f'metrics/epochs_{metrics_prefix}.json'
        print(f"Looking for epochs file: {epochs_file}")
        if os.path.exists(epochs_file):
            print(f"Found epochs file")
            with open(epochs_file, 'r') as f:
                epochs_data = json.load(f)
                print(f"Epochs data: {epochs_data}")
                if 'metrics' in epochs_data:
                    # Iterate through each round/client combination
                    for round_client_data in epochs_data['metrics'].values():
                        # Add each metric to the consolidated metrics
                        for metric_name, values in round_client_data.items():
                            if metric_name in consolidated_metrics['metrics']:
                                # Handle both list and float values
                                if isinstance(values, (list, tuple)):
                                    consolidated_metrics['metrics'][metric_name].extend(values)
                                else:
                                    consolidated_metrics['metrics'][metric_name].append(values)
                                print(f"Added values to {metric_name}")
        else:
            print("Epochs file not found")
        
        # Save consolidated metrics
        output_file = f'metrics/metrics_{metrics_prefix}.json'
        print(f"Saving consolidated metrics to: {output_file}")
        print(f"Final metrics: {consolidated_metrics}")
        with open(output_file, 'w') as f:
            json.dump(consolidated_metrics, f)
        
        # Call the plot function after saving
        plot_metrics_from_file(output_file)
        
        # Clean up intermediate files
        #3for f in [epochs_file, rounds_file]:
        #    if os.path.exists(f):
        #        os.remove(f)
        #        print(f"Cleaned up {f}")
    
    atexit.register(on_exit)
    
    # Create strategy without histories parameter
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        model_type=model_type,
        learning_rate=learning_rate,
        beta=beta,
        gamma=gamma,
        temperature=temperature,
        negative_penalty=negative_penalty,
        popularity_penalty=popularity_penalty,
        learning_rate_schedule=learning_rate_schedule
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds)
    )


app = ServerApp(server_fn=server_fn)