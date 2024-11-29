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

def server_fn(context: Context) -> ServerApp:
    """Create server instance with initial parameters."""
    # Get config values
    model_type = context.run_config["model-type"]
    num_rounds = context.run_config["num-server-rounds"]
    local_epochs = context.run_config["local-epochs"]
    top_k = context.run_config["top-k"]
    learning_rate = str(context.run_config["learning-rate"]).replace('.', '')
    temperature = context.run_config["temperature"]
    negative_penalty = str(context.run_config["negative-penalty"]).replace('.', '')
    popularity_penalty = str(context.run_config["popularity-penalty"]).replace('.', '')
    beta = str(context.run_config["beta"]).replace('.', '')
    gamma = str(context.run_config["gamma"]).replace('.', '')
    learning_rate_schedule = context.run_config["learning-rate-schedule"]
    num_nodes = context.run_config.get("num-nodes", 2)
    
    # Create history filename based on parameters
    history_filename = (
        f"{num_rounds}-rounds_{local_epochs}-epochs_{top_k}-topk"
        f"_{learning_rate}-lr_{temperature}-temp_{negative_penalty}-negpen"
        f"_{popularity_penalty}-poppen_{beta}-beta_{gamma}-gamma"
        f"_{learning_rate_schedule}-lrsched_{num_nodes}-nodes"
    )
    
    # Initialize histories
    histories = {
        'parameters': {
            'num_rounds': num_rounds,
            'local_epochs': local_epochs,
            'beta': float(context.run_config["beta"]),
            'gamma': float(context.run_config["gamma"]),
            'num_nodes': num_nodes,
            'learning_rate_schedule': learning_rate_schedule
        },
        'metrics': {
            'ut_norm': [],
            'likable_prob': [],
            'nonlikable_prob': [],
            'correlated_mass': []
        },
        'history': {
            'train_loss': [],
            'train_rmse': [],
            'test_loss': [], 
            'test_rmse': [], 
            'precision_at_k': [],
            'recall_at_k': [],
            'ndcg_at_k': [],
            'coverage': [],
            'rounds': []
        }
    }
    
    def on_exit():
        # Save consolidated history to single JSON file
        os.makedirs('histories', exist_ok=True)
        history_file = f'histories/history_{history_filename}.json'
        with open(history_file, 'w') as f:
            json.dump(histories, f)
        
        # Call the plot function after saving the history
        plot_metrics_from_file(history_file)
    
    atexit.register(on_exit)
    
    # Update strategy to collect histories
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        histories=histories
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds)
    )


app = ServerApp(server_fn=server_fn)