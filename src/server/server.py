from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import ndarrays_to_parameters, Context
from .strategy import CustomFedAvg
from src.models.vae import VAE
from src.models.matrix_factorization import MatrixFactorization
from src.utils.model_utils import get_weights
import atexit
from src.utils.visualization import plot_metrics_from_files
from typing import Dict, Any
import atexit
import torch
from src.data import load_data

def server_fn(context: Context) -> ServerApp:
    """Create server instance with initial parameters."""
    # Get config values
    model_type = context.run_config["model-type"]
    num_rounds = context.run_config["num-server-rounds"]
    local_epochs = context.run_config["local-epochs"]
    top_k = context.run_config["top-k"]
    learning_rate = str(context.run_config["learning-rate"]).replace('.', '')
    
    # Load data with dimensions
    trainloader, _, dimensions = load_data(model_type=model_type)
    
    # Add dimensions to run_config (not config)
    context.run_config["num-users"] = dimensions['num_users']
    context.run_config["num-items"] = dimensions['num_items']
    
    # Create initial model with loaded dimensions
    model = (
        MatrixFactorization(num_users=dimensions['num_users'], num_items=dimensions['num_items'])
        if model_type == "mf"
        else VAE(num_items=dimensions['num_items'], latent_dim=200)
    )
    
    # Create strategy with initial parameters
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=ndarrays_to_parameters(get_weights(model)),
    )
    
    # Update plot filename with client count after training
    plot_filename = f"{model_type}_{num_rounds}-rounds_{local_epochs}-epochs_{top_k}-topk_{learning_rate}-lr"
    def on_exit():
        plot_metrics_from_files(
            'train_history.json',
            'test_history.json',
            f"{plot_filename}_{strategy.num_clients}-nodes"
        )
    
    atexit.register(on_exit)

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds)
    )


app = ServerApp(server_fn=server_fn)