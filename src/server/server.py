from flwr.server import ServerApp, Server
from flwr.common import ndarrays_to_parameters, Context
from src.models.vae import VAE
from src.models.matrix_factorization import MatrixFactorization
from src.utils.model_utils import get_weights
from .strategy import CustomFedAvg
import atexit
from src.utils.visualization import plot_history
from typing import Dict, Any


def get_initial_parameters(num_items, model_type, context):
    """Get initial model parameters based on model type."""
    if model_type == "mf":
        num_users = context.run_config.get("num-users", 0)
        model = MatrixFactorization(num_users=num_users, num_items=num_items)
    else:  # default to VAE
        model = VAE(num_items=num_items)
    
    ndarrays = get_weights(model)
    return ndarrays_to_parameters(ndarrays)

def server_fn(context: Context) -> ServerApp:
    num_rounds = context.run_config["num-server-rounds"]
    num_items = context.run_config["num-items"] #9724
    net = VAE(num_items=num_items)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)
    strategy = CustomFedAvg(
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerApp(strategy=strategy, config=config)

import atexit

def create_server_app() -> ServerApp:
    # Return ServerApp instance
    return ServerApp(server_fn=server_fn)

def cleanup(strategy):
    """Perform cleanup operations."""
    print("Performing cleanup operations")
    if hasattr(strategy, 'history'):
        plot_history(strategy.history)