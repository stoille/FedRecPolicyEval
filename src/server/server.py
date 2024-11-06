from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import ndarrays_to_parameters, Context
from src.models.vae import VAE
from src.models.matrix_factorization import MatrixFactorization
from src.utils.model_utils import get_weights
from .strategy import CustomFedAvg
import atexit
from src.utils.visualization import plot_history
from typing import Dict, Any
import atexit


def server_fn(context: Context) -> ServerApp:
    """Create server instance with initial parameters."""
    # Get config values
    num_items = context.run_config["num-items"]
    num_users = context.run_config["num-users"]
    model_type = context.run_config["model-type"]
    num_rounds = context.run_config["num-server-rounds"]
    
    # Create initial model
    model = (
        MatrixFactorization(num_users=num_users, num_items=num_items)
        if model_type == "mf"
        else VAE(num_items=num_items, latent_dim=100)
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

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds)
    )


app = ServerApp(server_fn=server_fn)

def cleanup(strategy):
    """Perform cleanup operations."""
    print("Performing cleanup operations")
    if hasattr(strategy, 'history'):
        plot_history(strategy.history)