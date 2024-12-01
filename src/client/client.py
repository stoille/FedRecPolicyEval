from flwr.common import Context, NDArrays
from flwr.client import NumPyClient, ClientApp, Client
from typing import Dict, Tuple, Any
import logging
import sys
from flwr.common import GetParametersIns, GetParametersRes, Status, Code, Scalar
import numpy as np
from flwr.common import ndarrays_to_parameters
import json

# Configure logging at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MovieLensClient")

from src.data.data_loader import load_data
from src.models.matrix_factorization import MatrixFactorization
from src.models.vae import VAE
from src.utils.metrics import train, evaluate_fn
from src.utils.preference_evolution import PreferenceEvolution
from src.utils.model_utils import set_weights, get_weights
import torch
import torch.nn.functional as F

class MovieLensClient(NumPyClient):
    def __init__(
        self,
        trainloader,
        eval_loader,
        model_type: str,
        num_users: int,
        num_items: int,
        learning_rate: float,
        local_epochs: int,
        top_k: int,
        device: str,
        dimensions: dict,
        temperature: float,
        negative_penalty: float,
        popularity_penalty: float,
        beta: float,
        gamma: float,
        learning_rate_schedule: str,
        client_id: int,
        preference_init_scale: float = 0.3,
        num_nodes: int = 1,
        num_rounds: int = 100
    ):
        self.trainloader = trainloader
        self.eval_loader = eval_loader
        self.model_type = model_type
        self.num_users = num_users
        self.num_items = num_items
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.top_k = top_k
        self.device = device
        self.dimensions = dimensions
        self.temperature = temperature
        self.negative_penalty = negative_penalty
        self.popularity_penalty = popularity_penalty
        self.client_id = client_id
        self.beta = beta
        self.gamma = gamma
        self.learning_rate_schedule = learning_rate_schedule
        self.num_nodes = num_nodes
        self.num_rounds = num_rounds
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
        
        # Initialize model with provided dimensions
        if model_type == 'mf':
            self.model = MatrixFactorization(
                num_users=self.num_users,
                num_items=self.num_items,
                n_factors=100
            ).to(device)
            logger.info(f"MF Model embeddings - Users: {self.model.user_factors.num_embeddings}, Items: {self.model.item_factors.num_embeddings}")
        else:
            self.model = VAE(num_items=self.num_items).to(device)
            
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
                lr=learning_rate
            )
        
        self.preference_evolution = PreferenceEvolution(
            initial_preferences=torch.randn(dimensions['num_items'], device=self.device) * preference_init_scale,
            beta=beta,
            gamma=gamma,
            learning_rate_schedule=learning_rate_schedule
        )
    
    def get_parameters(self, config) -> NDArrays:
        """Get model parameters as a list of NumPy arrays."""
        return get_weights(self.model)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train model parameters on local data."""
        set_weights(self.model, parameters)
        logger.info(f"Starting training with config: {config}")
        
        # Get training metrics
        train_metrics = train(
            model=self.model,
            train_loader=self.trainloader,
            optimizer=self.optimizer,
            device=self.device,
            epochs=self.local_epochs,
            model_type=self.model_type,
            preference_evolution=self.preference_evolution
        )
        
        # Finalize round-level metrics
        round_preferences = self.preference_evolution.finalize_round()
        
        # Provide default values if round_preferences is None
        if round_preferences is None:
            round_preferences = {
                'ut_norm': [],
                'likable_prob': [],
                'nonlikable_prob': [],
                'correlated_mass': []
            }
        
        # Get experiment config from client init params
        config_object = {
            'num_nodes': self.num_nodes,
            'rounds': self.num_rounds,
            'epochs': self.local_epochs,
            'lr': self.learning_rate,
            'beta': self.preference_evolution.beta,
            'gamma': self.preference_evolution.gamma,
            'temp': self.temperature,
            'neg_pen': self.negative_penalty,
            'pop_pen': self.popularity_penalty,
            'lr_schedule': self.preference_evolution.learning_rate_schedule
        }
        
        epochs_file = f'metrics/epochs_{self.metrics_prefix}.json'
        
        # Update epochs file
        try:
            with open(epochs_file, 'r') as f:
                all_epoch_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_epoch_data = {
                'config': config_object,
                'metrics': {}
            }
        
        # Initialize client entry if it doesn't exist
        client_key = f"client_{self.client_id}"
        if client_key not in all_epoch_data['metrics']:
            all_epoch_data['metrics'][client_key] = {
                'train_ut_norm': [],
                'train_likable_prob': [],
                'train_nonlikable_prob': [],
                'train_correlated_mass': [],
                'train_loss': [],
                'train_rmse': []
            }
        
        # Append metrics as single values to flatten the arrays
        all_epoch_data['metrics'][client_key]['train_ut_norm'].extend(train_metrics['ut_norm'])
        all_epoch_data['metrics'][client_key]['train_likable_prob'].extend(train_metrics['likable_prob'])
        all_epoch_data['metrics'][client_key]['train_nonlikable_prob'].extend(train_metrics['nonlikable_prob'])
        all_epoch_data['metrics'][client_key]['train_correlated_mass'].extend(train_metrics['correlated_mass'])
        all_epoch_data['metrics'][client_key]['train_loss'].extend(train_metrics['train_loss'])
        all_epoch_data['metrics'][client_key]['train_rmse'].extend(train_metrics['train_rmse'])
        
        with open(epochs_file, 'w') as f:
            json.dump(all_epoch_data, f)

        # Calculate final metrics for the round
        metrics = {
            'train_loss': float(train_metrics['train_loss'][-1]),
            'train_rmse': float(train_metrics['train_rmse'][-1]),
            'ut_norm': float(train_metrics['ut_norm'][-1]),
            'likable_prob': float(train_metrics['likable_prob'][-1]),
            'nonlikable_prob': float(train_metrics['nonlikable_prob'][-1]),
            'correlated_mass': float(train_metrics['correlated_mass'][-1])
        }
        
        print(f"Client {self.client_id} returning fit results with metrics: {metrics}")
        return get_weights(self.model), len(self.trainloader.dataset), metrics

    def get_batch_predictions(self, batch):
        """Get model predictions for a batch."""
        self.model.eval()
        with torch.no_grad():
            items = batch.to(self.device)
            recon_x, _, _ = self.model(items)
            return items, recon_x

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model parameters on the locally held eval set."""
        
        """Evaluate model parameters on eval data."""
        set_weights(self.model, parameters)
        
        # Calculate eval metrics
        eval_metrics = evaluate_fn(
            model=self.model,
            eval_loader=self.eval_loader,
            device=self.device,
            top_k=self.top_k,
            model_type=self.model_type,
            num_items=self.num_items,
            user_map=self.dimensions['user_map'],
            temperature=self.temperature,
            negative_penalty=self.negative_penalty,
            popularity_penalty=self.popularity_penalty
        )
        
        # Process test batches for preference evolution
        for batch in self.eval_loader:
            items, scores = self.get_batch_predictions(batch)
            self.preference_evolution.update_preferences(items, scores, is_round=True)
        
        # Calculate basic test metrics
        eval_loss = 0.0
        eval_rmse = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                items = batch.to(self.device)
                recon_x, mu, logvar = self.model(items)
                loss = self.model.loss_function(recon_x, items, mu, logvar)
                rmse = torch.sqrt(F.mse_loss(recon_x, items))
                
                eval_loss += loss.item()
                eval_rmse += rmse.item()
                num_batches += 1
        
        eval_metrics.update({
            'eval_loss': eval_loss / num_batches,
            'eval_rmse': eval_rmse / num_batches
        })
        
        # Finalize the round to get aggregated metrics
        preference_evolution_metrics = self.preference_evolution.finalize_round()
        
        # Add preference evolution metrics if available
        if preference_evolution_metrics:
            eval_metrics.update({
                'eval_ut_norm': preference_evolution_metrics['ut_norm'],
                'eval_likable_prob': preference_evolution_metrics['likable_prob'],
                'eval_nonlikable_prob': preference_evolution_metrics['nonlikable_prob'],
                'eval_correlated_mass': preference_evolution_metrics['correlated_mass']
            })
        
        return float(eval_metrics['eval_loss']), len(self.eval_loader.dataset), eval_metrics

"""Create and configure client application."""
def client_fn(context: Context) -> Client:
    """Create a MovieLens client."""
    config = context.run_config
    client_id = int(context.node_id)

    # Get data with dimensions
    trainloader, eval_loader, dimensions = load_data(model_type=config["model-type"])

    # Fix device setting to pass string instead of torch.device
    device_str = ("cuda" if torch.cuda.is_available() 
                 else "mps" if torch.backends.mps.is_available() and config["model-type"] == 'vae'
                 else "cpu")

    numpy_client = MovieLensClient(
        trainloader=trainloader,
        eval_loader=eval_loader,
        model_type=config["model-type"],
        num_users=dimensions['num_users'],
        num_items=dimensions['num_items'],
        learning_rate=float(config["learning-rate"]),
        local_epochs=int(config["local-epochs"]),
        top_k=int(config["top-k"]),
        device=device_str,
        dimensions=dimensions,
        temperature=float(config["temperature"]),
        negative_penalty=float(config["negative-penalty"]),
        popularity_penalty=float(config["popularity-penalty"]),
        beta=float(config["beta"]),
        gamma=float(config["gamma"]),
        learning_rate_schedule=config["learning-rate-schedule"],
        client_id=client_id,
        preference_init_scale=float(config["preference-init-scale"]),
        num_nodes=int(config["num-nodes"]),
        num_rounds=int(config["num-server-rounds"])
    )
    numpy_client.client_id = client_id  # Set client_id after initialization
    return numpy_client.to_client()

# Create the app instance
app = ClientApp(client_fn=client_fn)