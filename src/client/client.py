from flwr.common import Context, NDArrays
from flwr.client import NumPyClient, ClientApp, Client
from typing import Dict, Tuple, Any
import logging
import sys
from flwr.common import GetParametersIns, GetParametersRes, Status, Code
import numpy as np
from flwr.common import ndarrays_to_parameters

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
from src.utils.metrics import train, test
from src.utils.preference_evolution import PreferenceEvolution
from src.utils.model_utils import set_weights, get_weights
import torch

class MovieLensClient(NumPyClient):
    def __init__(
        self,
        trainloader,
        testloader,
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
        preference_init_scale: float = 0.3
    ):
        self.trainloader = trainloader
        self.testloader = testloader
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

    def fit(self, parameters, config) -> Tuple[NDArrays, int, Dict]:
        """Train model parameters on local data."""
        set_weights(self.model, parameters)
        logger.info(f"Starting training with config: {config}")
        
        # Log data info
        for batch in self.trainloader:
            if self.model_type == 'mf':
                user_ids, item_ids, _ = batch
                logger.info(f"Max user_id: {user_ids.max()}, num_users: {self.num_users}")
                logger.info(f"Max item_id: {item_ids.max()}, num_items: {self.num_items}")
            break
        
        # Track preference evolution during training
        for batch in self.trainloader:
            if self.model_type == 'mf':
                user_ids, item_ids, ratings = [b.to(self.device) for b in batch]
                items = self.model.item_factors.weight[item_ids]
                scores = self.model(user_ids, item_ids)
            else:  # VAE
                items = batch.to(self.device)
                scores, _, _ = self.model(items)
                
            self.preference_evolution.update_preferences(items, scores)
            
        metrics = train(
            model=self.model,
            train_loader=self.trainloader,
            optimizer=self.optimizer,
            device=self.device,
            epochs=self.local_epochs,
            model_type=self.model_type
        )
        
        # Add preference evolution metrics
        metrics.update({
            'ut_norm': self.preference_evolution.history['ut_norm'][-1],
            'likable_prob': self.preference_evolution.history['likable_prob'][-1],
            'nonlikable_prob': self.preference_evolution.history['nonlikable_prob'][-1],
            'correlated_mass': self.preference_evolution.history['correlated_mass'][-1]
        })
        
        logger.info(f"Training completed with metrics: {metrics}")
        return get_weights(self.model), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        """Evaluate model parameters on test data."""
        set_weights(self.model, parameters)
        
        # Calculate test metrics
        metrics = test(
            model=self.model,
            test_loader=self.testloader,
            device=self.device,
            top_k=self.top_k,
            model_type=self.model_type,
            num_items=self.num_items,
            user_map=self.dimensions['user_map'],
            temperature=self.temperature,
            negative_penalty=self.negative_penalty,
            popularity_penalty=self.popularity_penalty
        )
        
        # Add preference evolution metrics
        metrics.update({
            'ut_norm': self.preference_evolution.history['ut_norm'][-1] if self.preference_evolution.history['ut_norm'] else 0.0,
            'likable_prob': self.preference_evolution.history['likable_prob'][-1] if self.preference_evolution.history['likable_prob'] else 0.0,
            'nonlikable_prob': self.preference_evolution.history['nonlikable_prob'][-1] if self.preference_evolution.history['nonlikable_prob'] else 0.0,
            'correlated_mass': self.preference_evolution.history['correlated_mass'][-1] if self.preference_evolution.history['correlated_mass'] else 0.0
        })
        
        logger.info(f"Evaluation metrics: {metrics}")
        return float(metrics["test_loss"]), len(self.testloader.dataset), metrics

"""Create and configure client application."""
def client_fn(context: Context) -> Client:
    """Create a MovieLens client."""
    config = context.run_config
    model_type = config["model-type"]

    # Get data with dimensions
    trainloader, testloader, dimensions = load_data(model_type=model_type)

    numpy_client = MovieLensClient(
        trainloader=trainloader,
        testloader=testloader,
        model_type=model_type,
        num_users=dimensions['num_users'],
        num_items=dimensions['num_items'],
        learning_rate=float(config["learning-rate"]),
        local_epochs=int(config["local-epochs"]),
        top_k=int(config["top-k"]),
        device=torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() and model_type == 'vae'
                          else "cpu"),
        dimensions=dimensions,
        temperature=float(config["temperature"]),
        negative_penalty=float(config["negative-penalty"]),
        popularity_penalty=float(config["popularity-penalty"]),
        beta=float(config["beta"]),
        gamma=float(config["gamma"]),
        learning_rate_schedule=config["learning-rate-schedule"],
        preference_init_scale=float(config["preference-init-scale"])
    )
    return numpy_client.to_client()

# Create the app instance
app = ClientApp(client_fn=client_fn)