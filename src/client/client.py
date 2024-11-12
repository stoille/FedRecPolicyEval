from flwr.common import Context, NDArrays
from flwr.client import NumPyClient, ClientApp, Client
from typing import Dict, Tuple, Any
import logging
import sys

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
from src.utils.model_utils import set_weights, get_weights
import torch

class MovieLensClient(NumPyClient):
    def __init__(
        self,
        trainloader,
        testloader,
        model_type: str,
        num_items: int,
        num_users: int,
        learning_rate: float,
        local_epochs: int,
        top_k: int,
        device: str = torch.device("cuda" if torch.cuda.is_available() 
                                 else "mps" if torch.backends.mps.is_available() 
                                 else "cpu")
    ):
        self.trainloader = trainloader
        self.testloader = testloader
        self.model_type = model_type
        self.num_items = num_items
        self.num_users = num_users
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.top_k = top_k
        self.device = device
        
        # Initialize model based on type
        if model_type == 'mf':
            self.model = MatrixFactorization(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=100
            ).to(device)
        else:
            self.model = VAE(num_items=num_items).to(device)
            
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

    def fit(self, parameters, config) -> Tuple[NDArrays, int, Dict]:
        """Train model parameters on local data."""
        set_weights(self.model, parameters)
        logger.info(f"Starting training with config: {config}")
        
        metrics = train(
            model=self.model,
            train_loader=self.trainloader,
            optimizer=self.optimizer,
            device=self.device,
            epochs=self.local_epochs,
            model_type=self.model_type
        )
        
        logger.info(f"Training completed with metrics: {metrics}")
        return get_weights(self.model), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        """Evaluate model parameters on test data."""
        set_weights(self.model, parameters)
        logger.info("Starting evaluation")
        
        metrics = test(
            model=self.model,
            test_loader=self.testloader,
            device=self.device,
            top_k=self.top_k,
            model_type=self.model_type,
            num_items=self.num_items
        )
        
        logger.info(f"Evaluation completed with metrics: {metrics}")
        return float(metrics["test_loss"]), len(self.testloader.dataset), metrics

"""Create and configure client application."""
def client_fn(context: Context) -> Client:
    """Create a MovieLens client."""
    config = context.run_config

    trainloader, testloader, num_items = load_data(
        num_users=int(config["num-users"]),
        model_type=config["model-type"]
    )

    # Create NumPyClient instance
    numpy_client = MovieLensClient(
        trainloader=trainloader,
        testloader=testloader,
        model_type=config["model-type"],
        num_items=num_items,
        num_users=int(config["num-users"]),
        learning_rate=float(config["learning-rate"]),
        local_epochs=int(config["local-epochs"]),
        top_k=int(config["top-k"]),
        device=torch.device("cuda" if torch.cuda.is_available() 
                                 else "mps" if torch.backends.mps.is_available() 
                                 else "cpu")
    )
    
    # Convert to Client and return
    return numpy_client.to_client()

# Create the app instance
app = ClientApp(client_fn=client_fn)