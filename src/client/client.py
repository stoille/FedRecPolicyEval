from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import torch
from src.models.vae import VAE
from src.models.matrix_factorization import MatrixFactorization
from src.utils.model_utils import get_weights, set_weights
from src.utils.metrics import train, test, train_mf, test_mf
from src.utils.visualization import visualize_latent_space
from src.data.data_loader import load_data

class MovieLensClient(NumPyClient):
    def __init__(self, trainloader, testloader, local_epochs, learning_rate, 
                 num_items, top_k, model_type, num_users):
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cpu")  # Force CPU for sparse operations
        self.num_items = num_items
        self.top_k = top_k
        self.model_type = model_type
        self.num_users = num_users
        
        # Initialize appropriate model
        if self.model_type == "mf":
            self.model = MatrixFactorization(num_users=num_users, num_items=num_items)
        else:
            self.model = VAE(num_items=num_items)

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        if self.model_type == "mf":
            train_mf(
                self.model, 
                self.trainloader, 
                self.local_epochs, 
                self.lr, 
                self.device
            )
        else:
            set_weights(self.model, parameters)
            train(
                self.model, 
                self.trainloader, 
                self.local_epochs, 
                self.lr, 
                self.device, 
                self.num_items
            )
            visualize_latent_space(self.model, self.testloader, self.device)

        return get_weights(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the local test dataset."""
        if self.model_type == "mf":
            metrics = test_mf(
                self.model, 
                self.testloader, 
                self.device, 
                top_k=self.top_k, 
                total_items=self.num_items
            )
        else:
            set_weights(self.model, parameters)
            metrics = test(
                self.model, 
                self.testloader, 
                self.device, 
                self.top_k, 
                self.num_items
            )

        return float(metrics["val_loss"]), len(self.testloader.dataset), metrics

"""Create and configure client application."""
def client_fn(context: Context) -> NumPyClient:
    # Get parameters from context
    model_type = context.run_config["model-type"]
    num_items = context.run_config["num-items"]
    learning_rate = context.run_config["learning-rate"]
    local_epochs = context.run_config["local-epochs"]
    top_k = context.run_config["top-k"]
    num_users = context.run_config["num-users"]

    # Create data loaders
    trainloader, testloader = load_data(
        num_items=num_items,
        num_users=num_users
    )

    # Create and return client
    return MovieLensClient(
        trainloader=trainloader,
        testloader=testloader,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        num_items=num_items,
        top_k=top_k,
        model_type=model_type,
        num_users=num_users
    )

def create_client_app() -> ClientApp:    
    # Return ClientApp instance
    return ClientApp(client_fn=client_fn)