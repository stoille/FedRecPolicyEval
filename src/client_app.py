"""FedRecPolicyEval: A Flower / PyTorch package for evaluating recommender system policies."""

import torch
from src.task import VAE, get_weights, load_data, set_weights, test, train

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context


class MovieLensClient(NumPyClient):
    def __init__(self, trainloader, testloader, local_epochs, learning_rate, num_items):
        self.net = VAE(num_items=num_items)
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            epochs=self.local_epochs,
            learning_rate=self.lr,
            device=self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results
    
    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        val_loss = test(self.net, self.testloader, self.device)
        return float(val_loss), len(self.testloader.dataset), {"val_loss": float(val_loss)}

def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read the run_config to fetch hyperparameters relevant to this run
    trainloader, testloader, num_items = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    return MovieLensClient(
        trainloader, testloader, local_epochs, learning_rate, num_items
    ).to_client()


app = ClientApp(client_fn=client_fn)
