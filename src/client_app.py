"""FedRecPolicyEval: A Flower / PyTorch package for evaluating recommender system policies."""

import time
import torch
from src.task import VAE, get_weights, set_weights, train, test, load_data, popularity_baseline, visualize_latent_space, MatrixFactorization, train_mf, test_mf
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

class MovieLensClient(NumPyClient):
    def __init__(self, trainloader, testloader, local_epochs, learning_rate, num_items, top_k, model_type, num_users):
        self.net = VAE(num_items=num_items)
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        # self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # Force CPU usage since MPS doesn't fully support sparse operations when running on (Metal Macbook M1/M2)
        self.device = torch.device("cpu")
        self.num_items = num_items
        self.top_k = top_k
        self.model_type = model_type
        self.num_users = num_users

    def fit(self, parameters, config):
        if self.model_type == "mf":
            num_users = self.num_users
            model = MatrixFactorization(num_users, self.num_items)
            train_mf(model, self.trainloader, self.local_epochs, self.lr, self.device)
        else:
            set_weights(self.net, parameters)
            train(self.net, self.trainloader, self.local_epochs, self.lr, self.device, self.num_items)
            visualize_latent_space(self.net, self.testloader, self.device)

        return get_weights(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # Check if we are evaluating the VAE or MF model
        if self.model_type == "mf":
            num_users = self.num_users
            model = MatrixFactorization(num_users, self.num_items)
            metrics = test_mf(model, self.testloader, self.device, 
                             top_k=self.top_k, total_items=self.num_items)
        elif self.model_type == "vae":
            set_weights(self.net, parameters)
            metrics = test(self.net, self.testloader, self.device, 
                          self.top_k, self.num_items)
        
        # Get baseline metrics
        # baseline_metrics = popularity_baseline(self.trainloader, self.testloader, top_k=self.top_k)

        return float(metrics["val_loss"]), len(self.testloader.dataset), metrics

def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    model_type = context.run_config["model-type"]
    
    trainloader, testloader, num_items, num_users = load_data(
        partition_id, 
        num_partitions,
        mode='mf' if model_type == 'mf' else 'vae'
    )
    
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    top_k = context.run_config["top-k"]
    
    return MovieLensClient(
        trainloader, testloader, local_epochs, learning_rate, 
        num_items, top_k, model_type, num_users
    ).to_client()

app = ClientApp(client_fn=client_fn)
