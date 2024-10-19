"""FedRecPolicyEval: A Flower / PyTorch package for evaluating recommender system policies."""

import time
import torch
from src.task import VAE, get_weights, set_weights, train, test, load_data
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
        self.num_items = num_items

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        start_time = time.time()
        results = train(
            self.net,
            self.trainloader,
            epochs=self.local_epochs,
            learning_rate=self.lr,
            device=self.device,
            total_items=self.num_items
        )
        end_time = time.time()
        convergence_speed = end_time - start_time
        results["convergence_speed"] = convergence_speed
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        metrics = test(
            self.net,
            self.testloader,
            self.device,
            k=10,
            total_items=self.num_items
        )
        return float(metrics["val_loss"]), len(self.testloader.dataset), metrics

def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, testloader, num_items = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    return MovieLensClient(
        trainloader, testloader, local_epochs, learning_rate, num_items
    ).to_client()

app = ClientApp(client_fn=client_fn)
