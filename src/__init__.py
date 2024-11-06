"""FedRecPolicyEval: A Flower / PyTorch package for evaluating recommender system policies."""

from src.server.server import create_server_app
from src.client.client import create_client_app

__all__ = ["create_server_app", "create_client_app"]
