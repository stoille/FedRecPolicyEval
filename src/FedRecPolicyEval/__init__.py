"""FedRecPolicyEval package."""
from .server.server import create_server_app
from .client.client import create_client_app

__all__ = ["create_server_app", "create_client_app"]