import pytest
import torch
from src.client.client import MovieLensClient
from torch.utils.data import DataLoader, TensorDataset
import copy
from unittest.mock import patch

@pytest.fixture
def mock_data():
    # Create mock user-item interaction matrix
    num_users, num_items = 100, 50
    
    # Create sparse interaction matrix (1s for interactions, 0s for no interactions)
    interactions = torch.zeros((num_users, num_items))
    # Randomly set some interactions (1s) - about 5% density
    num_interactions = int(0.05 * num_users * num_items)
    user_indices = torch.randint(0, num_users, (num_interactions,))
    item_indices = torch.randint(0, num_items, (num_interactions,))
    interactions[user_indices, item_indices] = 1.0
    
    # Split into train and test
    train_data = interactions[:80]  # First 80 users for training
    test_data = interactions[80:]   # Last 20 users for testing
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=32,
        shuffle=True
    )
    
    test_loader = DataLoader(
        TensorDataset(test_data),
        batch_size=32,
        shuffle=False
    )
    
    return train_loader, test_loader

class TestMovieLensClient:
    def test_client_initialization(self, mock_data):
        train_loader, test_loader = mock_data
        client = MovieLensClient(
            trainloader=train_loader,
            testloader=test_loader,
            model_type='vae',
            num_items=50,
            num_users=100,
            learning_rate=0.001,
            local_epochs=1,
            top_k=10
        )
        
        assert client.model_type == 'vae'
        assert client.num_items == 50
        assert client.num_users == 100

    @pytest.mark.integration
    def test_client_fit(self, mock_data):
        train_loader, test_loader = mock_data
        client = MovieLensClient(
            trainloader=train_loader,
            testloader=test_loader,
            model_type='vae',
            num_items=50,
            num_users=100,
            learning_rate=0.001,
            local_epochs=1,
            top_k=10
        )
        
        # Get initial parameters as a list of NumPy arrays from the state dict values
        initial_params = [val.detach().cpu().numpy() for val in client.model.state_dict().values()]
        
        # Mock the train function to avoid data loading issues
        def mock_train(*args, **kwargs):
            return {
                'train_loss': 0.5,
                'val_loss': 0.6,
                'rmse': 0.4,
                'precision_at_k': 0.3,
                'recall_at_k': 0.2,
                'ndcg_at_k': 0.1,
                'coverage': 0.8,
                'roc_auc': 0.7,
            }
        
        # Patch the train function
        with patch('src.client.client.train', mock_train):
            # Test fit method
            updated_params, num_examples, metrics = client.fit(initial_params, {})
        
        assert len(updated_params) == len(initial_params)
        assert isinstance(num_examples, int)
        assert isinstance(metrics, dict)
