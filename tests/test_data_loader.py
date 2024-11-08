import pytest
from src.data.data_loader import load_data
from torch.utils.data import DataLoader

class TestDataLoader:
    @pytest.mark.integration
    def test_load_data_vae(self):
        train_loader, test_loader, num_items = load_data(num_users=100, model_type='vae')
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        assert isinstance(num_items, int)
        assert num_items > 0
        
        # Test batch format for VAE
        batch = next(iter(train_loader))
        assert len(batch.shape) == 2
        assert batch.shape[1] == num_items

    @pytest.mark.integration
    def test_load_data_mf(self):
        train_loader, test_loader, num_items = load_data(num_users=100, model_type='mf')
        
        # Test batch format for Matrix Factorization
        users, items, ratings = next(iter(train_loader))
        assert len(users.shape) == 1
        assert len(items.shape) == 1
        assert len(ratings.shape) == 1
