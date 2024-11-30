import pytest
from src.data.data_loader import load_data
from torch.utils.data import DataLoader

class TestDataLoader:
    @pytest.mark.integration
    def eval_load_data_vae(self):
        train_loader, eval_loader, metadata = load_data(model_type='vae')
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(eval_loader, DataLoader)
        assert isinstance(metadata, dict)
        assert 'num_items' in metadata
        assert 'num_users' in metadata
        assert isinstance(metadata['num_items'], int)
        assert isinstance(metadata['num_users'], int)

    @pytest.mark.integration
    def eval_load_data_mf(self):
        train_loader, eval_loader, metadata = load_data(model_type='mf')
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(eval_loader, DataLoader)
        assert isinstance(metadata, dict)
        assert 'num_items' in metadata
        assert 'num_users' in metadata
        assert isinstance(metadata['num_items'], int)
        assert isinstance(metadata['num_users'], int)
        
        # Test batch format for Matrix Factorization
        users, items, ratings = next(iter(train_loader))
        assert len(users.shape) == 1
        assert len(items.shape) == 1
        assert len(ratings.shape) == 1
