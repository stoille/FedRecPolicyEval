import pytest
import torch
import numpy as np
from src.utils.metrics import (
    compute_rmse,
    MetricsLogger,
    loss_function,
    compute_metrics,
    mean_squared_error,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    coverage,
    compute_roc_auc
)
from src.models.matrix_factorization import MatrixFactorization

@pytest.fixture
def metrics_logger():
    return MetricsLogger()

class TestMetrics:
    def test_compute_rmse(self):
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        rmse = compute_rmse(predictions, targets)
        assert rmse == 0.0
        
    def test_metrics_logger(self, metrics_logger):
        # Log training metrics
        train_metrics = {
            'epoch_train_loss': 0.5,
            'epoch_train_rmse': 0.3
        }
        metrics_logger.log_metrics(train_metrics, is_training=True)
        metrics_logger.train_history['rounds'].append(1)
        
        # Log test metrics
        test_metrics = {
            'round_test_loss': 0.6,
            'round_test_rmse': 0.4,
            'precision_at_k': 0.7,
            'recall_at_k': 0.6,
            'ndcg_at_k': 0.8
        }
        metrics_logger.log_metrics(test_metrics, is_training=False)
        
        # Assert training metrics
        assert len(metrics_logger.train_history['rounds']) == 1
        assert metrics_logger.train_history['epoch_train_loss'][0] == 0.5
        assert metrics_logger.train_history['epoch_train_rmse'][0] == 0.3
        
        # Assert test metrics
        assert len(metrics_logger.test_history['rounds']) == 1
        assert metrics_logger.test_history['precision_at_k'][0] == 0.7
        assert metrics_logger.test_history['recall_at_k'][0] == 0.6
        assert metrics_logger.test_history['ndcg_at_k'][0] == 0.8

    def round_test_loss_function(self):
        recon_x = torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32)
        x = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
        mu = torch.zeros(2, 2)
        logvar = torch.zeros(2, 2)
        
        loss = loss_function(recon_x, x, mu, logvar, epoch=1, num_epochs=10)
        assert isinstance(loss.item(), float)
        assert loss.item() > 0

    def test_compute_metrics(self):
        class MockModel:
            def __call__(self, batch):
                predictions = torch.tensor([
                    [0.9, 0.1, 0.8],  # Predictions for first user
                    [0.2, 0.7, 0.3]   # Predictions for second user
                ], dtype=torch.float32)
                return predictions, torch.zeros_like(predictions), torch.zeros_like(predictions)

            def eval(self):
                pass

            def to(self, device):
                return self

        model = MockModel()
        batch = torch.tensor([
            [1, 0, 1],  # User 0 interactions
            [0, 1, 0]   # User 1 interactions
        ], dtype=torch.float32)
        
        # Mock get_recommendations function
        def mock_get_recommendations(model, user_vector, user_id, top_k, device):
            predictions = torch.tensor([0.9, 0.1, 0.8], dtype=torch.float32)
            _, indices = torch.topk(predictions, top_k)
            return indices.tolist()  # Return just the indices as a list

        # Patch the get_recommendations function
        from unittest.mock import patch
        with patch('src.utils.metrics.get_recommendations', mock_get_recommendations):
            metrics = compute_metrics(
                model=model,
                batch=batch,
                top_k=2,
                total_items=3,
                device='cpu',
                user_map={0: 0, 1: 1}
            )

        assert isinstance(metrics, dict)
        assert all(0 <= metrics[k] <= 1 for k in metrics if metrics[k] is not None)

    def test_mean_squared_error(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        mse = mean_squared_error(y_true, y_pred)
        assert mse == 0.0

    def test_precision_recall_at_k(self):
        y_true = np.array([5, 3, 4, 1, 2])
        y_pred = np.array([0.9, 0.2, 0.8, 0.1, 0.3])
        k = 2

        prec = precision_at_k(y_true, y_pred, k)
        rec = recall_at_k(y_true, y_pred, k)

        assert 0 <= prec <= 1
        assert 0 <= rec <= 1

    def ndcg_at_k(self):
        y_true = np.array([5, 3, 4, 1, 2])
        y_pred = np.array([0.9, 0.2, 0.8, 0.1, 0.3])
        k = 3

        score = ndcg_at_k(y_true, y_pred, k)
        assert 0 <= score <= 1

    def test_coverage(self):
        predictions = np.array([0.9, 0.2, 0.8, 0.1, 0.3])
        num_items = 5
        
        cov = coverage(predictions, num_items)
        assert 0 <= cov <= 1

    def test_compute_roc_auc(self):
        class MockModel(MatrixFactorization):
            def __init__(self):
                pass
                
            def __call__(self, users, items):
                # Return predictions matching input shape (2x3)
                return torch.tensor([[0.9, 0.2, 0.8], [0.1, 0.7, 0.3]])
            
            def eval(self):
                pass
            
            def to(self, device):
                return self

        model = MockModel()
        batch = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        roc_auc = compute_roc_auc(model, batch, 'cpu')
        assert 0 <= roc_auc <= 1
