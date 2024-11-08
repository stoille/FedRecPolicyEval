import pytest
import torch
import numpy as np
from src.utils.metrics import (
    compute_rmse,
    calculate_recommendation_metrics,
    MetricsLogger
)

@pytest.fixture
def metrics_logger():
    return MetricsLogger()

class TestMetrics:
    def test_compute_rmse(self):
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        rmse = compute_rmse(predictions, targets)
        assert rmse == 0.0

    def test_recommendation_metrics(self):
        top_k_items = np.array([1, 2, 3, 4, 5])
        relevant_items = np.array([1, 3, 5])
        metrics = calculate_recommendation_metrics(
            top_k_items=top_k_items,
            relevant_items=relevant_items,
            top_k=5,
            total_items=10
        )
        
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['ndcg'] <= 1
        assert metrics['roc_auc'] is not None

    def test_metrics_logger(self, metrics_logger):
        test_metrics = {
            'loss': 0.5,
            'precision_at_k': 0.7,
            'recall_at_k': 0.6,
            'ndcg_at_k': 0.8
        }
        
        metrics_logger.log_metrics(test_metrics, is_training=True)
        
        assert len(metrics_logger.history['train_loss']) == 1
        assert metrics_logger.history['train_loss'][0] == 0.5
        assert metrics_logger.history['precision_at_k'][0] == 0.7
