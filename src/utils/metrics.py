import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from src.utils.model_utils import get_recommendations
import logging
import sys
from typing import Dict
import atexit
from src.utils.visualization import plot_metrics_history

# Configure logging at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Metrics")

class MetricsLogger:
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'rmse': [],
            'precision_at_k': [],
            'recall_at_k': [],
            'ndcg_at_k': [],
            'coverage': [],
            'roc_auc': [],
            'rounds': []
        }
        self.current_round = 0
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics for current round."""
        if not metrics:
            return
            
        self.history['rounds'].append(self.current_round)
        
        metric_mapping = {
            'val_loss': 'val_loss',
            'loss': 'train_loss',
            'rmse': 'rmse',
            'precision_at_k': 'precision_at_k',
            'recall_at_k': 'recall_at_k',
            'ndcg_at_k': 'ndcg_at_k',
            'coverage': 'coverage',
            'roc_auc': 'roc_auc'
        }
        
        for metric_key, history_key in metric_mapping.items():
            if metric_key in metrics:
                self.history[history_key].append(metrics[metric_key])
            else:
                prev_value = self.history[history_key][-1] if self.history[history_key] else 0
                self.history[history_key].append(prev_value)
        
        self.current_round += 1
        logger.info(f"Logged metrics for round {self.current_round}: {metrics}")

# Create global metrics logger
metrics_logger = MetricsLogger()

# Register cleanup function
def cleanup():
    logger.info("Saving metrics plots and history...")
    plot_metrics_history(metrics_logger.history)

atexit.register(cleanup)

def loss_function(recon_x, x, mu, logvar, epoch=1, num_epochs=1):
    reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    beta = min(epoch / (num_epochs/4), 1.0) * 0.1
    return reconstruction_loss + beta * kl_loss

def compute_rmse(predictions, targets):
    """Compute Root Mean Square Error."""
    mse = F.mse_loss(predictions, targets)
    return torch.sqrt(mse).item()

def compute_metrics(model, batch, top_k, total_items, device):
    precisions, recalls, ndcgs, roc_aucs = [], [], [], []
    all_recommended_items = []

    for user_vector in batch:
        top_k_items = get_recommendations(model, user_vector, top_k, device)
        interacted_items = user_vector.nonzero(as_tuple=True)[0]
        relevant_items = interacted_items.cpu().numpy()
        
        metrics = calculate_recommendation_metrics(
            top_k_items, relevant_items, top_k, total_items
        )
        
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        ndcgs.append(metrics['ndcg'])
        if metrics.get('roc_auc'):
            roc_aucs.append(metrics['roc_auc'])
        all_recommended_items.append(top_k_items)

    coverage = len(set(np.concatenate(all_recommended_items))) / total_items
    
    return {
        "precision_at_k": np.mean(precisions),
        "recall_at_k": np.mean(recalls),
        "ndcg_at_k": np.mean(ndcgs),
        "roc_auc": np.mean(roc_aucs) if roc_aucs else None,
        "coverage": coverage,
    }

def calculate_recommendation_metrics(top_k_items, relevant_items, top_k, total_items):
    hits_arr = np.isin(top_k_items, relevant_items)
    hits = hits_arr.sum()
    
    precision = hits / top_k
    recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
    
    dcg = sum([1 / np.log2(i + 2) if top_k_items[i] in relevant_items else 0 
               for i in range(top_k)])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_items), top_k))])
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    actual = np.zeros(total_items)
    actual[relevant_items] = 1
    scores = np.zeros(total_items)
    scores[top_k_items] = 1
    
    roc_auc = None
    if len(np.unique(actual)) > 1:
        roc_auc = roc_auc_score(actual, scores)
    
    return {
        'precision': precision,
        'recall': recall,
        'ndcg': ndcg,
        'roc_auc': roc_auc
    }

def train(model, train_loader, optimizer, device, epochs, model_type: str):
    """Train model."""
    model.train()
    epoch_metrics = {
        'train_loss': 0,
        'val_loss': 0,
        'rmse': 0,
        'precision_at_k': 0,
        'recall_at_k': 0,
        'ndcg_at_k': 0,
        'coverage': 0,
        'roc_auc': 0,
    }
    num_samples = 0

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            if model_type == 'vae':
                ratings = batch.to(device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(ratings)
                loss = model.loss_function(recon_batch, ratings, mu, logvar)
                # Calculate RMSE for VAE
                epoch_metrics['rmse'] += compute_rmse(recon_batch, ratings) * len(ratings)
            else:  # model_type == 'mf'
                user_ids, item_ids, ratings = batch
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)
                optimizer.zero_grad()
                predictions = model(user_ids, item_ids)
                loss = nn.MSELoss()(predictions, ratings)
                # Calculate RMSE for MF
                epoch_metrics['rmse'] += compute_rmse(predictions, ratings) * len(ratings)
            
            loss.backward()
            optimizer.step()
            
            epoch_metrics['train_loss'] += loss.item() * len(ratings)
            num_samples += len(ratings)
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # Average metrics
    for k in epoch_metrics:
        if num_samples > 0:
            epoch_metrics[k] /= num_samples

    logger.info(f"Training Complete | Metrics: {epoch_metrics}")
    return epoch_metrics

def test(model, test_loader, device, top_k, model_type: str, num_items: int):
    """Test model."""
    model.eval()
    metrics = {
        'val_loss': 0,
        'loss': 0,
        'rmse': 0,
        'precision_at_k': 0,
        'recall_at_k': 0,
        'ndcg_at_k': 0,
        'coverage': 0,
        'roc_auc': 0,
    }
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if model_type == 'vae':
                ratings = batch.to(device)
                recon_batch, mu, logvar = model(ratings)
                loss = model.loss_function(recon_batch, ratings, mu, logvar)
                batch_metrics = compute_metrics(model, ratings, top_k, num_items, device)
                # Calculate RMSE for VAE
                metrics['rmse'] += compute_rmse(recon_batch, ratings) * len(ratings)
            else:  # model_type == 'mf'
                user_ids, item_ids, ratings = batch
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)
                predictions = model(user_ids, item_ids)
                loss = nn.MSELoss()(predictions, ratings)
                batch_metrics = compute_metrics(model, ratings, top_k, num_items, device)
                # Calculate RMSE for MF
                metrics['rmse'] += compute_rmse(predictions, ratings) * len(ratings)
            
            val_loss = loss.item() * len(ratings)
            metrics['val_loss'] += val_loss
            metrics['loss'] += val_loss
            
            for k, v in batch_metrics.items():
                if v is not None and k != 'rmse':  # Skip RMSE from batch_metrics
                    metrics[k] += v * len(ratings)
            
            num_samples += len(ratings)
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx}/{len(test_loader)} batches")
    
    # Average metrics
    for k in metrics:
        if num_samples > 0:
            metrics[k] /= num_samples
    
    logger.info(f"Testing Complete | Metrics: {metrics}")
    return metrics