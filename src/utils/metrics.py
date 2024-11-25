import numpy as np
import torch
import torch.nn.functional as F
from src.utils.model_utils import get_recommendations
import logging
import sys
from typing import Dict
import atexit
from sklearn.metrics import roc_auc_score
import json
from src.models.matrix_factorization import MatrixFactorization

# Configure logging at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Metrics")

class MetricsLogger:
    def __init__(self):
        self.train_history = {
            'train_loss': [],
            'train_rmse': [],
            'rounds': []
        }
        self.test_history = {
            'test_loss': [],
            'test_rmse': [],
            'precision_at_k': [],
            'recall_at_k': [],
            'ndcg_at_k': [],
            'coverage': [],
            'roc_auc': [],
            'rounds': []
        }
        self.current_round = 0  # Start from 0
        
    def log_metrics(self, metrics, is_training=True):
        history = self.train_history if is_training else self.test_history
        
        if is_training:
            # For training metrics, append to existing arrays
            for key in metrics:
                if key in history and metrics[key] is not None:
                    history[key].append(metrics[key])
            # Don't increment rounds for training metrics
        else:
            # For test metrics, increment round first
            self.current_round += 1
            # Then append all metrics
            history['rounds'].append(self.current_round)
            for key in metrics:
                if key in history and metrics[key] is not None:
                    history[key].append(metrics[key])
        
        self._save_histories()

    def _save_histories(self):
        try:
            # Save histories
            with open('train_history.json', 'w') as f:
                json.dump(self.train_history, f)
            with open('test_history.json', 'w') as f:
                json.dump(self.test_history, f)
            
        except Exception as e:
            logger.error(f"Error saving histories: {str(e)}")

# Create global metrics logger
metrics_logger = MetricsLogger()

def compute_metrics(model, batch, top_k, total_items, device, user_map):
    precisions, recalls, ndcgs, roc_aucs = [], [], [], []
    all_recommended_items = []

    for batch_idx, user_vector in enumerate(batch):
        # Get original user_id from the mapping
        user_id = list(user_map.keys())[list(user_map.values()).index(batch_idx)]
        top_k_items = get_recommendations(model, user_vector, user_id, top_k, device)
        interacted_items = user_vector.nonzero(as_tuple=True)[0]
        relevant_items = interacted_items.cpu().numpy()
        logger.info(f"Top-k items: {top_k_items}")
        logger.info(f"Relevant items: {relevant_items}")
        
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
    logger.info(f"Hits array: {hits_arr}")
    
    # Debug logging to understand the values
    #logger.info(f"Top-k items selected: {top_k_items[:5]}")  # Will show indices
    #logger.info(f"Relevant items: {relevant_items[:5]}")  # Ground truth indices
    
    hits = hits_arr.sum()
    logger.info(f"Hits = hits_arr.sum() = {hits}")
    
    overlaps = set(top_k_items) & set(relevant_items)
    logger.info(f"Overlapping items: {overlaps}")
    
    precision = hits / top_k
    recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
    logger.info(f"Precision = hits / top_k = {hits} / {top_k} = {precision}")
    logger.info(f"Recall = hits / len(relevant_items) = {hits} / {len(relevant_items)} = {recall}")
    
    # Compute NDCG
    dcg = sum([1 / np.log2(i + 2) if top_k_items[i] in relevant_items else 0 
               for i in range(top_k)])
    logger.info(f"DCG = sum([1 / np.log2(i + 2) if top_k_items[i] in relevant_items else 0 for i in range(top_k)]) = {dcg}")
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_items), top_k))])
    logger.info(f"IDCG = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_items), top_k))]) = {idcg}")
    ndcg = dcg / idcg if idcg > 0 else 0.0
    logger.info(f"NDCG = dcg / idcg = {dcg} / {idcg} = {ndcg}")
    # Compute ROC AUC
    actual = np.zeros(total_items)
    actual[relevant_items] = 1
    scores = np.zeros(total_items)
    scores[top_k_items] = 1
    
    roc_auc = None
    if len(np.unique(actual)) > 1:
        roc_auc = roc_auc_score(actual, scores)
    logger.info(f"ROC AUC: {roc_auc}")
    return {
        'precision': precision,
        'recall': recall,
        'ndcg': ndcg,
        'roc_auc': roc_auc
    }

def train(model, train_loader, optimizer, device, epochs, model_type: str):
    model.train()
    total_loss = 0.0
    total_rmse = 0.0
    num_batches = 0
    
    # Add annealing for VAE
    beta_start = 0.0
    beta_end = 0.1
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_rmse = 0.0
        epoch_batches = 0
        
        # Gradually increase beta for KL term
        beta = min(epoch / (epochs/2), 1.0) * (beta_end - beta_start) + beta_start
        
        for batch in train_loader:
            if model_type == 'mf':
                user_ids, item_ids, ratings = [b.to(device) for b in batch]
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(user_ids, item_ids)
                
                # Compute loss only on non-zero ratings
                mask = ratings > 0
                loss = F.mse_loss(predictions[mask], ratings[mask])
                
                # Add L2 regularization
                l2_reg = 0.01 * (
                    model.user_factors.weight.norm(2) + 
                    model.item_factors.weight.norm(2) +
                    model.user_biases.weight.norm(2) + 
                    model.item_biases.weight.norm(2)
                )
                loss += l2_reg
                
                # Compute RMSE
                rmse = compute_rmse(predictions[mask], ratings[mask])
                
            else:  # VAE case
                ratings = batch.to(device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(ratings)
                loss = model.loss_function(recon_batch, ratings, mu, logvar, beta=beta)
                rmse = compute_rmse(recon_batch, ratings)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_rmse += rmse
            epoch_batches += 1
            
        # Average metrics for this epoch
        avg_epoch_loss = epoch_loss / epoch_batches
        avg_epoch_rmse = epoch_rmse / epoch_batches
        
        total_loss += avg_epoch_loss
        total_rmse += avg_epoch_rmse
        num_batches += 1
     
        # Log metrics for this epoch
        metrics_logger.log_metrics({"train_loss": avg_epoch_loss, "train_rmse": avg_epoch_rmse}, is_training=True)   
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_epoch_loss:.4f} | RMSE: {avg_epoch_rmse:.4f}")
    
    # Average metrics across epochs
    avg_metrics = {
        "train_loss": total_loss / num_batches,
        "train_rmse": total_rmse / num_batches
    }
    
    return avg_metrics

def test(model, test_loader, device, top_k, model_type, num_items, user_map):
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    num_batches = 0
    all_top_k_items = []
    all_top_k_scores = []
    all_ground_truth = []
    
    def process_recommendations(user_id, user_vector, ground_truth, model_type):
        try:
            top_k_items, all_scores = get_recommendations(model, user_vector, user_id, top_k, device)
            all_top_k_items.append(top_k_items)
            all_top_k_scores.append(all_scores)
            all_ground_truth.append(ground_truth)
        except Exception as e:
            logger.error(f"Error processing {model_type} user {user_id}: {str(e)}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if isinstance(model, MatrixFactorization):
                user_ids, item_ids, ratings = [b.to(device) for b in batch]
                predictions = model(user_ids, item_ids)
                loss = torch.nn.MSELoss()(predictions, ratings)
                
                for i in range(len(user_ids)):
                    user_id = user_ids[i].item()
                    ground_truth = item_ids[i][ratings[i] > 0].cpu().numpy()
                    process_recommendations(user_id, None, ground_truth, "MF")
            
            else:  # VAE case
                batch = batch.to(device)
                recon_batch, mu, logvar = model(batch)
                loss = model.loss_function(recon_batch, batch, mu, logvar)
                
                for i in range(batch.size(0)):
                    user_vector = batch[i]
                    ground_truth = torch.nonzero(user_vector).squeeze().cpu().numpy()
                    process_recommendations(None, user_vector, ground_truth, "VAE")
            
            rmse = torch.sqrt(loss)
            total_loss += loss.item()
            total_rmse += rmse.item()
            num_batches += 1
    
    # Calculate metrics
    metrics = calculate_global_metrics(
        np.array(all_top_k_items),
        np.array(all_top_k_scores),
        all_ground_truth,
        top_k,
        num_items
    )
    
    # Add average loss and RMSE
    metrics.update({
        'test_loss': total_loss / num_batches if num_batches > 0 else float('inf'),
        'test_rmse': total_rmse / num_batches if num_batches > 0 else float('inf')
    })
    
    metrics_logger.log_metrics(metrics, is_training=False)
    
    return metrics

def calculate_global_metrics(top_k_items, top_k_scores, ground_truth, top_k, num_items):
    n_users = len(top_k_items)
    precisions = []
    recalls = []
    ndcgs = []
    all_recommended = set()
    
    # For ROC AUC calculation
    all_true = np.zeros((n_users, num_items))
    all_pred = np.zeros((n_users, num_items))
    
    for i, (items, scores, truth) in enumerate(zip(top_k_items, top_k_scores, ground_truth)):
        # Ensure arrays are 1D
        items = np.atleast_1d(items)
        scores = np.atleast_1d(scores)
        truth = np.atleast_1d(truth)
        
        if truth.size == 0:
            logger.warning(f"Empty ground truth for user {i}, skipping")
            continue
            
        all_recommended.update(items)
        hits = len(set(items.tolist()) & set(truth.tolist()))
        
        precisions.append(hits / top_k)
        recalls.append(hits / len(truth) if len(truth) > 0 else 0)
        
        # NDCG calculation
        dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(items) if item in truth)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(truth), top_k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
        
        # Set up ROC AUC arrays
        all_true[i, truth] = 1
        for j, (item, score) in enumerate(zip(items[:top_k], scores[:top_k])):
            all_pred[i, item] = score
    
    if not precisions:
        logger.warning("No valid predictions found")
        return {
            'precision_at_k': 0.0,
            'recall_at_k': 0.0,
            'ndcg_at_k': 0.0,
            'coverage': 0.0,
            'roc_auc': 0.5
        }
    
    # Calculate ROC AUC
    try:
        roc_auc = roc_auc_score(all_true.flatten(), all_pred.flatten())
    except ValueError:
        logger.warning("ROC AUC calculation failed, defaulting to 0.5")
        roc_auc = 0.5
    
    return {
        'precision_at_k': np.mean(precisions),
        'recall_at_k': np.mean(recalls),
        'ndcg_at_k': np.mean(ndcgs),
        'coverage': len(all_recommended) / num_items,
        'roc_auc': roc_auc
    }

def loss_function(recon_x, x, mu, logvar, epoch=1, num_epochs=1):
    # Compute binary cross entropy loss for reconstruction
    reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
    # Compute KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # Compute beta value for KL divergence loss
    beta = min(epoch / (num_epochs/4), 1.0) * 0.1
    # Return total loss
    return reconstruction_loss + beta * kl_loss

def compute_rmse(predictions, targets):
    # Compute mean squared error
    mse = F.mse_loss(predictions, targets)
    # Return square root of mean squared error
    return torch.sqrt(mse).item()

def mean_squared_error(y_true, y_pred):
    # Return mean squared error
    return np.mean((y_true - y_pred) ** 2)

def precision_at_k(y_true, y_pred, k):
    # Convert inputs to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get top k predictions
    if y_pred.dtype == np.float64:
        top_k_items = np.argsort(y_pred)[-k:]
    else:
        # Handle case where y_pred already contains indices
        top_k_items = y_pred[:k]
    
    # Convert to sets for intersection
    relevant_items = set(y_true)
    recommended_items = set(top_k_items)
    
    hits = len(relevant_items.intersection(recommended_items))
    return hits / k if k > 0 else 0.0

def recall_at_k(y_true, y_pred, k):
    # Convert inputs to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get top k predictions
    if y_pred.dtype == np.float64:
        top_k_items = np.argsort(y_pred)[-k:]
    else:
        top_k_items = y_pred[:k]
    
    # Convert to sets for intersection
    relevant_items = set(y_true)
    recommended_items = set(top_k_items)
    
    hits = len(relevant_items.intersection(recommended_items))
    return hits / len(relevant_items) if len(relevant_items) > 0 else 0.0

def ndcg_at_k(y_true, y_pred, k):
    # Convert inputs to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_pred.dtype == np.float64:
        top_k_items = np.argsort(y_pred)[-k:]
    else:
        top_k_items = y_pred[:k]
    
    relevant_items = set(y_true)
    
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(top_k_items) if item in relevant_items)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
    
    return dcg / idcg if idcg > 0 else 0.0

def coverage(all_recommended_items, total_items, item_popularity=None):
    """Calculate coverage weighted by item popularity.
    
    Args:
        all_recommended_items: Set of items recommended across all users
        total_items: Total number of possible items
        item_popularity: Dict mapping item_id to its popularity score (interaction count)
    """
    if item_popularity is None:
        # Fallback to basic coverage if popularity data not available
        return len(all_recommended_items) / total_items
        
    # Calculate popularity-weighted coverage
    recommended_pop_sum = sum(item_popularity.get(item, 0) for item in all_recommended_items)
    total_pop_sum = sum(item_popularity.values())
    
    return recommended_pop_sum / total_pop_sum if total_pop_sum > 0 else 0

def compute_roc_auc(model, batch, device):
    """Compute ROC AUC score for model predictions."""
    try:
        if isinstance(model, MatrixFactorization):
            ratings = batch.to(device)
            num_users, num_items = ratings.shape
            
            # Create all possible user-item pairs
            user_ids = torch.arange(num_users, device=device).unsqueeze(1).expand(-1, num_items).reshape(-1)
            item_ids = torch.arange(num_items, device=device).unsqueeze(0).expand(num_users, -1).reshape(-1)
            
            # Get true labels and predictions
            y_true = ratings.flatten().cpu().numpy()
            y_pred = model(user_ids, item_ids).cpu().detach().numpy().flatten()
            
        else:  # VAE case
            ratings = batch.to(device)
            y_true = ratings.cpu().numpy()
            recon_batch, _, _ = model(ratings)
            y_pred = recon_batch.cpu().detach().numpy()
            
            # Keep original shape for VAE predictions
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
        
        # Filter out missing/unobserved interactions first
        mask = y_true != 0
        if mask.sum() > 0:
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            # Convert to binary after filtering
            y_true_binary = (y_true > 3).astype(int)
            
            # Add logging to debug class distribution
            unique_classes = np.unique(y_true_binary)
            logger.info(f"Unique classes in batch: {unique_classes}, Positive samples: {sum(y_true_binary)}, Total: {len(y_true_binary)}")
            
            if len(unique_classes) >= 2:
                return roc_auc_score(y_true_binary, y_pred)
            else:
                logger.warning(f"Not enough classes. Found classes: {unique_classes}")
        
        return 0.5  # Default value (random classifier performance)
        
    except Exception as e:
        logger.warning(f"ROC AUC calculation failed: {str(e)}")
        return 0.5