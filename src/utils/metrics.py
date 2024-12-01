import numpy as np
import torch
import torch.nn.functional as F
from src.utils.model_utils import get_recommendations
import logging
import sys
from typing import Dict, List
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
        self.eval_history = {
            'round_eval_loss': [],
            'round_eval_rmse': [],
            'precision_at_k': [],
            'recall_at_k': [],
            'ndcg_at_k': [],
            'coverage': [],
            'roc_auc': [],
            'rounds': []
        }
        self.current_round = 0
        
    def log_metrics(self, metrics, is_training=True):
        history = self.train_history if is_training else self.eval_history
        
        if is_training:
            # For training metrics, append to existing arrays
            for key in metrics:
                if key in history and metrics[key] is not None:
                    history[key].append(metrics[key])
        else:
            # For eval metrics, increment round first
            self.current_round += 1
            # Then append all metrics
            history['rounds'].append(self.current_round)
            for key in metrics:
                if key in history and metrics[key] is not None:
                    history[key].append(metrics[key])

# Create global metrics logger
metrics_logger = MetricsLogger()

def compute_metrics(model, batch, top_k, total_items, device, user_map, temperature, negative_penalty):
    metrics = {}
    
    # Get user preference vectors
    if isinstance(model, MatrixFactorization):
        ut_vector = model.user_factors.weight.data.clone().cpu().numpy()
    else:  # VAE
        ut_vector = model.encoder[0].weight.data.clone().cpu().numpy()
    
    precisions, recalls, ndcgs, roc_aucs = [], [], [], []
    all_recommended_items = []

    for batch_idx, user_vector in enumerate(batch):
        user_id = list(user_map.keys())[list(user_map.values()).index(batch_idx)]
        top_k_items, all_scores = get_recommendations(
            model, 
            user_vector, 
            user_id, 
            top_k, 
            device,
            temperature=temperature,
            negative_penalty=negative_penalty
        )
        interacted_items = user_vector.nonzero(as_tuple=True)[0]
        relevant_items = interacted_items.cpu().numpy()
        #logger.info(f"Top-k items: {top_k_items}")
        #logger.info(f"Relevant items: {relevant_items}")
        
        metrics = calculate_recommendation_metrics(
            top_k_items, relevant_items, top_k, total_items
        )
        
        precisions.append(metrics['precision_at_k'])
        recalls.append(metrics['recall_at_k'])
        ndcgs.append(metrics['ndcg_at_k'])
        if metrics.get('roc_auc'):
            roc_aucs.append(metrics['roc_auc'])
        all_recommended_items.extend(top_k_items)

    # Calculate coverage as unique items / total items
    coverage = len(set(all_recommended_items)) / total_items
    
    return {
        'precision_at_k': np.mean(precisions),
        'recall_at_k': np.mean(recalls),
        'ndcg_at_k': np.mean(ndcgs),
        'roc_auc': np.mean(roc_aucs) if roc_aucs else None,
        'coverage': coverage,
        'eval_ut_vector': ut_vector,
        'eval_ut_norm': np.linalg.norm(ut_vector)
    }

def calculate_recommendation_metrics(top_k_items, relevant_items, top_k, total_items):
    hits_arr = np.isin(top_k_items, relevant_items)
    
    hits = hits_arr.sum()
    #logger.info(f"Hits = hits_arr.sum() = {hits}")
    
    overlaps = set(top_k_items) & set(relevant_items)
    #logger.info(f"Overlapping items: {overlaps}")
    
    precision = hits / top_k
    recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
    #logger.info(f"Precision = hits / top_k = {hits} / {top_k} = {precision}")
    #logger.info(f"Recall = hits / len(relevant_items) = {hits} / {len(relevant_items)} = {recall}")
    
    # Compute NDCG
    dcg = sum([1 / np.log2(i + 2) if top_k_items[i] in relevant_items else 0 
               for i in range(top_k)])
    #logger.info(f"DCG = sum([1 / np.log2(i + 2) if top_k_items[i] in relevant_items else 0 for i in range(top_k)]) = {dcg}")
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_items), top_k))])
    #logger.info(f"IDCG = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_items), top_k))]) = {idcg}")
    ndcg = dcg / idcg if idcg > 0 else 0.0
    #logger.info(f"NDCG = dcg / idcg = {dcg} / {idcg} = {ndcg}")
    #logger.info(f"NDCG = {ndcg}")
    # Compute ROC AUC
    actual = np.zeros(total_items)
    actual[relevant_items] = 1
    scores = np.zeros(total_items)
    scores[top_k_items] = 1
    
    roc_auc = None
    if len(np.unique(actual)) > 1:
        roc_auc = roc_auc_score(actual, scores)
    #logger.info(f"ROC AUC: {roc_auc}")
    return {
        'precision_at_k': precision,
        'recall_at_k': recall,
        'ndcg_at_k': ndcg,
        'roc_auc': roc_auc
    }

def train(model, train_loader, optimizer, device, epochs, model_type: str, preference_evolution=None):
    model.train()
    total_loss = 0.0
    total_rmse = 0.0
    num_batches = 0
    
    # Add annealing for VAE
    beta_start = 0.0
    beta_end = 0.1
    epoch_losses = []  # Track losses as array
    epoch_rmses = []   # Track RMSEs as array
    epoch_preferences = {
        'ut_norm': [],
        'likable_prob': [],
        'nonlikable_prob': [],
        'correlated_mass': []
    }
    
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
        # Append to arrays
        epoch_losses.append(avg_epoch_loss)
        epoch_rmses.append(avg_epoch_rmse)
        
        total_loss += avg_epoch_loss
        total_rmse += avg_epoch_rmse
        num_batches += 1
     
        # Log metrics for this epoch
        metrics_logger.log_metrics({"train_loss": avg_epoch_loss, "train_rmse": avg_epoch_rmse}, is_training=True)   
        #logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_epoch_loss:.4f} | RMSE: {avg_epoch_rmse:.4f}")
        
        # Update preferences simulation
        if model_type == 'mf':
            items = model.item_factors.weight[item_ids]
            scores = predictions.view(-1, 1)
        else:
            items = batch.to(device)
            recon_x, mu, logvar = model(items)
            # Compute per-user reconstruction loss
            per_user_recon_loss = F.binary_cross_entropy(recon_x, items, reduction='none').sum(dim=1)
            # Invert loss to get scores (higher is better)
            scores = -per_user_recon_loss  # Invert loss for scores
        
        preference_evolution.update_preferences(items, scores, is_round=False)
        divergence_metrics = preference_evolution.get_current_metrics(items, scores)
        
        for key in epoch_preferences:
            epoch_preferences[key].append(np.mean(divergence_metrics[key]))
    
    # Return metrics dictionary with arrays
    return {
        'train_loss': epoch_losses,
        'train_rmse': epoch_rmses,
        'ut_norm': epoch_preferences['ut_norm'],
        'likable_prob': epoch_preferences['likable_prob'],
        'nonlikable_prob': epoch_preferences['nonlikable_prob'],
        'correlated_mass': epoch_preferences['correlated_mass']
    }

def evaluate_fn(model, eval_loader, device, top_k, model_type, num_items, user_map, temperature, negative_penalty, popularity_penalty):
    """Test the model and return metrics."""
    model.eval()
    round_eval_loss = 0.0
    round_eval_rmse = 0.0
    all_precisions = []
    all_recalls = []
    all_ndcgs = []
    all_coverages = []
    all_roc_aucs = []
    with torch.no_grad():
        for batch in eval_loader:
            # Move batch to device
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device)
            elif isinstance(batch, (list, tuple)):
                batch = [b.to(device) for b in batch]
            
            # Get recommendations and compute metrics
            metrics = compute_metrics(
                model=model, 
                batch=batch,
                top_k=top_k, 
                total_items=num_items, 
                device=device, 
                user_map=user_map, 
                temperature=temperature, 
                negative_penalty=negative_penalty
            )
            
            #logger.info(f"foo Metrics: {metrics}")
            
            # Extract metrics using correct keys
            all_precisions.append(metrics['precision_at_k'])
            all_recalls.append(metrics['recall_at_k'])
            all_ndcgs.append(metrics['ndcg_at_k'])
            all_coverages.append(metrics['coverage'])
            all_roc_aucs.append(metrics['roc_auc'])
            
            # Compute loss and RMSE
            if model_type == "vae":
                recon_batch, mu, logvar = model(batch)
                loss = loss_function(recon_batch, batch, mu, logvar)
                rmse = torch.sqrt(F.mse_loss(recon_batch, batch))
            else:
                output = model(batch)
                loss = F.mse_loss(output, batch)
                rmse = torch.sqrt(loss)
            
            round_eval_loss += loss.item()
            round_eval_rmse += rmse.item()
    
    # Average metrics
    avg_round_eval_loss = round_eval_loss / len(eval_loader)
    avg_round_eval_rmse = round_eval_rmse / len(eval_loader)
    avg_precision = np.mean(all_precisions) if all_precisions else 0.0
    avg_recall = np.mean(all_recalls) if all_recalls else 0.0
    avg_ndcg = np.mean(all_ndcgs) if all_ndcgs else 0.0
    coverage = np.mean(all_coverages) if all_coverages else 0.0
    avg_roc_auc = np.mean(all_roc_aucs) if all_roc_aucs else 0.0
    #logger.info(f"Avg round eval loss: {avg_round_eval_loss}")
    #logger.info(f"Avg round test rmse: {avg_round_eval_rmse}")
    #logger.info(f"Avg precision: {avg_precision}")
    #logger.info(f"Avg recall: {avg_recall}")
    #logger.info(f"Avg ndcg: {avg_ndcg}")
    #logger.info(f"Coverage: {coverage}")
    return {
        'eval_loss': avg_round_eval_loss,
        'eval_rmse': avg_round_eval_rmse,
        'precision_at_k': avg_precision,
        'recall_at_k': avg_recall,
        'ndcg_at_k': avg_ndcg,
        'coverage': coverage,
        'roc_auc': avg_roc_auc
    }

def calculate_global_metrics(top_k_items, top_k_scores, ground_truth, top_k, num_items, popularity_penalty, model_type="MF"):
    n_users = len(top_k_items)
    precisions = []
    recalls = []
    ndcgs = []
    all_recommended = set()
    
    # For ROC AUC calculation
    all_true = np.zeros((n_users, num_items))
    all_pred = np.zeros((n_users, num_items))
    
    # Modify recommendation diversity by penalizing popular items
    item_counts = {}
    for items in top_k_items:
        for item in items:
            item_counts[item] = item_counts.get(item, 0) + 1
            
    # Add popularity penalty to scores
    for i, scores in enumerate(top_k_scores):
        # Only penalize the top-k items
        popularity_values = np.array([item_counts.get(item, 0) / len(top_k_items) for item in top_k_items[i][:top_k]])
        scores[:top_k] = scores[:top_k] - popularity_penalty * popularity_values
        top_k_scores[i] = scores
    
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
    
    # Calculate ROC AUC using raw scores instead of binary predictions
    try:
        # Flatten and combine all predictions and ground truth
        all_true_flat = all_true.flatten()
        all_pred_flat = all_pred.flatten()
        
        # Only consider items that have either a prediction or are in ground truth
        mask = (all_true_flat > 0) | (all_pred_flat > 0)
        if mask.sum() > 0:
            all_true_flat = all_true_flat[mask]
            all_pred_flat = all_pred_flat[mask]
            
            if len(np.unique(all_true_flat)) >= 2:
                roc_auc = roc_auc_score(all_true_flat, all_pred_flat)
            else:
                roc_auc = 0.5
        else:
            roc_auc = 0.5
    except ValueError:
        logger.warning("ROC AUC calculation failed, defaulting to 0.5")
        roc_auc = 0.5
    
    # Track unique recommendations differently for each model type
    if model_type == "MF":
        # For MF, actively promote diversity
        all_recommended = set()
        item_counts = {}
        for items in top_k_items:
            all_recommended.update(items)
            for item in items:
                item_counts[item] = item_counts.get(item, 0) + 1
                
        # Adjust scores based on popularity for MF only
        for i, scores in enumerate(top_k_scores):
            popularity_values = np.array([item_counts.get(item, 0) / len(top_k_items) for item in top_k_items[i]])
            top_k_scores[i] = scores - popularity_penalty * popularity_values
    else:  # VAE case
        # VAE has natural diversity, just track unique items
        all_recommended = set(np.concatenate(top_k_items))
    
    coverage = len(all_recommended) / num_items

    return {
        'precision_at_k': np.mean(precisions),
        'recall_at_k': np.mean(recalls),
        'ndcg_at_k': np.mean(ndcgs),
        'coverage': coverage,
        'roc_auc': roc_auc
    }

def loss_function(recon_x, x, mu, logvar, epoch=1, num_epochs=1):
    reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Add L2 regularization
    l2_reg = 0.01 * (mu.pow(2).mean() + logvar.exp().mean())
    
    # Gradual KL annealing
    beta = min(epoch / (num_epochs/2), 1.0) * 0.1
    
    return reconstruction_loss + beta * kl_loss + l2_reg

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

def update_histories(histories: Dict, metrics: Dict, phase: str = 'train') -> None:
    """Update consolidated histories with new metrics."""
    history_key = f'{phase}_history'
    
    if phase == 'train':
        if 'train_loss' in metrics:
            histories[history_key]['train_loss'].append(metrics['train_loss'])
        if 'train_rmse' in metrics:
            histories[history_key]['train_rmse'].append(metrics['train_rmse'])
        if 'rounds' in metrics:
            histories[history_key]['rounds'].append(metrics['rounds'])
    else:  # test
        for key in ['round_eval_loss', 'round_eval_rmse', 'precision_at_k', 'recall_at_k', 'ndcg_at_k', 'coverage']:
            if key in metrics:
                histories[history_key][key].append(metrics[key])

def calculate_model_divergence(local_vectors: List[np.ndarray], global_vector: np.ndarray) -> Dict:
    """Calculate divergence metrics between local and global models."""
    divergence_metrics = {}
    
    # Ensure vectors are properly shaped for cosine distance
    def reshape_vector(v):
        return v.flatten() if len(v.shape) > 1 else v
    
    # Reshape global vector
    global_vector = reshape_vector(global_vector)
    
    # Calculate cosine distances between each local model and global model
    cosine_distances = []
    for local_vec in local_vectors:
        local_vec = reshape_vector(local_vec)
        norm_product = np.linalg.norm(local_vec) * np.linalg.norm(global_vector)
        if norm_product > 0:
            cos_dist = 1 - np.dot(local_vec, global_vector) / norm_product
            cosine_distances.append(cos_dist)
    
    # Calculate pairwise distances (personalization) between local models if we have at least 2 models
    pairwise_distances = []
    if len(local_vectors) >= 2:
        for i in range(len(local_vectors)):
            vec_i = reshape_vector(local_vectors[i])
            for j in range(i + 1, len(local_vectors)):
                vec_j = reshape_vector(local_vectors[j])
                norm_product = np.linalg.norm(vec_i) * np.linalg.norm(vec_j)
                if norm_product > 0:
                    cos_dist = 1 - np.dot(vec_i, vec_j) / norm_product
                    pairwise_distances.append(cos_dist)
    
    # Calculate metrics with safeguards
    local_global_div = np.mean(cosine_distances) if cosine_distances else 0.0
    personalization_degree = np.mean(pairwise_distances) if pairwise_distances else 0.0
    max_local_div = max(pairwise_distances) if pairwise_distances else 0.0
    
    divergence_metrics.update({
        'local_global_divergence': float(local_global_div),
        'personalization_degree': float(personalization_degree),
        'max_local_divergence': float(max_local_div)
    })
    
    return divergence_metrics