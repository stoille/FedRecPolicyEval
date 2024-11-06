import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from src.utils.model_utils import get_recommendations
import logging
import sys

# Configure logging at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Metrics")

def loss_function(recon_x, x, mu, logvar, epoch=1, num_epochs=1):
    reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    beta = min(epoch / (num_epochs/4), 1.0) * 0.1
    return reconstruction_loss + beta * kl_loss

def compute_rmse(recon_batch, batch):
    mask = batch > 0
    mse = F.mse_loss(recon_batch[mask], batch[mask], reduction='mean')
    rmse = torch.sqrt(mse)
    return rmse.item()

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

def train_mf(model, train_loader, optimizer, device, epochs):
    """Train matrix factorization model."""
    model.train()
    total_loss = 0
    num_samples = 0
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (user_ids, item_ids, ratings) in enumerate(train_loader):
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            total_loss += loss.item() * len(ratings)
            num_samples += len(ratings)
            
            if batch_idx % 10 == 0:  # Log every 10 batches
                logger.info(f"MF Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"MF Epoch {epoch}/{epochs} Complete | Avg Loss: {avg_epoch_loss:.4f}")

    avg_loss = total_loss / num_samples
    logger.info(f"MF Training Complete | Final Avg Loss: {avg_loss:.4f} | Samples: {num_samples}")
    return {
        "loss": avg_loss,
        "num_samples": num_samples
    }

def train(model, train_loader, optimizer, device, epochs):
    """Train VAE model."""
    model.train()
    total_loss = 0
    num_samples = 0

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = model.loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_loss += loss.item()
            num_samples += batch.size(0)
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch}/{epochs} Complete | Avg Loss: {avg_epoch_loss:.4f}")

    avg_loss = total_loss / num_samples
    logger.info(f"Training Complete | Final Avg Loss: {avg_loss:.4f} | Samples: {num_samples}")
    return {"loss": avg_loss, "num_samples": num_samples}

def test(model, test_loader, device, top_k):
    """Test VAE model."""
    model.eval()
    total_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, (user_ids, item_ids, ratings) in enumerate(test_loader):
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            
            if isinstance(model, VAE):
                recon_batch, mu, logvar = model(ratings)
                loss = model.loss_function(recon_batch, ratings, mu, logvar)
            else:
                predictions = model(user_ids, item_ids)
                loss = nn.MSELoss()(predictions, ratings)
            
            total_loss += loss.item() * len(ratings)
            num_samples += len(ratings)
            
            if batch_idx % 10 == 0:  # Log every 10 batches
                logger.info(f"Test Batch {batch_idx}/{len(test_loader)} | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_samples
    logger.info(f"Testing Complete | Avg Loss: {avg_loss:.4f} | Samples: {num_samples}")
    return {
        "loss": avg_loss,
        "num_samples": num_samples
    }