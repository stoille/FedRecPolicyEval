import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import json
import logging
import torch

logger = logging.getLogger("Visualization")

def plot_metrics_history(history: dict, save_path: str = "metrics_plots.png"):
    """Plot all metrics from history."""
    if not history['rounds']:
        logger.warning("No metrics to plot")
        return
        
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss plot
    axs[0, 0].plot(history['rounds'], history['train_loss'], label='Training Loss')
    axs[0, 0].plot(history['rounds'], history['val_loss'], label='Validation Loss')
    axs[0, 0].set_xlabel('Rounds')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Training and Validation Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # RMSE plot
    axs[0, 1].plot(history['rounds'], history['rmse'], label='RMSE', color='orange')
    axs[0, 1].set_xlabel('Rounds')
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].set_title('Root Mean Squared Error')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Precision/Recall plot
    axs[0, 2].plot(history['rounds'], history['precision_at_k'], label='Precision@K')
    axs[0, 2].plot(history['rounds'], history['recall_at_k'], label='Recall@K')
    axs[0, 2].set_xlabel('Rounds')
    axs[0, 2].set_ylabel('Score')
    axs[0, 2].set_title('Precision@K and Recall@K')
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # NDCG plot
    axs[1, 0].plot(history['rounds'], history['ndcg_at_k'], label='NDCG@K', color='green')
    axs[1, 0].set_xlabel('Rounds')
    axs[1, 0].set_ylabel('NDCG')
    axs[1, 0].set_title('Normalized Discounted Cumulative Gain')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # ROC AUC plot
    axs[1, 1].plot(history['rounds'], history['roc_auc'], label='ROC AUC', color='blue')
    axs[1, 1].set_xlabel('Rounds')
    axs[1, 1].set_ylabel('ROC AUC')
    axs[1, 1].set_title('ROC AUC')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Coverage plot
    axs[1, 2].plot(history['rounds'], history['coverage'], label='Coverage', color='brown')
    axs[1, 2].set_xlabel('Rounds')
    axs[1, 2].set_ylabel('Coverage')
    axs[1, 2].set_title('Coverage')
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # Save history to JSON
    with open('history.json', 'w') as f:
        json.dump(history, f)
    logger.info(f"Saved metrics plot to {save_path} and history to history.json")

def visualize_latent_space(model, dataloader, device):
    if not hasattr(model, 'encode'):
        return
    
    model.eval()
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                user_data = batch.to(device)
                mu, _ = model.encode(user_data)
                latent_vectors.append(mu.cpu().numpy())
                labels.extend(user_data.sum(dim=1).cpu().numpy())

    if not latent_vectors:
        return

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    plot_tsne_visualization(latent_vectors, labels)

def plot_tsne_visualization(latent_vectors, labels):
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter, label='Number of rated items')
    plt.title('t-SNE visualization of latent space')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('latent_space_visualization.png')
    plt.close()