import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import json
import logging
import torch

logger = logging.getLogger("Visualization")

def plot_metrics_history(train_history: dict, test_history: dict, save_path: str = "metrics_plots.png"):
    """Plot all metrics from both train and test histories."""
    if not train_history.get('rounds') and not test_history.get('rounds'):
        logger.warning("No metrics to plot")
        return

    fig, axs = plt.subplots(4, 2, figsize=(15, 24))
    
    # Generate proper sequential indices
    train_indices = range(1, len(train_history.get('train_loss', [])) + 1)
    test_indices = range(1, len(test_history.get('test_loss', [])) + 1)

    # Training Loss plot
    ax1 = axs[0, 0]
    if train_history.get('train_loss'):
        ax1.plot(train_indices, train_history['train_loss'], label='Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    # Test Loss plot
    ax2 = axs[0, 1]
    if test_history.get('test_loss'):
        ax2.plot(test_indices, test_history['test_loss'], label='Test Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Test Loss')
    ax2.legend()
    ax2.grid(True)

    # Training RMSE plot
    ax3 = axs[1, 0]
    if train_history.get('train_rmse'):
        ax3.plot(train_indices, train_history['train_rmse'], label='Training RMSE', color='orange')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('RMSE')
    ax3.set_title('Training RMSE')
    ax3.legend()
    ax3.grid(True)

    # Test RMSE plot
    ax4 = axs[1, 1]
    if test_history.get('test_rmse'):
        ax4.plot(test_indices, test_history['test_rmse'], label='Test RMSE', color='red')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('RMSE')
    ax4.set_title('Test RMSE')
    ax4.legend()
    ax4.grid(True)

    # Precision and Recall plot
    ax5 = axs[2, 0]
    if test_history.get('precision_at_k'):
        ax5.plot(test_indices, test_history['precision_at_k'], label='Precision@K')
    if test_history.get('recall_at_k'):
        ax5.plot(test_indices, test_history['recall_at_k'], label='Recall@K')
    ax5.set_xlabel('Rounds')
    ax5.set_ylabel('Score')
    ax5.set_title('Precision and Recall')
    ax5.legend()
    ax5.grid(True)

    # NDCG plot
    ax6 = axs[2, 1]
    if test_history.get('ndcg_at_k'):
        ax6.plot(test_indices, test_history['ndcg_at_k'], label='NDCG@K', color='green')
    ax6.set_xlabel('Rounds')
    ax6.set_ylabel('Score')
    ax6.set_title('NDCG@K')
    ax6.legend()
    ax6.grid(True)

    # Coverage plot
    ax7 = axs[3, 0]
    if test_history.get('coverage'):
        ax7.plot(test_indices, test_history['coverage'], label='Coverage', color='sienna')
    ax7.set_xlabel('Rounds')
    ax7.set_ylabel('Coverage')
    ax7.set_title('Coverage')
    ax7.legend()
    ax7.grid(True)

    # ROC AUC plot
    ax8 = axs[3, 1]
    if test_history.get('roc_auc'):
        ax8.plot(test_indices, test_history['roc_auc'], label='ROC AUC', color='navy')
    ax8.set_xlabel('Rounds')
    ax8.set_ylabel('ROC AUC')
    ax8.set_title('ROC AUC')
    ax8.legend()
    ax8.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Saved metrics plot to {save_path}")

def plot_metrics_from_files(train_history_file='train_history.json', test_history_file='test_history.json', save_path='metrics_plots.png'):
    """Load histories from JSON files and plot metrics."""
    try:
        with open(train_history_file, 'r') as f:
            train_history = json.load(f)
    except FileNotFoundError:
        logger.warning(f"{train_history_file} not found.")
        train_history = {'rounds': []}

    try:
        with open(test_history_file, 'r') as f:
            test_history = json.load(f)
    except FileNotFoundError:
        logger.warning(f"{test_history_file} not found.")
        test_history = {'rounds': []}

    plot_metrics_history(train_history, test_history, save_path)

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