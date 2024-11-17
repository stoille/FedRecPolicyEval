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

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Loss plot
    if train_history.get('rounds'):
        # Create x-axis values for training (fractional steps between rounds)
        train_steps = np.linspace(train_history['rounds'][0], train_history['rounds'][-1], len(train_history['train_loss']))
        axs[0, 0].plot(train_steps, train_history['train_loss'], label='Training Loss')
    if test_history.get('rounds'):
        axs[0, 0].plot(test_history['rounds'], test_history['test_loss'], label='Test Loss')
    axs[0, 0].set_xlabel('Rounds')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # RMSE plot
    if train_history.get('rounds'):
        train_steps = np.linspace(train_history['rounds'][0], train_history['rounds'][-1], len(train_history['train_rmse']))
        axs[0, 1].plot(train_steps, train_history['train_rmse'], label='Training RMSE', color='orange')
    if test_history.get('rounds'):
        axs[0, 1].plot(test_history['rounds'], test_history['test_rmse'], label='Test RMSE', color='red')
    axs[0, 1].set_xlabel('Rounds')
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].set_title('Root Mean Squared Error')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Precision/Recall plot
    axs[0, 2].plot(test_history['rounds'], test_history['precision_at_k'], label='Precision@K')
    axs[0, 2].plot(test_history['rounds'], test_history['recall_at_k'], label='Recall@K')
    axs[0, 2].set_xlabel('Rounds')
    axs[0, 2].set_ylabel('Score')
    axs[0, 2].set_title('Precision@K and Recall@K')
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # NDCG plot
    axs[1, 0].plot(test_history['rounds'], test_history['ndcg_at_k'], label='NDCG@K', color='lime')
    axs[1, 0].set_xlabel('Rounds')
    axs[1, 0].set_ylabel('NDCG')
    axs[1, 0].set_title('Normalized Discounted Cumulative Gain')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # ROC AUC plot
    axs[1, 1].plot(test_history['rounds'], test_history['roc_auc'], label='ROC AUC', color='navy')
    axs[1, 1].set_xlabel('Rounds')
    axs[1, 1].set_ylabel('ROC AUC')
    axs[1, 1].set_title('ROC AUC')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Coverage plot
    axs[1, 2].plot(test_history['rounds'], test_history['coverage'], label='Coverage', color='sienna')
    axs[1, 2].set_xlabel('Rounds')
    axs[1, 2].set_ylabel('Coverage')
    axs[1, 2].set_title('Coverage')
    axs[1, 2].legend()
    axs[1, 2].grid(True)

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