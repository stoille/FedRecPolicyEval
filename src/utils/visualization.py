import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

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

def plot_history(history):
    rounds = sorted(history.keys())
    metrics = {
        'losses': ('train_loss', 'val_loss'),
        'error': ('rmse',),
        'ranking': ('precision_at_k', 'recall_at_k'),
        'ndcg': ('ndcg_at_k',),
        'roc': ('roc_auc',),
        'coverage': ('coverage',)
    }

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    plot_metrics(axs, rounds, history, metrics)
    
    plt.tight_layout()
    plt.savefig('metrics_plots.png')
    plt.close()

def plot_metrics(axs, rounds, history, metrics):
    for (i, j), (title, metric_keys) in zip(
        [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
        metrics.items()
    ):
        ax = axs[i, j]
        for key in metric_keys:
            values = [history[r].get(key) for r in rounds]
            ax.plot(rounds, values, label=key)
        ax.set_xlabel('Rounds')
        ax.set_ylabel(title.capitalize())
        ax.set_title(f'{title.capitalize()} Metrics')
        ax.legend()