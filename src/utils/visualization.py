import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import json
import logging
import torch
import os

logger = logging.getLogger("Visualization")

def plot_metrics_from_file(history_file: str):
    """Plot all metrics from a consolidated history file."""
    with open(history_file, 'r') as f:
        data = json.load(f)
    
    base_filename = os.path.splitext(os.path.basename(history_file))[0]
    output_dir = os.path.join("plots")
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    metrics = data['metrics']
    
    # Training loss plot
    if 'epoch_train_loss' in metrics:
        epochs = np.arange(1, len(metrics['epoch_train_loss']) + 1)
        axs[0, 0].plot(epochs, metrics['epoch_train_loss'], label='Training Loss')
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].set_xlabel('Client Epochs')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
    
    # Eval loss plot
    if 'eval_loss' in metrics:
        rounds = np.arange(1, len(metrics['eval_loss']) + 1)
        axs[0, 1].plot(rounds, metrics['eval_loss'], label='Eval Loss')
        axs[0, 1].set_title('Eval Loss')
        axs[0, 1].set_xlabel('Aggregation Rounds')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
    
    # Training RMSE plot
    if 'epoch_train_rmse' in metrics:
        epochs = np.arange(1, len(metrics['epoch_train_rmse']) + 1)
        axs[1, 0].plot(epochs, metrics['epoch_train_rmse'], label='Training RMSE')
        axs[1, 0].set_title('Training RMSE')
        axs[1, 0].set_xlabel('Client Epochs')
        axs[1, 0].set_ylabel('RMSE')
        axs[1, 0].legend()
    
    # Eval RMSE plot
    if 'eval_rmse' in metrics:
        rounds = np.arange(1, len(metrics['eval_rmse']) + 1)
        axs[1, 1].plot(rounds, metrics['eval_rmse'], label='Eval RMSE')
        axs[1, 1].set_title('Eval RMSE')
        axs[1, 1].set_xlabel('Aggregation Rounds')
        axs[1, 1].set_ylabel('RMSE')
        axs[1, 1].legend()
    
    # ROC_AUC plot
    if 'roc_auc' in metrics:
        rounds = np.arange(1, len(metrics['roc_auc']) + 1)
        axs[2, 0].plot(rounds, metrics['roc_auc'], label='ROC AUC')
        axs[2, 0].set_title('ROC AUC Score')
        axs[2, 0].set_xlabel('Aggregation Rounds')
        axs[2, 0].set_ylabel('Score')
        axs[2, 0].legend()
    
    # Recommendation metrics
    metrics_to_plot = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'coverage']
    for metric in metrics_to_plot:
        if metric in metrics:
            rounds = range(1, len(metrics[metric]) + 1)
            axs[1, 1].plot(rounds, metrics[metric], label=metric.replace('eval_', '').replace('_', ' ').title())
    axs[1, 1].set_title('Recommendation Metrics')
    axs[1, 1].set_xlabel('Rounds')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].legend()
    
    # Preference evolution metrics (bottom row)
    if 'eval_ut_norm' in metrics:
        rounds = range(1, len(metrics['eval_ut_norm']) + 1)
        axs[2, 0].plot(rounds, metrics['eval_ut_norm'], label='User Preference Norm')
        axs[2, 0].set_title('User Preference Norm')
        axs[2, 0].set_xlabel('Rounds')
        axs[2, 0].set_ylabel('Norm')
        axs[2, 0].legend()
        
        if 'eval_likable_prob' in metrics and 'eval_nonlikable_prob' in metrics:
            axs[2, 1].plot(rounds, metrics['eval_likable_prob'], label='Likable Items')
            axs[2, 1].plot(rounds, metrics['eval_nonlikable_prob'], label='Non-likable Items')
            axs[2, 1].set_title('Item Type Probabilities')
            axs[2, 1].set_xlabel('Rounds')
            axs[2, 1].set_ylabel('Probability')
            axs[2, 1].legend()
    
    if 'local_global_divergence' in metrics:
        rounds = range(1, len(metrics['local_global_divergence']) + 1)
        axs[2, 1].plot(rounds, metrics['local_global_divergence'], label='Local-Global')
        if 'personalization_degree' in metrics:
            axs[2, 1].plot(rounds, metrics['personalization_degree'], label='Local-Local')
        if 'max_local_divergence' in metrics:
            axs[2, 1].plot(rounds, metrics['max_local_divergence'], label='Max Local')
        axs[2, 1].set_title('Model Divergence')
        axs[2, 1].set_xlabel('Rounds')
        axs[2, 1].set_ylabel('Cosine Distance')
        axs[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{base_filename}.png")
    plt.close()

    logger.info(f"Saved plot to {output_dir}/{base_filename}.png")