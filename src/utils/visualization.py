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
    
    fig, axs = plt.subplots(4, 2, figsize=(15, 24))
    metrics = data['metrics']
    
    # Training metrics (top row)
    if 'train_loss' in metrics:
        epochs = np.arange(1, len(metrics['train_loss']) + 1)
        axs[0, 0].plot(epochs, metrics['train_loss'], label='Training Loss')
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].set_xlabel('Client Epochs')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
    
    if 'eval_loss' in metrics:
        rounds = np.arange(1, len(metrics['eval_loss']) + 1)
        axs[0, 1].plot(rounds, metrics['eval_loss'], label='Eval Loss')
        axs[0, 1].set_title('Eval Loss')
        axs[0, 1].set_xlabel('Aggregation Rounds')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
    
    # RMSE metrics (middle row, left)
    if 'train_rmse' in metrics:
        epochs = np.arange(1, len(metrics['train_rmse']) + 1)
        axs[1, 0].plot(epochs, metrics['train_rmse'], label='Training RMSE')
    if 'eval_rmse' in metrics:
        rounds = np.arange(1, len(metrics['eval_rmse']) + 1)
        axs[1, 0].plot(rounds, metrics['eval_rmse'], label='Eval RMSE')
    axs[1, 0].set_title('RMSE Metrics')
    axs[1, 0].set_xlabel('Epochs/Rounds')
    axs[1, 0].set_ylabel('RMSE')
    axs[1, 0].legend()
    
    # ROC AUC (middle row, right)
    if 'roc_auc' in metrics:
        rounds = np.arange(1, len(metrics['roc_auc']) + 1)
        axs[1, 1].plot(rounds, metrics['roc_auc'], label='ROC AUC')
        axs[1, 1].set_title('ROC AUC Score')
        axs[1, 1].set_xlabel('Rounds')
        axs[1, 1].set_ylabel('Score')
        axs[1, 1].set_ylim([0, 1])
        axs[1, 1].legend()
    
    # Recommendation metrics (bottom-1 row, left)
    rec_metrics = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'coverage']
    for metric in rec_metrics:
        if metric in metrics:
            rounds = np.arange(1, len(metrics[metric]) + 1)
            axs[2, 0].plot(rounds, metrics[metric], label=metric.replace('_', ' ').title())
    axs[2, 0].set_title('Recommendation Metrics')
    axs[2, 0].set_xlabel('Rounds')
    axs[2, 0].set_ylabel('Value')
    axs[2, 0].legend()
    
    # User Preference Evolution (bottom-1 row, right)
    if 'eval_ut_norm' in metrics:
        rounds = np.arange(1, len(metrics['eval_ut_norm']) + 1)
        axs[2, 1].plot(rounds, metrics['eval_ut_norm'], label='User Preference Norm')
        axs[2, 1].set_title('User Preference Evolution')
        axs[2, 1].set_xlabel('Rounds')
        axs[2, 1].set_ylabel('Norm')
        axs[2, 1].legend()
    
    # Probability metrics (bottom row)
    prob_metrics = ['eval_likable_prob', 'eval_nonlikable_prob']
    for metric in prob_metrics:
        if metric in metrics:
            rounds = np.arange(1, len(metrics[metric]) + 1)
            axs[3, 0].plot(rounds, metrics[metric], 
                          label=metric.replace('eval_', '').replace('_prob', '').replace('_', ' ').title())
    axs[3, 0].set_title('Item Type Probabilities')
    axs[3, 0].set_xlabel('Rounds')
    axs[3, 0].set_ylabel('Probability')
    axs[3, 0].legend()
    
    # Keep the last subplot empty
    axs[3, 1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{base_filename}.png")
    plt.close()

    logger.info(f"Saved plot to {output_dir}/{base_filename}.png")