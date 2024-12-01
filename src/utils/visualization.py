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
    
    fig, axs = plt.subplots(5, 2, figsize=(15, 30))
    metrics = data['metrics']
    
    # Loss metrics (row 0)
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
    
    # RMSE metrics (row 1)
    if 'train_rmse' in metrics:
        epochs = np.arange(1, len(metrics['train_rmse']) + 1)
        axs[1, 0].plot(epochs, metrics['train_rmse'], label='Training RMSE')
        axs[1, 0].set_title('Training RMSE')
        axs[1, 0].set_xlabel('Client Epochs')
        axs[1, 0].set_ylabel('RMSE')
        axs[1, 0].legend()
    
    if 'eval_rmse' in metrics:
        rounds = np.arange(1, len(metrics['eval_rmse']) + 1)
        axs[1, 1].plot(rounds, metrics['eval_rmse'], label='Eval RMSE')
        axs[1, 1].set_title('Eval RMSE')
        axs[1, 1].set_xlabel('Aggregation Rounds')
        axs[1, 1].set_ylabel('RMSE')
        axs[1, 1].legend()
    
    # User Preference Evolution (row 2)
    if 'train_ut_norm' in metrics:
        epochs = np.arange(1, len(metrics['train_ut_norm']) + 1)
        axs[2, 0].plot(epochs, metrics['train_ut_norm'], label='Training User Preference Norm')
    if 'eval_ut_norm' in metrics:
        rounds = np.arange(1, len(metrics['eval_ut_norm']) + 1)
        axs[2, 0].plot(rounds, metrics['eval_ut_norm'], label='Eval User Preference Norm')
    axs[2, 0].set_title('User Preference Evolution')
    axs[2, 0].set_xlabel('Rounds/Epochs')
    axs[2, 0].set_ylabel('Norm')
    axs[2, 0].legend()
    
    # Probability metrics (row 2)
    prob_metrics = ['train_likable_prob', 'train_nonlikable_prob', 'eval_likable_prob', 'eval_nonlikable_prob']
    for metric in prob_metrics:
        if metric in metrics:
            x = np.arange(1, len(metrics[metric]) + 1)
            label = metric.replace('_prob', '').replace('_', ' ').title()
            if 'train' in metric:
                label = f"Training {label.replace('Train ', '')}"
            else:
                label = f"Eval {label.replace('Eval ', '')}"
            axs[2, 1].plot(x, metrics[metric], 
                          label=label,
                          linestyle='--' if 'train' in metric else '-')
    axs[2, 1].set_title('Item Type Probabilities')
    axs[2, 1].set_xlabel('Rounds/Epochs')
    axs[2, 1].set_ylabel('Probability')
    axs[2, 1].legend()
    
    # Recommendation metrics (row 3)
    rec_metrics = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'coverage']
    for metric in rec_metrics:
        if metric in metrics:
            rounds = np.arange(1, len(metrics[metric]) + 1)
            axs[3, 0].plot(rounds, metrics[metric], label=metric.replace('_', ' ').title())
    axs[3, 0].set_title('Recommendation Metrics')
    axs[3, 0].set_xlabel('Rounds')
    axs[3, 0].set_ylabel('Value')
    axs[3, 0].legend()
    
    # ROC AUC (row 3)
    if 'roc_auc' in metrics:
        rounds = np.arange(1, len(metrics['roc_auc']) + 1)
        axs[3, 1].plot(rounds, metrics['roc_auc'], label='ROC AUC')
        axs[3, 1].set_title('ROC AUC Score')
        axs[3, 1].set_xlabel('Rounds')
        axs[3, 1].set_ylabel('Score')
        axs[3, 1].set_ylim([0, 1])
        axs[3, 1].legend()
    
    # Model Divergence metrics (row 4)
    div_metrics = ['local_global_divergence', 'personalization_degree']
    for metric in div_metrics:
        if metric in metrics:
            rounds = np.arange(1, len(metrics[metric]) + 1)
            axs[4, 0].plot(rounds, metrics[metric], label=metric.replace('_', ' ').title())
    axs[4, 0].set_title('Model Divergence Metrics')
    axs[4, 0].set_xlabel('Rounds')
    axs[4, 0].set_ylabel('Divergence')
    axs[4, 0].legend()
    axs[4, 0].set_ylim(bottom=0)
    
    # Correlated Mass (row 4)
    if 'train_correlated_mass' in metrics:
        epochs = np.arange(1, len(metrics['train_correlated_mass']) + 1)
        axs[4, 1].plot(epochs, metrics['train_correlated_mass'], label='Training')
    if 'eval_correlated_mass' in metrics:
        rounds = np.arange(1, len(metrics['eval_correlated_mass']) + 1)
        axs[4, 1].plot(rounds, metrics['eval_correlated_mass'], label='Eval')
    axs[4, 1].set_title('Correlated Mass')
    axs[4, 1].set_xlabel('Rounds/Epochs')
    axs[4, 1].set_ylabel('Value')
    axs[4, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{base_filename}.png")
    plt.close()

    logger.info(f"Saved plot to {output_dir}/{base_filename}.png")