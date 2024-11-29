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
        histories = json.load(f)
    
    base_filename = os.path.splitext(os.path.basename(history_file))[0]
    output_dir = os.path.join("plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a single large figure with 3x2 subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    history = histories['history']  # Use consolidated history
    
    # Training plots (top row)
    if history.get('train_loss'):
        axs[0, 0].plot(history['train_loss'], label='Training Loss')
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].set_xlabel('Rounds')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
    
    if history.get('train_rmse'):
        axs[0, 1].plot(history['train_rmse'], label='Training RMSE')
        axs[0, 1].set_title('Training RMSE')
        axs[0, 1].set_xlabel('Rounds')
        axs[0, 1].set_ylabel('RMSE')
        axs[0, 1].legend()
    
    # Test plots (middle row)
    if history.get('test_rmse'):
        axs[1, 0].plot(history['test_rmse'], label='Test RMSE')
        axs[1, 0].set_title('Test RMSE')
        axs[1, 0].set_xlabel('Rounds')
        axs[1, 0].set_ylabel('RMSE')
        axs[1, 0].legend()
    
    # Recommendation metrics
    metrics_to_plot = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'coverage']
    for metric in metrics_to_plot:
        if metric in history:
            axs[1, 1].plot(history[metric], label=metric.replace('_', ' ').title())
    axs[1, 1].set_title('Recommendation Metrics')
    axs[1, 1].set_xlabel('Rounds')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].legend()
    
    # Preference evolution metrics (bottom row)
    if 'metrics' in histories:
        metrics = histories['metrics']
        
        if 'ut_norm' in metrics:
            axs[2, 0].plot(metrics['ut_norm'], label='User Preference Norm')
            axs[2, 0].set_title('User Preference Norm')
            axs[2, 0].set_xlabel('Rounds')
            axs[2, 0].set_ylabel('Norm')
            axs[2, 0].legend()
        
        if 'likable_prob' in metrics and 'nonlikable_prob' in metrics:
            axs[2, 1].plot(metrics['likable_prob'], label='Likable Items')
            axs[2, 1].plot(metrics['nonlikable_prob'], label='Non-likable Items')
            axs[2, 1].set_title('Item Type Probabilities')
            axs[2, 1].set_xlabel('Rounds')
            axs[2, 1].set_ylabel('Probability')
            axs[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_{base_filename}.png")
    plt.close()

    logger.info(f"Saved plot to {output_dir}/plot_{base_filename}.png")