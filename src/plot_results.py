import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np

# Load history
with open('history.json', 'r') as f:
    history = json.load(f)

# Extract rounds
rounds = sorted(map(int, history.keys()))

# Prepare data
train_losses = [history[str(r)].get('train_loss') for r in rounds]
val_losses = [history[str(r)].get('val_loss') for r in rounds]
rmse = [history[str(r)].get('rmse') for r in rounds]
precision_at_k = [history[str(r)].get('precision_at_k') for r in rounds]
recall_at_k = [history[str(r)].get('recall_at_k') for r in rounds]
ndcg_at_k = [history[str(r)].get('ndcg_at_k') for r in rounds]
hit_rate_at_k = [history[str(r)].get('hit_rate_at_k') for r in rounds]
baseline_hit_rate = [history[str(r)].get('baseline_hit_rate') for r in rounds]
baseline_ndcg = [history[str(r)].get('baseline_ndcg') for r in rounds]
coverage = [history[str(r)].get('coverage') for r in rounds]

# Prepare data for ROC AUC
roc_auc = [history[str(r)].get('roc_auc') for r in rounds]  # Assuming 'roc_auc' is the key for ROC AUC values

# Initialize subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Plot Training and Validation Loss
axs[0, 0].plot(rounds, train_losses, label='Training Loss')
axs[0, 0].plot(rounds, val_losses, label='Validation Loss')
axs[0, 0].set_xlabel('Rounds')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].set_title('Training and Validation Loss')
axs[0, 0].legend()

# Plot RMSE
axs[0, 1].plot(rounds, rmse, label='RMSE', color='orange')
axs[0, 1].set_xlabel('Rounds')
axs[0, 1].set_ylabel('RMSE')
axs[0, 1].set_title('Root Mean Squared Error')
axs[0, 1].legend()

# Plot Precision@K and Recall@K
axs[0, 2].plot(rounds, precision_at_k, label='Precision@K')
axs[0, 2].plot(rounds, recall_at_k, label='Recall@K')
axs[0, 2].set_xlabel('Rounds')
axs[0, 2].set_ylabel('Score')
axs[0, 2].set_title('Precision@K and Recall@K')
axs[0, 2].legend()

# Plot NDCG@K
axs[1, 0].plot(rounds, ndcg_at_k, label='NDCG@K', color='green')
axs[1, 0].plot(rounds, baseline_ndcg, label='Baseline NDCG', color='red', linestyle='--')
axs[1, 0].set_xlabel('Rounds')
axs[1, 0].set_ylabel('NDCG')
axs[1, 0].set_title('Normalized Discounted Cumulative Gain')
axs[1, 0].legend()

# Plot ROC AUC instead of Hit Rate@K
axs[1, 1].plot(rounds, roc_auc, label='ROC AUC', color='blue')  # Change the label and color as needed
# axs[1, 1].plot(rounds, baseline_hit_rate, label='Baseline Hit Rate', color='red', linestyle='--')  # Remove baseline hit rate plot
axs[1, 1].set_xlabel('Rounds')
axs[1, 1].set_ylabel('ROC AUC')
axs[1, 1].set_title('ROC AUC')
axs[1, 1].legend()

# Plot Coverage
axs[1, 2].plot(rounds, coverage, label='Coverage', color='brown')
axs[1, 2].set_xlabel('Rounds')
axs[1, 2].set_ylabel('Coverage')
axs[1, 2].set_title('Coverage')
axs[1, 2].legend()

plt.tight_layout()
plt.savefig('metrics_plots.png')
plt.show()
