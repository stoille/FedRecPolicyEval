import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

json_data = [
    {
        "test_loss": [0.043523, 0.036008, 0.041728],
        "test_rmse": [0.063213, 0.07444, 0.067307],
        "precision_at_k": [0.109054, 0.143368, 0.133338],
        "recall_at_k": [0.049003, 0.066544, 0.075022],
        "ndcg_at_k": [0.101419, 0.148554, 0.147737],
        "coverage": [0.005947, 0.018349, 0.01936],
        "roc_auc": [0.517542, 0.512049, 0.566201],
        "rounds": [1, 2, 3],
        "user_norm": [1.1, 1.8, 2.7],
        "avg_prob_likable": [0.1, 0.2, 0.3],
        "ut_local": [0.45, 0.355, 0.85],
        "ut_global": [0.24, 0.56, 0.88],
        "pers_degree": [0.5, 0.6, 0.7],
        "N": 1,
    },
    {
        "test_loss": [0.036778, 0.041722, 0.036233],
        "test_rmse": [0.064842, 0.061849, 0.06612],
        "precision_at_k": [0.10131, 0.149023, 0.140566],
        "recall_at_k": [0.044708, 0.068018, 0.079813],
        "ndcg_at_k": [0.109197, 0.144086, 0.160923],
        "coverage": [0.005879, 0.016165, 0.02284],
        "roc_auc": [0.54874, 0.584139, 0.522526],
        "rounds": [1, 2, 3],
        "user_norm": [1.2, 2.0, 2.8],
        "avg_prob_likable": [0.15, 0.25, 0.35],
        "ut_local": [0.45, 0.25, 0.65],
        "ut_global": [0.15, 0.55, 0.85],
        "pers_degree": [0.55, 0.65, 0.75],
        "N": 20,
    },
    {
        "test_loss": [0.036446, 0.042446, 0.039065],
        "test_rmse": [0.066635, 0.071072, 0.070204],
        "precision_at_k": [0.103313, 0.135441, 0.154232],
        "recall_at_k": [0.048933, 0.067583, 0.074915],
        "ndcg_at_k": [0.103452, 0.136647, 0.156559],
        "coverage": [0.006804, 0.016289, 0.02251],
        "roc_auc": [0.536972, 0.511474, 0.533227],
        "rounds": [1, 2, 3],
        "user_norm": [1.3, 2.1, 3.0],
        "avg_prob_likable": [0.2, 0.3, 0.4],
        "ut_local": [0.27, 0.33, 0.55],
        "ut_global": [0.25, 0.35, 0.45],
        "pers_degree": [0.6, 0.7, 0.8],
        "N": 50,
    },
    {
        "test_loss": [0.036575, 0.042021, 0.036983],
        "test_rmse": [0.071647, 0.065514, 0.071071],
        "precision_at_k": [0.110734, 0.148883, 0.135956],
        "recall_at_k": [0.04641, 0.067143, 0.073064],
        "ndcg_at_k": [0.119763, 0.157592, 0.15432],
        "coverage": [0.006036, 0.018427, 0.020392],
        "roc_auc": [0.54739, 0.51105, 0.540537],
        "rounds": [1, 2, 3],
        "user_norm": [1.4, 2.3, 3.2],
        "avg_prob_likable": [0.25, 0.35, 0.45],
        "ut_local": [0.25, 0.35, 0.45],
        "ut_global": [0.25, 0.35, 0.45],
        "pers_degree": [0.65, 0.75, 0.85],
        "N": 100,
    }
]

json=[
  {
    "parameters": {
      "model_type": "mf",
      "num_rounds": 3,
      "local_epochs": 5,
      "beta": 0.01,
      "gamma": 0.01,
      "num_nodes": 1,
      "learning_rate_schedule": "constant"
    },
    "metrics": {
      "ut_norm":  [1.3, 2.2, 3.0],
      "ut_local": [[0.45, 0.355, 0.85], [0.45, 0.25, 0.65], [0.4, 0.3, 0.7]],
      "ut_global": [[0.24, 0.56, 0.88], [0.15, 0.55, 0.85], [0.2, 0.6, 0.9]],
      "likable_prob": [0.1, 0.2, 0.3],
      "nonlikable_prob": [0.9, 0.8, 0.7], # Are you calculation this already for me? see formla in table of page 3 paper? I assume yes have no porblem to do it my self
      "correlated_mass": [0.05, 0.07, 0.1]
    },
    "train_history": {
      "train_loss": [0.05, 0.045, 0.04],
      "train_rmse": [0.07, 0.065, 0.06],
      "rounds": [1, 2, 3]
    },
    "test_history": {
      "test_loss": [0.043523, 0.036008, 0.041728],
      "test_rmse": [0.063213, 0.07444, 0.067307],
      "precision_at_k": [0.109054, 0.143368, 0.133338],
      "recall_at_k": [0.049003, 0.066544, 0.075022],
      "ndcg_at_k": [0.101419, 0.148554, 0.147737],
      "coverage": [0.005947, 0.018349, 0.01936]
    }
  },
  {
    "parameters": {
      "model_type": "mf",
      "num_rounds": 3,
      "local_epochs": 10,
      "beta": 0.005,
      "gamma": 0.02,
      "num_nodes": 20,
      "learning_rate_schedule": "adaptive"
    },
    "metrics": {
      "ut_norm": [1.5, 2.5, 3.4],
      "ut_local": [[0.27, 0.33, 0.55], [0.25, 0.35, 0.45], [0.3, 0.4, 0.5]],
      "ut_global": [[0.25, 0.35, 0.45], [0.15, 0.55, 0.85], [0.2, 0.6, 0.9]],
      "likable_prob": [0.15, 0.25, 0.35],
      "nonlikable_prob": [0.85, 0.75, 0.65],
      "correlated_mass": [0.06, 0.08, 0.12]
    },
    "train_history": {
      "train_loss": [0.045, 0.042, 0.039],
      "train_rmse": [0.065, 0.061, 0.058],
      "rounds": [1, 2, 3]
    },
    "test_history": {
      "test_loss": [0.036575, 0.042021, 0.036983],
      "test_rmse": [0.071647, 0.065514, 0.071071],
      "precision_at_k": [0.110734, 0.148883, 0.135956],
      "recall_at_k": [0.04641, 0.067143, 0.073064],
      "ndcg_at_k": [0.119763, 0.157592, 0.15432],
      "coverage": [0.006036, 0.018427, 0.020392]
    }
  },
  {
    "parameters": {
      "model_type": "nn",
      "num_rounds": 3,
      "local_epochs": 15,
      "beta": 0.02,
      "gamma": 0.03,
      "num_nodes": 50,
      "learning_rate_schedule": "decay"
    },
    "metrics": {
      "ut_norm": [1.7, 2.7, 3.7],
      "ut_local": [[0.35, 0.45, 0.55], [0.4, 0.5, 0.6], [0.45, 0.55, 0.65]],
      "ut_global": [[0.3, 0.4, 0.5], [0.35, 0.45, 0.55], [0.4, 0.5, 0.6]],
      "likable_prob": [0.2, 0.3, 0.4],
      "nonlikable_prob": [0.8, 0.7, 0.6],
      "correlated_mass": [0.07, 0.09, 0.11]
    },
    "train_history": {
      "train_loss": [0.04, 0.037, 0.035],
      "train_rmse": [0.06, 0.055, 0.05],
      "rounds": [1, 2, 3]
    },
    "test_history": {
      "test_loss": [0.034523, 0.033008, 0.031728],
      "test_rmse": [0.062213, 0.06444, 0.061307],
      "precision_at_k": [0.119054, 0.153368, 0.143338],
      "recall_at_k": [0.059003, 0.076544, 0.085022],
      "ndcg_at_k": [0.111419, 0.158554, 0.157737],
      "coverage": [0.007947, 0.018349, 0.01936]
    }
  },
  {
    "parameters": {
      "model_type": "rf",
      "num_rounds": 3,
      "local_epochs": 8,
      "beta": 0.01,
      "gamma": 0.02,
      "num_nodes": 100,
      "learning_rate_schedule": "linear"
    },
    "metrics": {
      "ut_norm": [1.2, 1.7, 2.2],
      "ut_local": [[0.25, 0.35, 0.45], [0.3, 0.4, 0.5], [0.35, 0.45, 0.55]],
      "ut_global": [[0.2, 0.3, 0.4], [0.25, 0.35, 0.45], [0.3, 0.4, 0.5]],
      "likable_prob": [0.12, 0.22, 0.32],
      "nonlikable_prob": [0.88, 0.78, 0.68],
      "correlated_mass": [0.04, 0.06, 0.08]
    },
    "train_history": {
      "train_loss": [0.048, 0.043, 0.038],
      "train_rmse": [0.066, 0.062, 0.059],
      "rounds": [1, 2, 3]
    },
    "test_history": {
      "test_loss": [0.037523, 0.036008, 0.039728],
      "test_rmse": [0.064213, 0.06944, 0.067307],
      "precision_at_k": [0.105054, 0.135368, 0.125338],
      "recall_at_k": [0.045003, 0.063544, 0.070022],
      "coverage": [0.007947, 0.018349, 0.01936]
    }
  }]


def calculate_cosine_similarity_for_vectors(local, global_):
    dot_product = np.dot(local, global_)
    norm_local = np.linalg.norm(local)
    norm_global = np.linalg.norm(global_)
    return dot_product / (norm_local * norm_global)

# Prepare data and calculate cosine similarities
cosine_similarities = {}
for data in json:
    num_nodes = data['parameters']['num_nodes']
    cosine_similarities[f"N{num_nodes}"] = [
        calculate_cosine_similarity_for_vectors(
            np.array(data['metrics']["ut_local"][i]),
            np.array(data['metrics']["ut_global"][i])
        )
        for i in range(len(data['metrics']["ut_local"]))
    ]

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

# Plot 1: User norm ||ut||
for data in json:
    num_nodes = data['parameters']['num_nodes']
    axes[0].plot(
        range(1, len(data['metrics']['ut_norm']) + 1),
        data['metrics']['ut_norm'],
        linewidth=2,
        marker="o",
        label=f"N={num_nodes}"
    )
axes[0].set_title("User norm ||ut||", fontsize=16)
axes[0].set_xlabel("Training Rounds", fontsize=14)
axes[0].set_ylabel("User norm ||ut||", fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True)

# Plot 2: Likable and non-likable probabilities
for data in json:
    num_nodes = data['parameters']['num_nodes']
    axes[1].plot(
        range(1, len(data['metrics']['likable_prob']) + 1),
        data['metrics']['likable_prob'],
        linewidth=2,
        marker="o",
        label=f"Likable (N={num_nodes})"
    )
    axes[1].plot(
        range(1, len(data['metrics']['nonlikable_prob']) + 1),
        data['metrics']['nonlikable_prob'],
        linewidth=2,
        marker="o",
        linestyle='dashed',
        label=f"Non-likable (N={num_nodes})"
    )
axes[1].set_title("Average Probability for Likable/Non-likable Items", fontsize=16)
axes[1].set_xlabel("Training Rounds", fontsize=14)
axes[1].set_ylabel("Probability", fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True)

# Plot 3: Cosine similarity
for data in json:
    num_nodes = data['parameters']['num_nodes']
    axes[2].plot(
        range(1, len(cosine_similarities[f"N{num_nodes}"]) + 1),
        cosine_similarities[f"N{num_nodes}"],
        linewidth=2,
        marker="o",
        label=f"N={num_nodes}"
    )
axes[2].set_title("Cosine Similarity (Local vs Global)", fontsize=16)
axes[2].set_xlabel("Training Rounds", fontsize=14)
axes[2].set_ylabel("Cosine Similarity", fontsize=14)
axes[2].legend(fontsize=10)
axes[2].grid(True)

# Empty placeholder or additional plot
axes[3].set_visible(False)  # You can use this for any additional plot

plt.tight_layout()
plt.show()
