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
def calculate_cosine_similarity_for_vectors(local, global_):

    dot_product = np.dot(local, global_)
    norm_local = np.linalg.norm(local)
    norm_global = np.linalg.norm(global_)
    return dot_product / (norm_local * norm_global)

N_values = []
cosine_similarities = []
cosine_similarities_local = []

for data in json_data:
    N_values.append(data['N'])
    # Cosine similarity between ut_local and ut_global
    cosine_similarity = calculate_cosine_similarity_for_vectors(
        np.array(data["ut_local"]), np.array(data["ut_global"])
    )
    cosine_similarities.append(cosine_similarity)

# Calculate pairwise cosine similarities between ut_local vectors across different N
for data_i in json_data:
    local_similarities = []
    for data_j in json_data:
        similarity = calculate_cosine_similarity_for_vectors(
            np.array(data_i["ut_local"]), np.array(data_j["ut_local"])
        )
        local_similarities.append(similarity)
    cosine_similarities_local.append(local_similarities)


fig, axes = plt.subplots(2, 2, figsize=(15, 12))

titles_and_metrics = [
    ("User norm ||ut||", "user_norm"),
    ("Average probability for likable items", "avg_prob_likable"),
]

for ax, (title, metric) in zip(axes.flat, titles_and_metrics):
    for data in json_data:
        if metric in data:
            ax.plot(
                data["rounds"],
                data[metric],
                linewidth=2,
                marker="o",
                label=f"N={data['N']}"
            )
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Training Rounds", fontsize=14)
    ax.set_ylabel(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True)

# plot Local vs global model divergence
x_labels = [f'N{N}' for N in N_values]
next_ax_index = len(titles_and_metrics)
ax = axes.flat[next_ax_index]
ax.bar(x_labels, cosine_similarities, color='skyblue')
ax.set_title('Local vs global model divergence', fontsize=16)
ax.set_xlabel('N', fontsize=14)
ax.set_ylabel('Cosine Similarity', fontsize=14)
ax.grid(True)
ax = axes.flat[next_ax_index + 1]

#plot Personalisation degree
heatmap_labels = [f'N{N} ut_local' for N in N_values]
sns.heatmap(
    data=cosine_similarities_local,
    annot=True,
    fmt=".3f",
    xticklabels=heatmap_labels,
    yticklabels=heatmap_labels,
    ax=ax,
)
ax.set_title('Personalisation degree after '+
str(len(json_data.__getitem__(0)['rounds']))+' rounds', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('')

plt.tight_layout()
plt.show()
