"""FedRecPolicyEval: A Flower / PyTorch package for evaluating recommender system policies."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from collections import OrderedDict

class VAE(nn.Module):
    def __init__(self, num_items, latent_dim=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_items, 600),
            nn.BatchNorm1d(600),  # Add batch norm
            nn.ReLU(),
            nn.Linear(600, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Decoder with more capacity
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(),
            nn.Linear(600, num_items),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

class MovieLensDataset(Dataset):
    def __init__(self, data, num_items):
        self.data = data
        self.num_items = num_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user_ratings = self.data[index]
        user_input = torch.zeros(self.num_items)
        indices = user_ratings[:, 0].astype(int)
        values = user_ratings[:, 1]
        if np.any(indices >= self.num_items):
            raise ValueError("An index is out of bounds.")
        user_input[indices] = torch.tensor(values / 5.0, dtype=torch.float32)
        return user_input

def load_data(partition_id, num_partitions):
    #ratings = pd.read_csv(
    #    "~/dev/ml-latest-small/ratings.csv",
    #    sep=",",
    #    usecols=["userId", "movieId", "rating"],
    #)
    ratings = pd.read_csv(
        "~/dev/ml-1m/ratings.dat",
        sep="::",
        engine='python',
        names=["userId", "movieId", "rating", "timestamp"],
        usecols=["userId", "movieId", "rating"],
    )
    unique_movie_ids = ratings['movieId'].unique()
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}
    ratings['movieId'] = ratings['movieId'].map(movie_id_to_index)
    num_items = len(unique_movie_ids)
    user_group = ratings.groupby("userId")
    user_ids = list(user_group.groups.keys())
    total_users = len(user_ids)
    #print(f"total_users: {total_users}")
    #print(f"num_partitions: {num_partitions}")
    partition_size = total_users // num_partitions
    #print(f"partition_size: {partition_size}")
    start_idx = partition_id * partition_size
    end_idx = total_users if partition_id == num_partitions - 1 else start_idx + partition_size
    partition_user_ids = user_ids[start_idx:end_idx]
    partition_data = []
    for user_id in partition_user_ids:
        user_data = user_group.get_group(user_id)[["movieId", "rating"]].values
        partition_data.append(user_data)
    dataset = MovieLensDataset(partition_data, num_items)
    #dataset = MovieLensDataset(partition_data[:100], num_items)  # Use only first 100 users

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32)
    return trainloader, testloader, num_items

def loss_function(recon_x, x, mu, logvar, epoch = 1, num_epochs = 1):
    #print(f"Sample input: {x[0]}")
    #print(f"Sample reconstruction: {recon_x[0]}")
    # Multi-objective loss
    reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
    
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Ranking loss component
    #pos_mask = (x > 0).float()
    #neg_mask = 1 - pos_mask
    
    # Margin ranking loss
    #margin = 0.5
    #diff = recon_x * pos_mask - recon_x * neg_mask
    #ranking_loss = torch.mean(torch.clamp(margin - diff, min=0.0))
    
    beta = min(epoch / (num_epochs/4), 1.0) * 0.1
    return reconstruction_loss + beta * kl_loss #+ 0.1 * ranking_loss
    #return reconstruction_loss # Remove KL divergence and ranking loss for now to debug
def compute_rmse(recon_batch, batch):
    mask = batch > 0
    mse = F.mse_loss(recon_batch[mask], batch[mask], reduction='mean')
    print(f"MSE: {mse}")
    rmse = torch.sqrt(mse)
    print(f"RMSE: {rmse}")
    return rmse.item()

# Used for baseline comparison
def popularity_baseline(trainloader, testloader, top_k):
    # Aggregate all training data
    train_matrix = torch.zeros(trainloader.dataset.dataset.num_items)
    for batch in trainloader:
        train_matrix += torch.sum(batch, dim=0)
    
    # Get top-k most popular items
    top_items = torch.argsort(train_matrix, descending=True)[:top_k]
    
    # Evaluate on test set
    hits, ndcg_scores = [], []
    for batch in testloader:
        for user_ratings in batch:
            relevant_items = user_ratings.nonzero(as_tuple=True)[0]
            
            # Calculate hits
            hits_arr = np.isin(top_items.numpy(), relevant_items.numpy())
            hits.append(float(hits_arr.sum() > 0))
            
            # Calculate NDCG
            dcg = sum([1 / np.log2(i + 2) if top_items[i] in relevant_items else 0 
                      for i in range(top_k)])
            idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_items), top_k))])
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
    
    return {
        'hit_rate': np.mean(hits),
        'ndcg': np.mean(ndcg_scores)
    }
def get_recommendations(model, user_vector, top_k=10, device='cpu', penalty_value = 0.1):
    model.eval()
    with torch.no_grad():
        user_vector = user_vector.to(device)
        recon_vector, _, _ = model(user_vector.unsqueeze(0)) # Get model's predictions
        recon_vector = recon_vector.squeeze(0)  # Remove batch dimension
        print("Recon Vector (before filtering):", recon_vector)
        recon_vector[user_vector.nonzero(as_tuple=True)] -= penalty_value # Downrank items already rated
        _, indices = torch.topk(recon_vector, top_k) # Get top-k items
        
        assert user_vector.device == recon_vector.device
        assert user_vector.shape == recon_vector.shape
        
        return indices.cpu().numpy()
    
def compute_metrics(model, batch, top_k, total_items, device):
    precisions, recalls, hit_rates, ndcgs = [], [], [], []
    all_recommended_items = []
    
    for user_vector in batch:
        #print(f"relevant_items: {relevant_items}, len: {len(relevant_items)}")
        top_k_items = get_recommendations(model, user_vector, top_k, device)
        #print(f"top_k_items: {top_k_items}, len: {len(top_k_items)}")
        
        interacted_items = user_vector.nonzero(as_tuple=True)[0]
        relevant_items = interacted_items.cpu().numpy()
        hits_arr = np.isin(top_k_items, relevant_items)
        hits = hits_arr.sum()
        print(f"hits: {hits:.10f}")
        precision = hits / top_k
        print(f"precision: {precision:.10f}")
        recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
        print(f"recall: {recall:.10f}")
        hit_rate = 1.0 if hits > 0 else 0.0
        print(f"hit_rate: {hit_rate:.10f}")
        
        dcg = sum([1 / np.log2(i + 2) if top_k_items[i] in relevant_items else 0 for i in range(top_k)])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_items), top_k))])
        ndcg = dcg / idcg if idcg > 0 else 0.0
        print(f"ndcg: {ndcg:.10f}")
        precisions.append(precision)
        recalls.append(recall)
        hit_rates.append(hit_rate)
        ndcgs.append(ndcg)
        all_recommended_items.append(top_k_items)
    
    coverage = len(set(np.concatenate(all_recommended_items))) / total_items
    return {
        "precision_at_k": np.mean(precisions),
        "recall_at_k": np.mean(recalls),
        "hit_rate_at_k": np.mean(hit_rates),
        "ndcg_at_k": np.mean(ndcgs),
        "coverage": coverage,
    }

def train(net, trainloader, epochs, learning_rate, device, total_items):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    net.train()
    total_train_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in trainloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = net(batch)
            loss = loss_function(recon_batch, batch, mu, logvar, epoch, epochs)
            if torch.isnan(loss):
                print("NaN loss encountered during training.")
                return {"train_loss": float('nan')}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # Add gradient clipping
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(trainloader)}")
        total_train_loss += epoch_loss
    avg_train_loss = total_train_loss / (len(trainloader.dataset) * epochs)
    return {"train_loss": float(avg_train_loss)}

def test(net, testloader, device, top_k=10, total_items=None):
    net.to(device)
    net.eval()
    total_loss = 0.0
    total_rmse = 0.0
    all_metrics = []
    with torch.no_grad():
        for batch in testloader:
            batch = batch.to(device)
            recon_batch, mu, logvar = net(batch)
            loss = loss_function(recon_batch, batch, mu, logvar)
            if torch.isnan(loss):
                print("NaN loss encountered during evaluation.")
                return {"val_loss": float('nan')}
            total_loss += loss.item()
            total_rmse += compute_rmse(recon_batch, batch) * batch.size(0)
            metrics = compute_metrics(net, batch, top_k, total_items, device)
            all_metrics.append(metrics)
    avg_loss = total_loss / len(testloader.dataset)
    avg_rmse = total_rmse / len(testloader.dataset)
    aggregated_metrics = {
        "val_loss": avg_loss,
        "rmse": avg_rmse,
        "precision_at_k": np.mean([m["precision_at_k"] for m in all_metrics]),
        "recall_at_k": np.mean([m["recall_at_k"] for m in all_metrics]),
        "hit_rate_at_k": np.mean([m["hit_rate_at_k"] for m in all_metrics]),
        "ndcg_at_k": np.mean([m["ndcg_at_k"] for m in all_metrics]),
        "coverage": np.mean([m["coverage"] for m in all_metrics]),
    }
    return aggregated_metrics

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def analyze_latent_space(model, dataloader, device):
    model.eval()
    latent_vectors = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            mu, logvar = model.encode(batch)
            latent_vectors.append(mu.cpu().numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    
    mean = np.mean(latent_vectors, axis=0)
    std = np.std(latent_vectors, axis=0)
    
    print(f"Latent space mean: {mean.mean():.4f}")
    print(f"Latent space std: {std.mean():.4f}")
    
    active_dims = np.sum(std > 0.01)
    print(f"Number of active dimensions: {active_dims} / {len(std)}")

def interpolate_users(model, user1, user2, steps=10, device='cpu'):
    model.eval()
    with torch.no_grad():
        user1 = user1.to(device)
        user2 = user2.to(device)
        mu1, _ = model.encode(user1.unsqueeze(0))
        mu2, _ = model.encode(user2.unsqueeze(0))
        
        for alpha in np.linspace(0, 1, steps):
            interpolated = alpha * mu1 + (1 - alpha) * mu2
            decoded = model.decode(interpolated)
            top_k = torch.topk(decoded.squeeze(), 10).indices.cpu().numpy()
            print(f"Alpha: {alpha:.2f}, Top 10 recommendations: {top_k}")

# Call this function after training
#analyze_latent_space(net, testloader, device)

# Example usage
#user1 = next(iter(testloader))[0]
#user2 = next(iter(testloader))[1]
#interpolate_users(net, user1, user2, device=device)

def visualize_latent_space(model, dataloader, device):
    model.eval()
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            latent_vectors.append(mu.cpu().numpy())
            labels.extend(batch.sum(dim=1).cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

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

# Call this function after training
#visualize_latent_space(net, testloader, device)

