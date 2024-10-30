"""FedRecPolicyEval: A Flower / PyTorch package for evaluating recommender system policies."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

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
    def __init__(self, ratings_df, num_items, mode='vae'):
        self.mode = mode
        self.num_items = num_items
        
        if mode == 'vae':
            # Group ratings by user and create sparse vectors
            user_group = ratings_df.groupby("userId")
            self.data = []
            for _, group in user_group:
                user_data = group[["movieId", "rating"]].values
                self.data.append(user_data)
        else:  # matrix factorization mode
            self.users = ratings_df['userId'].values
            self.items = ratings_df['movieId'].values
            self.ratings = ratings_df['rating'].values / 5.0  # Normalize ratings

    def __len__(self):
        if self.mode == 'vae':
            return len(self.data)
        return len(self.users)

    def __getitem__(self, index):
        if self.mode == 'vae':
            user_ratings = self.data[index]
            user_input = torch.zeros(self.num_items)
            indices = user_ratings[:, 0].astype(int)
            values = user_ratings[:, 1]
            user_input[indices] = torch.tensor(values / 5.0, dtype=torch.float32)
            return user_input
        else:
            return (
                torch.tensor(self.users[index], dtype=torch.long),
                torch.tensor(self.items[index], dtype=torch.long),
                torch.tensor(self.ratings[index], dtype=torch.float32)
            )

def load_data(partition_id, num_partitions, mode='vae'):
    # Load ratings
    ratings = pd.read_csv(
        "~/dev/ml-1m/ratings.dat",
        sep="::",
        engine='python',
        names=["userId", "movieId", "rating", "timestamp"],
        usecols=["userId", "movieId", "rating"],
    )
    
    # Map movie IDs to consecutive indices
    unique_movie_ids = ratings['movieId'].unique()
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}
    ratings['movieId'] = ratings['movieId'].map(movie_id_to_index)
    
    # Map user IDs to consecutive indices
    unique_user_ids = ratings['userId'].unique()
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
    ratings['userId'] = ratings['userId'].map(user_id_to_index)
    
    num_items = len(unique_movie_ids)
    num_users = len(unique_user_ids)
    
    # Partition data
    total_users = len(unique_user_ids)
    partition_size = total_users // num_partitions
    start_idx = partition_id * partition_size
    end_idx = total_users if partition_id == num_partitions - 1 else start_idx + partition_size
    
    # Filter ratings for partition
    partition_user_ids = unique_user_ids[start_idx:end_idx]
    partition_ratings = ratings[ratings['userId'].isin(partition_user_ids)]
    
    # Create dataset
    dataset = MovieLensDataset(partition_ratings, num_items, mode)
    
    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    batch_size = 32 if mode == 'vae' else 1024  # Larger batch size for MF
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloader, testloader, num_items, num_users

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
        recon_vector[user_vector.nonzero(as_tuple=True)] -= penalty_value # Downrank items already rated
        _, indices = torch.topk(recon_vector, top_k) # Get top-k items
        
        assert user_vector.device == recon_vector.device
        assert user_vector.shape == recon_vector.shape
        
        return indices.cpu().numpy()
    
def compute_metrics(model, batch, top_k, total_items, device):
    precisions, recalls, ndcgs = [], [], []
    all_recommended_items = []
    roc_aucs = []  # List to store ROC AUC scores

    for user_vector in batch:
        top_k_items = get_recommendations(model, user_vector, top_k, device)
        
        interacted_items = user_vector.nonzero(as_tuple=True)[0]
        relevant_items = interacted_items.cpu().numpy()
        hits_arr = np.isin(top_k_items, relevant_items)
        hits = hits_arr.sum()
        
        # Prepare binary labels and scores for ROC AUC
        actual = np.zeros(total_items)
        actual[relevant_items] = 1
        scores = np.zeros(total_items)
        scores[top_k_items] = 1  # Assign scores of 1 for recommended items
        
        # Compute ROC AUC
        if len(np.unique(actual)) > 1:  # ROC AUC is undefined if there's no variation in actual
            roc_auc = roc_auc_score(actual, scores)
            roc_aucs.append(roc_auc)
        
        precision = hits / top_k
        recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
        ndcg = sum([1 / np.log2(i + 2) if top_k_items[i] in relevant_items else 0 for i in range(top_k)])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_items), top_k))])
        ndcg = ndcg / idcg if idcg > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        all_recommended_items.append(top_k_items)
        
        print(f"Precision: {precision:.10f}, Recall: {recall:.10f}, NDCG: {ndcg:.10f}, ROC AUC: {roc_auc:.10f}")
    
    coverage = len(set(np.concatenate(all_recommended_items))) / total_items
    return {
        "precision_at_k": np.mean(precisions),
        "recall_at_k": np.mean(recalls),
        "ndcg_at_k": np.mean(ndcgs),
        "roc_auc": np.mean(roc_aucs),  # Return average ROC AUC
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
        "ndcg_at_k": np.mean([m["ndcg_at_k"] for m in all_metrics]),
        "roc_auc": np.mean([m["roc_auc"] for m in all_metrics]),  # Return average ROC AUC
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
    if not isinstance(model, VAE):
        return  # Skip visualization for non-VAE models
    
    model.eval()
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle different data formats
            if isinstance(batch, torch.Tensor):
                # VAE mode
                user_data = batch
            else:
                # MF mode - skip visualization
                return
                
            user_data = user_data.to(device)
            mu, _ = model.encode(user_data)
            latent_vectors.append(mu.cpu().numpy())
            labels.extend(user_data.sum(dim=1).cpu().numpy())

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

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, n_factors=100):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(num_users, n_factors, sparse=True)
        self.item_factors = nn.Embedding(num_items, n_factors, sparse=True)
        
        # Initialize weights with normal distribution
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)

    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)

def train_mf(model, trainloader, epochs, learning_rate, device):
    model = model.to(device)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user, item, rating in trainloader:
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device)
            
            optimizer.zero_grad()
            prediction = model(user, item)
            l2_reg = torch.norm(model.user_factors(user)) + torch.norm(model.item_factors(item))
            loss = F.mse_loss(prediction, rating) + 1e-5 * l2_reg
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(trainloader):.4f}")
    
    return model.cpu()  # Return CPU model for parameter aggregation

def test_mf(model, testloader, device, top_k=10, total_items=None, penalty_value=0.1):
    # Refactored test function to compute aggregated metrics similar to 'test'
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    all_metrics = []
    all_recommended_items = []
    user_item_dict = {}

    with torch.no_grad():
        # collect metrics and user-item interactions
        for user, item, rating in testloader:
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device)
            prediction = model(user, item)
            loss = F.mse_loss(prediction, rating.view(-1), reduction='mean')
            total_loss += loss.item()
            rmse = torch.sqrt(F.mse_loss(prediction, rating.view(-1), reduction='mean'))
            total_rmse += rmse.item() * user.size(0)

            # Collect user-item interactions
            for u, i in zip(user.cpu().numpy(), item.cpu().numpy()):
                u = int(u)
                i = int(i)
                if u not in user_item_dict:
                    user_item_dict[u] = set()
                user_item_dict[u].add(i)

        # compute recommendations and metrics
        for u in user_item_dict:
            user_tensor = torch.tensor([u], dtype=torch.long, device=device)
            user_embedding = model.user_factors(user_tensor)
            all_items = torch.arange(total_items, dtype=torch.long, device=device)
            item_embeddings = model.item_factors(all_items)
            scores = (user_embedding * item_embeddings).sum(1)
            rated_items = np.array(list(user_item_dict[u]))
            
            # Apply penalty instead of setting to -inf
            #scores[rated_items] -= penalty_value  # Downrank instead of exclude ex: scores[rated_items] = float('-inf')
            
            top_k_items = torch.topk(scores, top_k).indices.cpu().numpy()
            all_recommended_items.extend(top_k_items)
            relevant_items = rated_items
            hits_arr = np.isin(top_k_items, relevant_items)
            hits = hits_arr.sum()
            precision = hits / top_k
            recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
            ndcg = sum([1 / np.log2(i + 2) if top_k_items[i] in relevant_items else 0 for i in range(top_k)])
            idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_items), top_k))])
            ndcg = ndcg / idcg if idcg > 0 else 0.0
            actual = np.zeros(total_items)
            actual[relevant_items] = 1
            predicted_scores = np.zeros(total_items)
            predicted_scores[top_k_items] = scores[top_k_items].detach().cpu().numpy()
            if len(np.unique(actual)) > 1:
                roc_auc = roc_auc_score(actual, predicted_scores)
            else:
                roc_auc = float('nan')
            #print(f"\nUser {u} metrics:")
            #print(f"Top-k items: {top_k_items}")
            #print(f"Relevant items: {relevant_items}")
            #print(f"Hits array: {hits_arr}")
            print(f"Number of hits: {hits}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"NDCG: {ndcg}")
            
            # Add sanity checks
            if hits == 0:
                print("Warning: No hits found!")
                #print(f"Scores shape: {scores.shape}")
                #print(f"Top scores: {scores[top_k_items]}")
            
            if ndcg == 0:
                print("Warning: NDCG is 0!")
                print(f"IDCG: {idcg}")
                print(f"DCG components: {[1 / np.log2(i + 2) if top_k_items[i] in relevant_items else 0 for i in range(top_k)]}")

            all_metrics.append({
                "precision_at_k": precision,
                "recall_at_k": recall,
                "ndcg_at_k": ndcg,
                "roc_auc": roc_auc,
            })

    # Print aggregate statistics
    print("\nAggregate Statistics:")
    print(f"Total users evaluated: {len(user_item_dict)}")
    print(f"Average items per user: {np.mean([len(items) for items in user_item_dict.values()])}")
    print(f"Total recommendations made: {len(all_recommended_items)}")

    aggregated_metrics = {
        "val_loss": total_loss / len(testloader.dataset),
        "rmse": total_rmse / len(testloader.dataset),
        "precision_at_k": np.mean([m["precision_at_k"] for m in all_metrics]),
        "recall_at_k": np.mean([m["recall_at_k"] for m in all_metrics]),
        "ndcg_at_k": np.mean([m["ndcg_at_k"] for m in all_metrics]),
        "roc_auc": np.nanmean([m["roc_auc"] for m in all_metrics]),
        "coverage": len(set(all_recommended_items)) / total_items,
    }
    return aggregated_metrics
