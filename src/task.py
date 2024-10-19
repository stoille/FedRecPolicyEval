"""FedRecPolicyEval: A Flower / PyTorch package for evaluating recommender system policies."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from collections import OrderedDict

class VAE(nn.Module):
    def __init__(self, num_items, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_items, 600),
            nn.Tanh(),
            nn.Linear(600, 200),
            nn.Tanh(),
        )
        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_logvar = nn.Linear(200, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 600),
            nn.Tanh(),
            nn.Linear(600, num_items),
            nn.Sigmoid(),
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
    ratings = pd.read_csv(
        "~/dev/ml-latest-small/ratings.csv",
        sep=",",
        usecols=["userId", "movieId", "rating"],
    )
    unique_movie_ids = ratings['movieId'].unique()
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}
    ratings['movieId'] = ratings['movieId'].map(movie_id_to_index)
    num_items = len(unique_movie_ids)
    user_group = ratings.groupby("userId")
    user_ids = list(user_group.groups.keys())
    total_users = len(user_ids)
    partition_size = total_users // num_partitions
    start_idx = partition_id * partition_size
    end_idx = total_users if partition_id == num_partitions - 1 else start_idx + partition_size
    partition_user_ids = user_ids[start_idx:end_idx]
    partition_data = []
    for user_id in partition_user_ids:
        user_data = user_group.get_group(user_id)[["movieId", "rating"]].values
        partition_data.append(user_data)
    dataset = MovieLensDataset(partition_data, num_items)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32)
    return trainloader, testloader, num_items

def loss_function(recon_x, x, mu, logvar, eps=1e-8):
    BCE = F.binary_cross_entropy(
        recon_x.clamp(min=eps, max=1 - eps), x, reduction="sum"
    )
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def compute_rmse(recon_batch, batch):
    mask = batch > 0
    mse = F.mse_loss(recon_batch[mask], batch[mask], reduction='mean')
    rmse = torch.sqrt(mse)
    return rmse.item()

def compute_metrics(recon_batch, batch, k, total_items):
    precisions, recalls, hit_rates, ndcgs = [], [], [], []
    all_recommended_items = []
    num_users = recon_batch.size(0)
    num_items = recon_batch.size(1)
    for user_idx in range(num_users):
        actual = batch[user_idx]
        predicted = recon_batch[user_idx]
        interacted_items = actual.nonzero(as_tuple=True)[0]
        scores = predicted.detach().cpu().numpy()
        scores[interacted_items.cpu().numpy()] = -np.inf
        top_k_indices = np.argpartition(scores, -k)[-k:]
        top_k_items = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
        relevant_items = interacted_items.cpu().numpy()
        hits = np.isin(top_k_items, relevant_items).sum()
        precision = hits / k
        recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
        hit_rate = 1.0 if hits > 0 else 0.0
        dcg = sum(
            [1 / np.log2(i + 2) if top_k_items[i] in relevant_items else 0 for i in range(k)]
        )
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_items), k))])
        ndcg = dcg / idcg if idcg > 0 else 0.0
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
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    total_train_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = net(batch)
            loss = loss_function(recon_batch, batch, mu, logvar)
            if torch.isnan(loss):
                print("NaN loss encountered during training.")
                return {"train_loss": float('nan')}
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
    avg_train_loss = total_train_loss / (len(trainloader.dataset) * epochs)
    return {"train_loss": float(avg_train_loss)}

def test(net, testloader, device, k=10, total_items=None):
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
            metrics = compute_metrics(recon_batch, batch, k, total_items)
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
