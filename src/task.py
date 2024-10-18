"""FedRecPolicyEval: A Flower / PyTorch package for evaluating recommender system policies."""

from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
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
        user_input = torch.zeros(self.num_items)  # Initialize a tensor of zeros

        # Get indices and values
        indices = user_ratings[:, 0].astype(int)
        values = user_ratings[:, 1]

        # Ensure indices are within the correct range
        if np.any(indices >= self.num_items):
            raise ValueError("An index is out of bounds.")

        # Assign ratings to corresponding indices
        user_input[indices] = torch.tensor(values / 5.0, dtype=torch.float32)  # Normalize ratings to [0, 1]

        return user_input

def load_data(partition_id, num_partitions):
    """Load partitioned MovieLens data."""
    # Load MovieLens dataset
    ratings = pd.read_csv(
        "~/dev/ml-latest-small/ratings.csv",
        sep=",",  # MovieLens 'ml-latest-small' uses commas as separators
        usecols=["userId", "movieId", "rating"],  # Only necessary columns
    )

    # Map movie IDs to indices
    unique_movie_ids = ratings['movieId'].unique()
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}
    ratings['movieId'] = ratings['movieId'].map(movie_id_to_index)

    num_items = len(unique_movie_ids)

    # Create user-item interactions
    user_group = ratings.groupby("userId")
    user_ids = list(user_group.groups.keys())

    # Partition users among clients
    total_users = len(user_ids)
    partition_size = total_users // num_partitions
    start_idx = partition_id * partition_size
    end_idx = total_users if partition_id == num_partitions - 1 else start_idx + partition_size
    partition_user_ids = user_ids[start_idx:end_idx]

    # Prepare data for the partition
    partition_data = []
    for user_id in partition_user_ids:
        user_data = user_group.get_group(user_id)[["movieId", "rating"]].values
        partition_data.append(user_data)

    dataset = MovieLensDataset(partition_data, num_items)

    # Split into train and test datasets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32)

    return trainloader, testloader, num_items

def loss_function(recon_x, x, mu, logvar, eps=1e-8):
    """VAE loss function."""
    BCE = F.binary_cross_entropy(recon_x.clamp(min=eps, max=1-eps), x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(net, trainloader, epochs, learning_rate, device):
    """Train the network on the training set."""
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
            
            # Check for NaN loss
            if torch.isnan(loss):
                print("NaN loss encountered during training.")
                print("Batch data:", batch)
                return float('nan')
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / (len(trainloader) * epochs)

    return {
        "train_loss": float(avg_train_loss),
    }

def test(net, testloader, device):
    """Evaluate the network on the test set."""
    net.to(device)
    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in testloader:
            batch = batch.to(device)
            recon_batch, mu, logvar = net(batch)
            loss = loss_function(recon_batch, batch, mu, logvar)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print("NaN loss encountered during evaluation.")
                print("Batch data:", batch)
                return float('nan')
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(testloader)
    return avg_loss

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
