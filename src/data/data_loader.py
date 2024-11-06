import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from .dataset import MovieLensDataset

def load_data(partition_id, num_partitions, mode='vae'):
    """Load and partition MovieLens dataset."""
    ratings = load_ratings()
    ratings = preprocess_ids(ratings)
    
    num_items = len(ratings['movieId'].unique())
    num_users = len(ratings['userId'].unique())
    
    partition_ratings = create_partition(
        ratings, partition_id, num_partitions
    )
    
    train_loader, test_loader = create_data_loaders(
        partition_ratings, num_items, mode
    )
    
    return train_loader, test_loader, num_items, num_users

def load_ratings():
    """Load ratings from file."""
    return pd.read_csv(
        "~/dev/ml-1m/ratings.dat",
        sep="::",
        engine='python',
        names=["userId", "movieId", "rating", "timestamp"],
        usecols=["userId", "movieId", "rating"],
    )

def preprocess_ids(ratings):
    """Map user and movie IDs to consecutive indices."""
    # Map movie IDs
    unique_movie_ids = ratings['movieId'].unique()
    movie_id_map = {mid: idx for idx, mid in enumerate(unique_movie_ids)}
    ratings['movieId'] = ratings['movieId'].map(movie_id_map)
    
    # Map user IDs
    unique_user_ids = ratings['userId'].unique()
    user_id_map = {uid: idx for idx, uid in enumerate(unique_user_ids)}
    ratings['userId'] = ratings['userId'].map(user_id_map)
    
    return ratings

def create_partition(ratings, partition_id, num_partitions):
    """Create a partition of the ratings data."""
    unique_user_ids = ratings['userId'].unique()
    total_users = len(unique_user_ids)
    partition_size = total_users // num_partitions
    
    start_idx = partition_id * partition_size
    end_idx = (total_users if partition_id == num_partitions - 1 
              else start_idx + partition_size)
    
    partition_user_ids = unique_user_ids[start_idx:end_idx]
    return ratings[ratings['userId'].isin(partition_user_ids)]

def create_data_loaders(ratings, num_items, mode):
    """Create train and test data loaders."""
    dataset = MovieLensDataset(ratings, num_items, mode)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    batch_size = 32 if mode == 'vae' else 1024
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )
    
    return train_loader, test_loader