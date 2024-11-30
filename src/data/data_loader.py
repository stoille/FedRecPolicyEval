import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from .dataset import MovieLensDataset
import logging

logger = logging.getLogger(__name__)

def load_data(model_type: str = 'vae') -> tuple[DataLoader, DataLoader, dict]:
    """Load MovieLens data and create train/test dataloaders."""
    current_dir = Path(__file__).parent
    data_path = current_dir.parent.parent / "ml-1m" / "ratings.dat"
    
    # Load and preprocess ratings
    ratings_df = pd.read_csv(
        data_path,
        sep="::",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python"
    )
    
    # Remap user and item IDs to start from 0 and be continuous
    user_map = {id_: idx for idx, id_ in enumerate(ratings_df['userId'].unique())}
    item_map = {id_: idx for idx, id_ in enumerate(ratings_df['movieId'].unique())}
    
    ratings_df['userId'] = ratings_df['userId'].map(user_map)
    ratings_df['movieId'] = ratings_df['movieId'].map(item_map)
    
    # Get dimensions from data
    dimensions = {
        'num_users': len(user_map),
        'num_items': len(item_map),
        'user_map': user_map
    }
    
    # Split data
    train_df = ratings_df.sample(frac=0.8, random_state=42)
    eval_df = ratings_df.drop(train_df.index)
    
    # Create datasets
    train_dataset = MovieLensDataset(
        ratings_df=train_df,
        num_items=len(item_map),
        mode=model_type
    )
    
    eval_dataset = MovieLensDataset(
        ratings_df=eval_df,
        num_items=len(item_map),
        mode=model_type
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    logger.info("Checking data ranges:")
    for batch in train_loader:
        if model_type == 'mf':
            user_ids, item_ids, _ = batch
            logger.info(f"Train data - Max user_id: {user_ids.max()}, Max item_id: {item_ids.max()}")
        break
    
    return train_loader, eval_loader, dimensions