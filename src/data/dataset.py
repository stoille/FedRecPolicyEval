import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self, ratings_df, num_items, mode='vae'):
        self.mode = mode
        self.num_items = num_items
        
        if mode == 'vae':
            user_group = ratings_df.groupby("userId")
            self.data = []
            for _, group in user_group:
                user_data = group[["movieId", "rating"]].values
                self.data.append(user_data)
        else:
            self.users = ratings_df['userId'].values
            self.items = ratings_df['movieId'].values
            self.ratings = ratings_df['rating'].values / 5.0

    def __len__(self):
        return len(self.data) if self.mode == 'vae' else len(self.users)

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