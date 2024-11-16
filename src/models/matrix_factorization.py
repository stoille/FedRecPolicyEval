import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, n_factors=100):
        super().__init__()
        self.user_factors = nn.Embedding(num_users, n_factors, sparse=True)
        self.item_factors = nn.Embedding(num_items, n_factors, sparse=True)
        
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)

    def forward(self, user, item):
        # Add dimension checks
        if torch.max(user) >= self.user_factors.num_embeddings:
            raise ValueError(f"User ID {torch.max(user).item()} out of bounds. Max allowed: {self.user_factors.num_embeddings-1}")
        if torch.max(item) >= self.item_factors.num_embeddings:
            raise ValueError(f"Item ID {torch.max(item).item()} out of bounds. Max allowed: {self.item_factors.num_embeddings-1}")
        
        return (self.user_factors(user) * self.item_factors(item)).sum(1)