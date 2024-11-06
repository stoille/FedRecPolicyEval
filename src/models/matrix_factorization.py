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
        return (self.user_factors(user) * self.item_factors(item)).sum(1)