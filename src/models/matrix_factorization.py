import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, n_factors=100):
        super().__init__()
        self.user_factors = nn.Embedding(num_users, n_factors)
        self.item_factors = nn.Embedding(num_items, n_factors)
        
        # Initialize embeddings with small random values
        self.user_factors.weight.data.normal_(0, 0.01)
        self.item_factors.weight.data.normal_(0, 0.01)
        
        # Optional: Add biases
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, user_ids, item_ids):
        # Get embeddings
        user_embeds = self.user_factors(user_ids)
        item_embeds = self.item_factors(item_ids)
        
        # Get biases
        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()
        
        # Compute dot product
        dot = (user_embeds * item_embeds).sum(dim=1)
        
        # Add biases
        rating = dot + user_bias + item_bias + self.global_bias
        
        return rating