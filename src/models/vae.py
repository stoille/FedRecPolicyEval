import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, num_items, hidden_dim=256, latent_dim=128, dropout_rate=0.3, beta=1.0):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_items),
            nn.Sigmoid()
        )
        
        self.beta = beta

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=0.1):
        """Calculate VAE loss with beta-VAE formulation."""
        # Reconstruction loss (binary cross entropy)
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence with annealing
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Add L2 regularization
        l2_reg = 0.01 * (mu.pow(2).mean() + logvar.exp().mean())
        
        # Add ranking loss component
        ranking_loss = 0.0
        pos_mask = (x > x.mean())  # Positive interactions
        neg_mask = (x <= x.mean())  # Negative interactions
        
        if pos_mask.any() and neg_mask.any():
            # Pre-compute means in one go
            means = torch.zeros(2, device=recon_x.device)
            means[0] = recon_x[pos_mask].mean()  # positive mean
            means[1] = recon_x[neg_mask].mean()  # negative mean
            
            # Simple margin loss with vectorized operation
            margin = 0.1
            ranking_loss = torch.relu(means[1] - means[0] + margin)
        
        return BCE + beta * KLD + 0.1 * ranking_loss + l2_reg