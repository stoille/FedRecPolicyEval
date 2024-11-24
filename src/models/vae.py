import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, num_items, latent_dim=200):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_items, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(800, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(600, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(600, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, num_items),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

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
        
        return BCE + beta * KLD