import pytest
import torch
from src.models.vae import VAE
from src.models.matrix_factorization import MatrixFactorization

@pytest.fixture
def vae_model():
    return VAE(num_items=100, latent_dim=20)

@pytest.fixture
def mf_model():
    return MatrixFactorization(num_users=50, num_items=100, n_factors=20)

class TestVAE:
    def test_vae_initialization(self, vae_model):
        assert isinstance(vae_model, VAE)
        assert vae_model.fc_mu.out_features == 20
        assert vae_model.fc_logvar.out_features == 20

    def test_vae_forward(self, vae_model):
        x = torch.randn(32, 100)
        recon_x, mu, logvar = vae_model(x)
        assert recon_x.shape == x.shape
        assert mu.shape == (32, 20)
        assert logvar.shape == (32, 20)

    def test_vae_loss(self, vae_model):
        x = torch.rand(32, 100)
        recon_x = torch.sigmoid(torch.randn(32, 100))
        mu = torch.randn(32, 20)
        logvar = torch.randn(32, 20)
        loss = vae_model.loss_function(recon_x, x, mu, logvar)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

class TestMatrixFactorization:
    def test_mf_initialization(self, mf_model):
        assert isinstance(mf_model, MatrixFactorization)
        assert mf_model.user_factors.num_embeddings == 50
        assert mf_model.item_factors.num_embeddings == 100

    def test_mf_forward(self, mf_model):
        users = torch.randint(0, 50, (32,))
        items = torch.randint(0, 100, (32,))
        predictions = mf_model(users, items)
        assert predictions.shape == (32,)
