from collections import OrderedDict
import torch
from src.models.matrix_factorization import MatrixFactorization
import logging

logger = logging.getLogger("Metrics")

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_recommendations(model, user_vector, user_id, top_k, device, temperature, negative_penalty):
    #logger.info("Starting get_recommendations")
    model.eval()
    with torch.no_grad():
        if isinstance(model, MatrixFactorization):
            user_id = torch.tensor([user_id]).to(device)
            item_ids = torch.arange(model.item_factors.num_embeddings).to(device)
            user_ids = user_id.repeat(len(item_ids))
            
            predictions = model(user_ids, item_ids)
            
            # Add exploration term based on item embeddings similarity
            item_embeds = model.item_factors.weight
            item_sims = torch.matmul(item_embeds, item_embeds.t())
            item_sims = torch.softmax(item_sims / temperature, dim=1)
            
            # Diversify predictions based on embedding similarities
            diversity_bonus = -negative_penalty * item_sims.mean(dim=1)
            item_scores = predictions + diversity_bonus
            
            # Mask interacted items if user_vector provided
            if user_vector is not None:
                interacted_mask = user_vector > 0
                item_scores[interacted_mask] = float('-inf')
                
                # Add randomness for exploration
                noise = torch.randn_like(item_scores) * (1.0 / (temperature + 1e-8))
                item_scores += noise
            
            # Get top-k items with diversity bonus
            values, indices = torch.topk(item_scores, k=top_k)
            return indices.cpu().numpy(), predictions.cpu().numpy()
        else:  # VAE case
            #logger.info(f"Processing VAE recommendations for user_vector shape: {user_vector.shape}")
            user_vector = user_vector.to(device)
            recon_vector, _, _ = model(user_vector.unsqueeze(0))
            recon_vector = recon_vector.squeeze(0)
            
            # Log reconstruction values
            #logger.info(f"Reconstruction vector stats - min: {recon_vector.min():.4f}, max: {recon_vector.max():.4f}, mean: {recon_vector.mean():.4f}")
            
            _, indices = torch.topk(recon_vector, top_k)
            return indices.cpu().numpy(), recon_vector.cpu().numpy()

def train(net, trainloader, epochs, learning_rate, device, total_items):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    net.train()
    
    total_epoch_train_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = train_epoch(net, trainloader, optimizer, epoch, epochs, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(trainloader)}")
        total_epoch_train_loss += epoch_loss
        
    return {"epoch_train_loss": float(total_epoch_train_loss / (len(trainloader.dataset) * epochs))}

def train_epoch(net, trainloader, optimizer, epoch, epochs, device):
    epoch_loss = 0.0
    for batch in trainloader:
        batch = batch.to(device)
        loss = train_step(net, batch, optimizer, epoch, epochs)
        epoch_loss += loss.item()
    return epoch_loss

def train_step(net, batch, optimizer, epoch, epochs):
    optimizer.zero_grad()
    recon_batch, mu, logvar = net(batch)
    loss = loss_function(recon_batch, batch, mu, logvar, epoch, epochs)
    if torch.isnan(loss):
        raise ValueError("NaN loss encountered during training.")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()
    return loss