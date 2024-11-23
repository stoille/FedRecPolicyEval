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

def get_recommendations(model, user_vector, user_id, top_k=10, device='cpu', penalty_value=0.1):
    model.eval()
    with torch.no_grad():
        if isinstance(model, MatrixFactorization):
            # Adjust user_id to be zero-indexed since MovieLens dataset is 1-indexed
            adjusted_user_id = user_id - 1 if user_id > 0 else user_id
            user_id = torch.tensor([adjusted_user_id]).to(device)
            item_ids = torch.arange(model.item_factors.num_embeddings).to(device)
            predictions = model(user_id.expand(len(item_ids)), item_ids)
            _, indices = torch.topk(predictions, top_k)
            return indices.cpu().numpy(), predictions.cpu().numpy()
        else:
            user_vector = user_vector.to(device)
            recon_vector, _, _ = model(user_vector.unsqueeze(0))
            recon_vector = recon_vector.squeeze(0)
            _, indices = torch.topk(recon_vector, top_k)
            return indices.cpu().numpy(), recon_vector.cpu().numpy()

def train(net, trainloader, epochs, learning_rate, device, total_items):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    net.train()
    
    total_train_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = train_epoch(net, trainloader, optimizer, epoch, epochs, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(trainloader)}")
        total_train_loss += epoch_loss
        
    return {"train_loss": float(total_train_loss / (len(trainloader.dataset) * epochs))}

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