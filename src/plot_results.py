import json
import matplotlib.pyplot as plt

# Load history
with open('history.json', 'r') as f:
    history = json.load(f)

# Extract data
rounds = sorted(map(int, history.keys()))
train_losses = [history[str(r)].get('train_loss', None) for r in rounds]
val_losses = [history[str(r)].get('val_loss', None) for r in rounds]

# Remove None values
train_losses = [loss for loss in train_losses if loss is not None]
val_losses = [loss for loss in val_losses if loss is not None]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Rounds')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()