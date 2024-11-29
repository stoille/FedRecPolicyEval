import torch
import torch.nn.functional as F
import numpy as np

class PreferenceEvolution:
    '''
    Preference evolution class for tracking user preferences during training.
    Based on the paper (Preference Amplification in Recommender Systems) [https://research.facebook.com/publications/preference-amplification-in-recommender-systems/]
    '''
    def __init__(self, initial_preferences, beta, gamma, learning_rate_schedule='constant'):
        self.ut = initial_preferences
        self.beta = beta
        self.gamma = gamma
        self.history = {
            'ut_norm': [],
            'likable_prob': [],
            'nonlikable_prob': [],
            'correlated_mass': []
        }
        self.t = 0
        self.schedule = learning_rate_schedule
        
    def update_preferences(self, items, scores):
        # Print shapes for debugging
        #print(f"Items shape: {items.shape}, Scores shape: {scores.shape}, ut shape: {self.ut.shape}")
        
        # Ensure scores is 1D
        if len(scores.shape) > 1:
            scores = scores.mean(dim=1)  # or .sum(dim=1) depending on your needs
        
        # Update ut based on equation (2) from paper
        eta_t = self._get_learning_rate()
        
        # Compute preference update
        score_weights = torch.softmax(self.beta * scores, dim=0)
        score_weighted_items = torch.sum(score_weights.unsqueeze(1) * items, dim=0)
        
        # Update preferences with learning rate schedule
        self.ut = self.ut + eta_t * self.gamma * score_weighted_items
        
        # Track metrics
        self._update_history(items, scores)
        self.t += 1
        
    def _get_learning_rate(self):
        t_adj = self.t // 20  # Adjust every 20 iterations
        if self.schedule == 'constant':
            return 1.0
        elif self.schedule == 'decay':
            return 1.0 / (1.0 + t_adj)
        elif self.schedule == 'sqrt_decay':
            return 1.0 / (1.0 + np.sqrt(t_adj))
        else:  # feature-specific
            return 1.0 / torch.abs(self.ut)
            
    def _update_history(self, items, scores):
        with torch.no_grad():
            # Track ut norm
            self.history['ut_norm'].append(torch.norm(self.ut).item())
            
            # Track probabilities for likable/non-likable items
            item_scores = torch.matmul(items.view(items.size(0), -1), self.ut.view(-1, 1))
            likable_mask = item_scores.squeeze() > 0
            
            probs = torch.softmax(self.beta * scores, dim=0)
            self.history['likable_prob'].append(probs[likable_mask].mean().item())
            self.history['nonlikable_prob'].append(probs[~likable_mask].mean().item())
            
            # Track correlated items probability mass
            cos_sim = F.cosine_similarity(items, self.ut.unsqueeze(0))
            top_k = int(0.05 * len(items))  # Top 5%
            _, indices = torch.topk(cos_sim, top_k)
            self.history['correlated_mass'].append(probs[indices].sum().item())