import torch
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PreferenceEvolution:
    '''
    Preference evolution class for tracking user preferences during training.
    Based on the paper (Preference Amplification in Recommender Systems) [https://research.facebook.com/publications/preference-amplification-in-recommender-systems/]
    '''
    def __init__(self, initial_preferences, beta, gamma, learning_rate_schedule):
        self.preferences = initial_preferences
        self.beta = beta
        self.gamma = gamma
        self.learning_rate_schedule = learning_rate_schedule
        self.t = 0  # Initialize iteration counter
        
        # Separate histories for epochs and rounds
        self.epoch_history = {
            'ut_norm': [],
            'likable_prob': [],
            'nonlikable_prob': [],
            'correlated_mass': []
        }
        self.round_history = {
            'ut_norm': [],
            'likable_prob': [],
            'nonlikable_prob': [],
            'correlated_mass': []
        }
        self.current_round_metrics = []
    
    def update_preferences(self, items, scores, is_round=True):
        """Update preferences based on items and scores."""
        self.t += 1
        
        #logger.info(f"Updating preferences - is_round: {is_round}")
        #logger.info(f"Items shape: {items.shape}, Scores shape: {scores.shape}")
        
        # Ensure scores is 1D
        if len(scores.shape) > 1:
            scores = scores.mean(dim=1)
        
        # Update preferences
        eta_t = self._get_learning_rate()
        score_weights = torch.softmax(self.beta * scores, dim=0)
        score_weighted_items = torch.sum(score_weights.unsqueeze(1) * items, dim=0)
        self.preferences = self.preferences + eta_t * self.gamma * score_weighted_items
        
        # Calculate current metrics
        metrics = self._calculate_metrics(items, scores)
        #logger.info(f"Calculated batch metrics: {metrics}")
        
        # Store metrics in appropriate history
        if is_round:
            self.current_round_metrics.append(metrics)
            #logger.info(f"Added to current round metrics. Length: {len(self.current_round_metrics)}")
        else:
            for k, v in metrics.items():
                self.epoch_history[k].append(v)
            
        return metrics
    
    def _get_learning_rate(self):
        t_adj = self.t // 20  # Adjust every 20 iterations
        if self.learning_rate_schedule == 'constant':
            return 1.0
        elif self.learning_rate_schedule == 'decay':
            return 1.0 / (1.0 + t_adj)
        elif self.learning_rate_schedule == 'sqrt_decay':
            return 1.0 / (1.0 + np.sqrt(t_adj))
        else:  # feature-specific
            return 1.0 / torch.abs(self.preferences)
            
    def get_current_metrics(self, items, scores):
        return self._calculate_metrics(items, scores)
    
    def finalize_round(self):
        """Finalize metrics for the current round."""
        #logger.info(f"Finalizing round with {len(self.current_round_metrics)} metrics")
        
        if self.current_round_metrics:
            round_metrics = {
                k: np.mean([m[k] for m in self.current_round_metrics])
                for k in self.current_round_metrics[0].keys()
            }
            
            # Store in round history
            for k, v in round_metrics.items():
                self.round_history[k].append(v)
            
            #logger.info(f"Round metrics calculated: {round_metrics}")
            #logger.info(f"Updated round history: {self.round_history}")
            
            # Clear current round metrics
            self.current_round_metrics = []
            
            return round_metrics
        return None
    
    def _calculate_metrics(self, items, scores):
        # Calculate item scores as dot product between preferences and items
        item_scores = torch.matmul(items, self.preferences.unsqueeze(1))
        
        return {
            'ut_norm': torch.norm(self.preferences).item(),
            'likable_prob': (item_scores > 0).float().mean().item(),  # Based on scores
            'nonlikable_prob': (item_scores < 0).float().mean().item(),  # Based on scores 
            'correlated_mass': self._calculate_correlation_mass(items, scores)
        }
    
    def _calculate_correlation_mass(self, items, scores):
        # Calculate cosine similarity between items and preferences
        item_scores = torch.matmul(items, self.preferences.unsqueeze(1))
        
        # Calculate normalized correlation (cosine similarity)
        items_norm = torch.norm(items, dim=1, keepdim=True)
        pref_norm = torch.norm(self.preferences)
        cos_sim = item_scores / (items_norm * pref_norm + 1e-8)
        
        # Use fixed threshold (e.g. 0.1) to see how correlations evolve
        threshold = 0.1  # We should see this fraction increase over time
        prob_well_correlated = (cos_sim > threshold).float().mean()
        
        # Debug: print distribution occasionally
        if torch.rand(1).item() < 0.01:
            logger.info(f"Cosine similarities: min={cos_sim.min():.3f}, max={cos_sim.max():.3f}, mean={cos_sim.mean():.3f}")
            logger.info(f"Fraction above threshold: {prob_well_correlated:.3f}")
        
        return 1 - prob_well_correlated.item()