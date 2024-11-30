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
        return {
            'ut_norm': torch.norm(self.preferences).item(),
            'likable_prob': (self.preferences > 0).float().mean().item(),
            'nonlikable_prob': (self.preferences < 0).float().mean().item(),
            'correlated_mass': self._calculate_correlation_mass(items, scores)
        }
    
    def _calculate_correlation_mass(self, items, scores):
        # Debug shapes
        #logger.debug(f"_calculate_correlation_mass - items shape: {items.shape}, scores shape: {scores.shape}")
        
        # For VAE, scores is a single value per item, so we need to expand it
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(1)  # Add dimension to make it [batch_size, 1]
        
        # Calculate item scores
        item_scores = torch.matmul(items, self.preferences.unsqueeze(1))
        
        # Ensure broadcasting works correctly
        scores = scores.expand_as(item_scores)
        
        return (scores * item_scores).mean().item()