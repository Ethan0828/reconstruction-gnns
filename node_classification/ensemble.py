"""
Ensemble methods for combining baseline GCN and reconstruction-based GCN.

This module provides several ensemble strategies to combine predictions from:
1. Baseline GCN (standard approach on full graph)
2. Reconstruction GCN (DECK approach with k-hop neighborhoods)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


class AverageEnsemble(nn.Module):
    """
    Simple average ensemble of two models.
    Output = (model1_output + model2_output) / 2
    """
    def __init__(self):
        super(AverageEnsemble, self).__init__()

    def forward(self, logits1, logits2):
        """
        Args:
            logits1: Logits from baseline model [num_nodes, num_classes]
            logits2: Logits from reconstruction model [num_nodes, num_classes]

        Returns:
            Ensemble logits [num_nodes, num_classes]
        """
        return (logits1 + logits2) / 2.0


class WeightedAverageEnsemble(nn.Module):
    """
    Weighted average ensemble with learnable weights.
    Output = weight1 * model1_output + weight2 * model2_output
    where weight1 + weight2 = 1
    """
    def __init__(self, init_weight1=0.5):
        super(WeightedAverageEnsemble, self).__init__()
        # Use sigmoid to ensure weights sum to 1
        self.weight_logit = nn.Parameter(torch.tensor([init_weight1]))

    def forward(self, logits1, logits2):
        """
        Args:
            logits1: Logits from baseline model [num_nodes, num_classes]
            logits2: Logits from reconstruction model [num_nodes, num_classes]

        Returns:
            Ensemble logits [num_nodes, num_classes]
        """
        weight1 = torch.sigmoid(self.weight_logit)
        weight2 = 1.0 - weight1
        return weight1 * logits1 + weight2 * logits2

    def get_weights(self):
        """Return current ensemble weights."""
        weight1 = torch.sigmoid(self.weight_logit).item()
        return weight1, 1.0 - weight1


class ConcatenationEnsemble(nn.Module):
    """
    Concatenate outputs from both models and pass through MLP.
    This allows the model to learn non-linear combinations.
    """
    def __init__(self, num_classes, hidden_size=128, dropout=0.5):
        super(ConcatenationEnsemble, self).__init__()

        # Input is concatenation of two logit vectors
        self.fc1 = Linear(num_classes * 2, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size)
        self.fc3 = Linear(hidden_size, num_classes)
        self.dropout = dropout

    def forward(self, logits1, logits2):
        """
        Args:
            logits1: Logits from baseline model [num_nodes, num_classes]
            logits2: Logits from reconstruction model [num_nodes, num_classes]

        Returns:
            Ensemble logits [num_nodes, num_classes]
        """
        # Concatenate logits
        x = torch.cat([logits1, logits2], dim=-1)

        # Pass through MLP
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x


class FeatureConcatenationEnsemble(nn.Module):
    """
    Concatenate features (before final layer) from both models.
    This allows combining representations rather than just predictions.
    """
    def __init__(self, hidden_size1, hidden_size2, num_classes, dropout=0.5):
        super(FeatureConcatenationEnsemble, self).__init__()

        # Combine features from both models
        combined_size = hidden_size1 + hidden_size2

        self.fc1 = Linear(combined_size, combined_size // 2)
        self.fc2 = Linear(combined_size // 2, num_classes)
        self.dropout = dropout

    def forward(self, features1, features2):
        """
        Args:
            features1: Features from baseline model [num_nodes, hidden_size1]
            features2: Features from reconstruction model [num_nodes, hidden_size2]

        Returns:
            Ensemble logits [num_nodes, num_classes]
        """
        # Concatenate features
        x = torch.cat([features1, features2], dim=-1)

        # Pass through MLP
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


class AttentionEnsemble(nn.Module):
    """
    Use attention mechanism to combine predictions from both models.
    The model learns to attend to different models based on the input.
    """
    def __init__(self, num_classes, hidden_size=64):
        super(AttentionEnsemble, self).__init__()

        # Attention network
        self.query = Linear(num_classes * 2, hidden_size)
        self.key1 = Linear(num_classes, hidden_size)
        self.key2 = Linear(num_classes, hidden_size)

    def forward(self, logits1, logits2):
        """
        Args:
            logits1: Logits from baseline model [num_nodes, num_classes]
            logits2: Logits from reconstruction model [num_nodes, num_classes]

        Returns:
            Ensemble logits [num_nodes, num_classes]
        """
        # Concatenate for query
        concat = torch.cat([logits1, logits2], dim=-1)
        query = F.relu(self.query(concat))  # [num_nodes, hidden_size]

        # Compute keys
        key1 = F.relu(self.key1(logits1))  # [num_nodes, hidden_size]
        key2 = F.relu(self.key2(logits2))  # [num_nodes, hidden_size]

        # Compute attention scores
        score1 = (query * key1).sum(dim=-1, keepdim=True)  # [num_nodes, 1]
        score2 = (query * key2).sum(dim=-1, keepdim=True)  # [num_nodes, 1]

        # Softmax to get attention weights
        scores = torch.cat([score1, score2], dim=-1)  # [num_nodes, 2]
        weights = F.softmax(scores, dim=-1)  # [num_nodes, 2]

        # Weighted combination
        output = weights[:, 0:1] * logits1 + weights[:, 1:2] * logits2

        return output


class StackingEnsemble(nn.Module):
    """
    Stacking ensemble: Use predictions from both models as features
    for a meta-learner (second-level classifier).
    """
    def __init__(self, num_classes, meta_hidden_size=64, dropout=0.3):
        super(StackingEnsemble, self).__init__()

        # Meta-learner: takes softmax probabilities from both models
        input_size = num_classes * 2

        self.meta_learner = nn.Sequential(
            Linear(input_size, meta_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(meta_hidden_size, meta_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(meta_hidden_size // 2, num_classes)
        )

    def forward(self, logits1, logits2):
        """
        Args:
            logits1: Logits from baseline model [num_nodes, num_classes]
            logits2: Logits from reconstruction model [num_nodes, num_classes]

        Returns:
            Ensemble logits [num_nodes, num_classes]
        """
        # Convert to probabilities
        prob1 = F.softmax(logits1, dim=-1)
        prob2 = F.softmax(logits2, dim=-1)

        # Concatenate probabilities
        meta_features = torch.cat([prob1, prob2], dim=-1)

        # Meta-learner prediction
        output = self.meta_learner(meta_features)

        return output


def get_ensemble_model(ensemble_type, **kwargs):
    """
    Factory function to create ensemble models.

    Args:
        ensemble_type: Type of ensemble ('average', 'weighted', 'concat',
                       'feature_concat', 'attention', 'stacking')
        **kwargs: Additional arguments for specific ensemble types

    Returns:
        Ensemble model instance
    """
    ensemble_type = ensemble_type.lower()

    if ensemble_type == 'average':
        return AverageEnsemble()

    elif ensemble_type == 'weighted':
        init_weight = kwargs.get('init_weight1', 0.5)
        return WeightedAverageEnsemble(init_weight1=init_weight)

    elif ensemble_type == 'concat':
        num_classes = kwargs.get('num_classes')
        hidden_size = kwargs.get('hidden_size', 128)
        dropout = kwargs.get('dropout', 0.5)
        return ConcatenationEnsemble(num_classes, hidden_size, dropout)

    elif ensemble_type == 'feature_concat':
        hidden_size1 = kwargs.get('hidden_size1')
        hidden_size2 = kwargs.get('hidden_size2')
        num_classes = kwargs.get('num_classes')
        dropout = kwargs.get('dropout', 0.5)
        return FeatureConcatenationEnsemble(
            hidden_size1, hidden_size2, num_classes, dropout
        )

    elif ensemble_type == 'attention':
        num_classes = kwargs.get('num_classes')
        hidden_size = kwargs.get('hidden_size', 64)
        return AttentionEnsemble(num_classes, hidden_size)

    elif ensemble_type == 'stacking':
        num_classes = kwargs.get('num_classes')
        meta_hidden_size = kwargs.get('meta_hidden_size', 64)
        dropout = kwargs.get('dropout', 0.3)
        return StackingEnsemble(num_classes, meta_hidden_size, dropout)

    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")
