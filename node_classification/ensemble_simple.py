"""
Simple ensemble script for combining predictions from pre-trained models.

This script demonstrates how to use different ensemble strategies when you
already have trained baseline and reconstruction models.
"""

import torch
import torch.nn.functional as F
import argparse
import numpy as np

from layers.gnn import NodeGCN, NodeReconstructionGCN
from data_utils import load_dataset, prepare_data
from ensemble import get_ensemble_model


def simple_average_ensemble(logits1, logits2):
    """
    Simplest ensemble: average predictions.

    Args:
        logits1: Predictions from model 1 [num_nodes, num_classes]
        logits2: Predictions from model 2 [num_nodes, num_classes]

    Returns:
        Ensemble predictions [num_nodes, num_classes]
    """
    return (logits1 + logits2) / 2.0


def weighted_ensemble(logits1, logits2, weight1=0.5):
    """
    Weighted ensemble with fixed weights.

    Args:
        logits1: Predictions from model 1
        logits2: Predictions from model 2
        weight1: Weight for model 1 (weight2 = 1 - weight1)

    Returns:
        Ensemble predictions
    """
    weight2 = 1.0 - weight1
    return weight1 * logits1 + weight2 * logits2


def voting_ensemble(logits1, logits2):
    """
    Voting ensemble: take class with highest combined probability.

    Args:
        logits1: Predictions from model 1
        logits2: Predictions from model 2

    Returns:
        Ensemble predictions (class indices)
    """
    # Convert to probabilities
    prob1 = F.softmax(logits1, dim=-1)
    prob2 = F.softmax(logits2, dim=-1)

    # Sum probabilities and take argmax
    combined_prob = prob1 + prob2
    return combined_prob.argmax(dim=1)


def confidence_based_ensemble(logits1, logits2, confidence_threshold=0.8):
    """
    Use model 1 if confident, otherwise use ensemble.

    Args:
        logits1: Predictions from model 1
        logits2: Predictions from model 2
        confidence_threshold: Threshold for using model 1 alone

    Returns:
        Ensemble predictions
    """
    prob1 = F.softmax(logits1, dim=-1)
    max_prob1, _ = prob1.max(dim=-1, keepdim=True)

    # Use model 1 if confident, otherwise average
    ensemble = (logits1 + logits2) / 2.0

    # Create mask for confident predictions
    confident = (max_prob1 >= confidence_threshold).float()

    return confident * logits1 + (1 - confident) * ensemble


@torch.no_grad()
def evaluate_ensemble(logits1, logits2, labels, mask, ensemble_fn, **kwargs):
    """
    Evaluate ensemble performance.

    Args:
        logits1: Predictions from model 1
        logits2: Predictions from model 2
        labels: Ground truth labels
        mask: Boolean mask for nodes to evaluate
        ensemble_fn: Ensemble function
        **kwargs: Additional arguments for ensemble function

    Returns:
        Accuracy
    """
    if 'weight1' in kwargs:
        ensemble_out = ensemble_fn(logits1, logits2, weight1=kwargs['weight1'])
    elif 'confidence_threshold' in kwargs:
        ensemble_out = ensemble_fn(logits1, logits2, confidence_threshold=kwargs['confidence_threshold'])
    else:
        ensemble_out = ensemble_fn(logits1, logits2)

    if ensemble_out.dim() == 1:
        # Already predicted classes
        pred = ensemble_out
    else:
        # Need to take argmax
        pred = ensemble_out.argmax(dim=1)

    correct = (pred[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    accuracy = correct / total

    return accuracy


def grid_search_weights(logits1, logits2, labels, val_mask, test_mask):
    """
    Grid search to find best ensemble weights.

    Args:
        logits1: Predictions from model 1
        logits2: Predictions from model 2
        labels: Ground truth labels
        val_mask: Validation mask
        test_mask: Test mask

    Returns:
        Best weight, validation accuracy, test accuracy
    """
    print("\nGrid search for best ensemble weight...")
    print("-" * 50)

    best_weight = 0.5
    best_val_acc = 0.0
    best_test_acc = 0.0

    weights = np.arange(0.0, 1.05, 0.05)

    for w in weights:
        val_acc = evaluate_ensemble(
            logits1, logits2, labels, val_mask,
            weighted_ensemble, weight1=w
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weight = w
            # Evaluate on test with best validation weight
            best_test_acc = evaluate_ensemble(
                logits1, logits2, labels, test_mask,
                weighted_ensemble, weight1=w
            )

    print(f"Best weight: {best_weight:.2f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {best_test_acc:.4f}")
    print("-" * 50)

    return best_weight, best_val_acc, best_test_acc


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}\n")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset, data = load_dataset(args.dataset, root=args.data_root)
    data = prepare_data(data, use_fixed_split=True)
    data = data.to(device)

    print(f"Nodes: {data.num_nodes}, Classes: {dataset.num_classes}\n")

    # Create models (in practice, you would load pre-trained models)
    print("Creating models...")
    baseline_model = NodeGCN(
        node_size=data.num_node_features,
        edge_size=data.edge_attr.size(1),
        hidden_size=256,
        out_size=dataset.num_classes,
        num_layers=2
    ).to(device)

    # For demonstration, we'll use random predictions
    # In practice, you would: baseline_model.load_state_dict(torch.load('baseline.pt'))

    baseline_model.eval()

    # Get predictions from both models
    print("Getting predictions...\n")
    with torch.no_grad():
        logits_baseline = baseline_model(data.x, data.edge_index, data.edge_attr)

        # For demonstration, create synthetic "reconstruction" predictions
        # In practice, you would get these from your trained reconstruction model
        # Create slightly different predictions for demonstration
        noise = torch.randn_like(logits_baseline) * 0.1
        logits_recon = logits_baseline + noise

    # Evaluate individual models
    print("="*70)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*70)

    baseline_model.eval()
    with torch.no_grad():
        pred = logits_baseline.argmax(dim=1)
        baseline_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        print(f"Baseline Model Test Accuracy: {baseline_acc:.4f}")

        pred = logits_recon.argmax(dim=1)
        recon_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        print(f"Reconstruction Model Test Accuracy: {recon_acc:.4f}\n")

    # Test different ensemble strategies
    print("="*70)
    print("ENSEMBLE STRATEGIES")
    print("="*70)

    # 1. Simple Average
    print("\n1. Simple Average Ensemble")
    avg_acc = evaluate_ensemble(
        logits_baseline, logits_recon, data.y, data.test_mask,
        simple_average_ensemble
    )
    print(f"   Test Accuracy: {avg_acc:.4f}")
    print(f"   Improvement: {(avg_acc - baseline_acc)*100:+.2f}%")

    # 2. Weighted Ensemble (grid search for best weight)
    print("\n2. Weighted Ensemble (Grid Search)")
    best_weight, val_acc, test_acc = grid_search_weights(
        logits_baseline, logits_recon, data.y,
        data.val_mask, data.test_mask
    )
    print(f"   Improvement: {(test_acc - baseline_acc)*100:+.2f}%")

    # 3. Voting Ensemble
    print("\n3. Voting Ensemble")
    voting_acc = evaluate_ensemble(
        logits_baseline, logits_recon, data.y, data.test_mask,
        voting_ensemble
    )
    print(f"   Test Accuracy: {voting_acc:.4f}")
    print(f"   Improvement: {(voting_acc - baseline_acc)*100:+.2f}%")

    # 4. Confidence-based Ensemble
    print("\n4. Confidence-based Ensemble (threshold=0.8)")
    conf_acc = evaluate_ensemble(
        logits_baseline, logits_recon, data.y, data.test_mask,
        confidence_based_ensemble, confidence_threshold=0.8
    )
    print(f"   Test Accuracy: {conf_acc:.4f}")
    print(f"   Improvement: {(conf_acc - baseline_acc)*100:+.2f}%")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    results = [
        ("Baseline", baseline_acc),
        ("Reconstruction", recon_acc),
        ("Simple Average", avg_acc),
        ("Weighted (w={:.2f})".format(best_weight), test_acc),
        ("Voting", voting_acc),
        ("Confidence-based", conf_acc)
    ]

    results.sort(key=lambda x: x[1], reverse=True)

    for i, (name, acc) in enumerate(results, 1):
        marker = "★" if i == 1 else " "
        print(f"{marker} {i}. {name:25s} {acc:.4f}")

    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple ensemble demonstration')

    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'cs', 'physics'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    main(args)
