"""
Quick example showing how to use ensemble methods.

This demonstrates the most common use case: combining predictions
from baseline and reconstruction models.
"""

import torch
import torch.nn.functional as F
from data_utils import load_dataset, prepare_data
from ensemble import get_ensemble_model


def ensemble_quick_example():
    """
    Minimal example of using ensemble methods.
    """
    print("="*70)
    print("Ensemble Methods - Quick Example")
    print("="*70)

    # Load data
    dataset, data = load_dataset('cora')
    data = prepare_data(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    print(f"\nDataset: Cora")
    print(f"Nodes: {data.num_nodes}, Classes: {dataset.num_classes}")

    # Simulate predictions from two models
    # In practice, these would come from your trained models
    print("\nGenerating predictions from two models...")

    num_nodes = data.num_nodes
    num_classes = dataset.num_classes

    # Simulate baseline predictions (in practice: baseline_model(...))
    torch.manual_seed(42)
    logits_baseline = torch.randn(num_nodes, num_classes).to(device)

    # Simulate reconstruction predictions (in practice: recon_model(...))
    torch.manual_seed(43)
    logits_recon = torch.randn(num_nodes, num_classes).to(device)

    # 1. Simple Average Ensemble
    print("\n1. Simple Average Ensemble")
    print("-" * 50)
    ensemble_avg = get_ensemble_model('average').to(device)
    logits_avg = ensemble_avg(logits_baseline, logits_recon)
    pred_avg = logits_avg.argmax(dim=1)
    acc_avg = (pred_avg[data.test_mask] == data.y[data.test_mask]).float().mean()
    print(f"   Test Accuracy: {acc_avg:.4f}")
    print(f"   Parameters: 0 (no training needed)")

    # 2. Weighted Average Ensemble
    print("\n2. Weighted Average Ensemble")
    print("-" * 50)
    ensemble_weighted = get_ensemble_model('weighted', init_weight1=0.6).to(device)

    # You can train this or use with fixed weights
    w1, w2 = ensemble_weighted.get_weights()
    print(f"   Initial weights: Baseline={w1:.3f}, Recon={w2:.3f}")

    logits_weighted = ensemble_weighted(logits_baseline, logits_recon)
    pred_weighted = logits_weighted.argmax(dim=1)
    acc_weighted = (pred_weighted[data.test_mask] == data.y[data.test_mask]).float().mean()
    print(f"   Test Accuracy: {acc_weighted:.4f}")
    print(f"   Parameters: 1 (learnable weight)")

    # 3. Concatenation Ensemble
    print("\n3. Concatenation Ensemble (with MLP)")
    print("-" * 50)
    ensemble_concat = get_ensemble_model(
        'concat',
        num_classes=num_classes,
        hidden_size=64,
        dropout=0.5
    ).to(device)

    logits_concat = ensemble_concat(logits_baseline, logits_recon)
    pred_concat = logits_concat.argmax(dim=1)
    acc_concat = (pred_concat[data.test_mask] == data.y[data.test_mask]).float().mean()

    num_params = sum(p.numel() for p in ensemble_concat.parameters())
    print(f"   Test Accuracy: {acc_concat:.4f}")
    print(f"   Parameters: {num_params:,} (learnable MLP)")

    # 4. Attention Ensemble
    print("\n4. Attention Ensemble")
    print("-" * 50)
    ensemble_attn = get_ensemble_model(
        'attention',
        num_classes=num_classes,
        hidden_size=32
    ).to(device)

    logits_attn = ensemble_attn(logits_baseline, logits_recon)
    pred_attn = logits_attn.argmax(dim=1)
    acc_attn = (pred_attn[data.test_mask] == data.y[data.test_mask]).float().mean()

    num_params = sum(p.numel() for p in ensemble_attn.parameters())
    print(f"   Test Accuracy: {acc_attn:.4f}")
    print(f"   Parameters: {num_params:,} (learnable attention)")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Simple Average:     {acc_avg:.4f}")
    print(f"Weighted Average:   {acc_weighted:.4f}")
    print(f"Concatenation:      {acc_concat:.4f}")
    print(f"Attention:          {acc_attn:.4f}")
    print("="*70)

    print("\nNOTE: These are random predictions for demonstration.")
    print("In practice, use trained models for actual ensemble performance.")
    print("\nTo train ensemble on real models, run:")
    print("  python ensemble_train.py --dataset cora --ensemble_type weighted")


def manual_ensemble_example():
    """
    Show how to manually combine predictions without using ensemble classes.
    """
    print("\n" + "="*70)
    print("Manual Ensemble - Code Example")
    print("="*70)

    # Example code snippet
    code = '''
# Get predictions from both models
with torch.no_grad():
    logits1 = baseline_model(data.x, data.edge_index, data.edge_attr)
    logits2 = recon_model(batch_data.x, batch_data.edge_index, ...)

# Method 1: Simple average
ensemble_logits = (logits1 + logits2) / 2.0

# Method 2: Weighted average
weight1, weight2 = 0.6, 0.4  # Can tune on validation set
ensemble_logits = weight1 * logits1 + weight2 * logits2

# Method 3: Voting (for classification)
pred1 = logits1.argmax(dim=1)
pred2 = logits2.argmax(dim=1)
# Take majority vote or use probability-based voting
prob1 = F.softmax(logits1, dim=-1)
prob2 = F.softmax(logits2, dim=-1)
ensemble_pred = (prob1 + prob2).argmax(dim=1)

# Make predictions
final_pred = ensemble_logits.argmax(dim=1)
accuracy = (final_pred[test_mask] == labels[test_mask]).float().mean()
'''
    print(code)


if __name__ == '__main__':
    ensemble_quick_example()
    manual_ensemble_example()

    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("1. Train baseline model:     python gcn.py --dataset cora")
    print("2. Train reconstruction:     python deck-gcn.py --dataset cora")
    print("3. Train ensemble:           python ensemble_train.py --dataset cora")
    print("4. Compare strategies:       python ensemble_simple.py --dataset cora")
    print("="*70)
