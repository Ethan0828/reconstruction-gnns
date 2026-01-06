"""
Baseline GCN for node classification.

This script trains a standard GCN on the full graph for node classification.
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
import sys
import os
import argparse
import numpy as np

from layers.gnn import NodeGCN
from data_utils import load_dataset, prepare_data, get_dataset_info


def train(model, data, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index, data.edge_attr)

    # Compute loss on training nodes
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, device):
    """
    Evaluate the model on train/val/test sets.
    """
    model.eval()

    # Forward pass
    out = model(data.x, data.edge_index, data.edge_attr)
    pred = out.argmax(dim=1)

    # Compute accuracy for each split
    accs = {}
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f'{split}_mask')
        correct = pred[mask] == data.y[mask]
        accs[split] = correct.sum().item() / mask.sum().item()

    return accs


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset, data = load_dataset(args.dataset, root=args.data_root)
    data = prepare_data(data, use_fixed_split=args.use_fixed_split, seed=args.seed)

    # Print dataset info
    info = get_dataset_info(data)
    print("\nDataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Move data to device
    data = data.to(device)

    # Create model
    model = NodeGCN(
        node_size=data.num_node_features,
        edge_size=data.edge_attr.size(1),
        hidden_size=args.hidden_size,
        out_size=dataset.num_classes,
        num_layers=args.num_layers
    ).to(device)

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0

    print("\nTraining started...")
    for epoch in range(1, args.epochs + 1):
        # Train
        loss = train(model, data, optimizer, device)

        # Evaluate
        if epoch % args.eval_freq == 0:
            accs = evaluate(model, data, device)

            print(f"Epoch {epoch:04d} | Loss: {loss:.4f} | "
                  f"Train: {accs['train']:.4f} | Val: {accs['val']:.4f} | Test: {accs['test']:.4f}")

            # Early stopping
            if accs['val'] > best_val_acc:
                best_val_acc = accs['val']
                best_test_acc = accs['test']
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Final evaluation
    print("\n" + "="*50)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {best_test_acc:.4f}")
    print("="*50)

    return best_val_acc, best_test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline GCN for node classification')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'cs', 'physics'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')
    parser.add_argument('--use_fixed_split', action='store_true', default=True,
                        help='Use fixed train/val/test split if available')

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GNN layers')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='Evaluation frequency')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs with different seeds')

    args = parser.parse_args()

    # Run multiple times with different seeds
    if args.runs > 1:
        print(f"\nRunning {args.runs} experiments with different seeds...\n")
        val_accs = []
        test_accs = []

        for run in range(args.runs):
            print(f"\n{'='*50}")
            print(f"Run {run + 1}/{args.runs} (seed: {args.seed + run})")
            print(f"{'='*50}")

            args.seed = args.seed + run
            val_acc, test_acc = main(args)
            val_accs.append(val_acc)
            test_accs.append(test_acc)

        print(f"\n{'='*50}")
        print(f"Results over {args.runs} runs:")
        print(f"Validation Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
        print(f"Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        print(f"{'='*50}")
    else:
        main(args)
