"""
Reconstruction-based GCN for node classification using the DECK approach.

This script implements node classification using the reconstruction conjecture:
1. For each node, extract its 2-hop neighborhood
2. Create multiple node-deleted subgraphs from this neighborhood
3. Process each subgraph with a GNN
4. Aggregate subgraph representations to get node embedding
5. Predict node label

This adapts the graph-level DECK approach to node-level predictions.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import sys
import os
import argparse
import numpy as np
from tqdm import tqdm

from layers.gnn import NodeReconstructionGCN
from data_utils import load_dataset, prepare_data, get_dataset_info
from subgraph_utils import get_reconstruction_subgraphs_for_node


def prepare_reconstruction_data(data, num_hops=2, delete_ratio=0.5, max_samples=10, device='cpu'):
    """
    Prepare reconstruction subgraphs for all nodes in the dataset.

    Args:
        data: PyG data object
        num_hops: Number of hops for neighborhood extraction
        delete_ratio: Ratio of nodes to delete
        max_samples: Maximum number of subgraph samples per node
        device: Device to use

    Returns:
        batch_data: Batched subgraph data
        subgraph_batch: Tensor mapping subgraphs to nodes
        weights: Weights for each subgraph
    """
    all_subgraphs = []
    subgraph_batch = []
    num_nodes = data.num_nodes

    print("Generating reconstruction subgraphs for all nodes...")

    for node_idx in tqdm(range(num_nodes)):
        # Get reconstruction subgraphs for this node
        subgraphs, center_indices = get_reconstruction_subgraphs_for_node(
            node_idx=node_idx,
            edge_index=data.edge_index,
            node_features=data.x,
            num_hops=num_hops,
            delete_ratio=delete_ratio,
            max_samples=max_samples,
            edge_attr=data.edge_attr,
            num_nodes=num_nodes
        )

        # Add subgraphs to list
        for subgraph in subgraphs:
            all_subgraphs.append(subgraph)
            subgraph_batch.append(node_idx)

    # Batch all subgraphs
    batch_data = Batch.from_data_list(all_subgraphs)

    # Create subgraph_batch tensor
    subgraph_batch = torch.tensor(subgraph_batch, dtype=torch.long)

    # Create uniform weights for all subgraphs
    num_subgraphs = len(all_subgraphs)
    hidden_size = 256  # Should match model hidden size
    weights = torch.ones((num_subgraphs, hidden_size), dtype=torch.float)

    return batch_data, subgraph_batch, weights


def train(model, batch_data, subgraph_batch, weights, labels, train_mask, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: NodeReconstructionGCN model
        batch_data: Batched subgraph data
        subgraph_batch: Tensor mapping subgraphs to nodes
        weights: Weights for each subgraph
        labels: Node labels
        train_mask: Training mask
        optimizer: Optimizer
        device: Device

    Returns:
        loss: Training loss
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(
        x=batch_data.x,
        edge_index=batch_data.edge_index,
        edge_attr=batch_data.edge_attr,
        batch=batch_data.batch,
        weights=weights,
        subgraph_batch=subgraph_batch
    )

    # Compute loss on training nodes
    loss = F.cross_entropy(out[train_mask], labels[train_mask])

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, batch_data, subgraph_batch, weights, labels, train_mask, val_mask, test_mask, device):
    """
    Evaluate the model on train/val/test sets.
    """
    model.eval()

    # Forward pass
    out = model(
        x=batch_data.x,
        edge_index=batch_data.edge_index,
        edge_attr=batch_data.edge_attr,
        batch=batch_data.batch,
        weights=weights,
        subgraph_batch=subgraph_batch
    )
    pred = out.argmax(dim=1)

    # Compute accuracy for each split
    accs = {}
    masks = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    for split, mask in masks.items():
        correct = pred[mask] == labels[mask]
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

    # Prepare reconstruction data
    print(f"\nPreparing reconstruction subgraphs (k={args.num_hops} hops, "
          f"delete_ratio={args.delete_ratio}, max_samples={args.max_samples})...")

    batch_data, subgraph_batch, weights = prepare_reconstruction_data(
        data,
        num_hops=args.num_hops,
        delete_ratio=args.delete_ratio,
        max_samples=args.max_samples,
        device=device
    )

    print(f"\nReconstructed {len(batch_data)} subgraphs for {data.num_nodes} nodes")
    print(f"Average subgraphs per node: {len(batch_data) / data.num_nodes:.2f}")

    # Move data to device
    batch_data = batch_data.to(device)
    subgraph_batch = subgraph_batch.to(device)
    weights = weights.to(device)
    labels = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    # Create model
    model = NodeReconstructionGCN(
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
        loss = train(model, batch_data, subgraph_batch, weights, labels, train_mask, optimizer, device)

        # Evaluate
        if epoch % args.eval_freq == 0:
            accs = evaluate(model, batch_data, subgraph_batch, weights, labels,
                          train_mask, val_mask, test_mask, device)

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
    parser = argparse.ArgumentParser(description='Reconstruction-based GCN for node classification')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'cs', 'physics'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')
    parser.add_argument('--use_fixed_split', action='store_true', default=True,
                        help='Use fixed train/val/test split if available')

    # Reconstruction arguments
    parser.add_argument('--num_hops', type=int, default=2,
                        help='Number of hops for neighborhood extraction')
    parser.add_argument('--delete_ratio', type=float, default=0.5,
                        help='Ratio of nodes to delete from neighborhood')
    parser.add_argument('--max_samples', type=int, default=10,
                        help='Maximum number of subgraph samples per node')

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
    parser.add_argument('--runs', type=int, default=1,
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
