"""
Train ensemble models combining baseline GCN and reconstruction-based GCN.

This script provides two training strategies:
1. Train both base models from scratch and then ensemble
2. Load pre-trained base models and train only the ensemble layer
"""

import torch
import torch.nn.functional as F
import argparse
import numpy as np
import os
from tqdm import tqdm

from layers.gnn import NodeGCN, NodeReconstructionGCN
from data_utils import load_dataset, prepare_data, get_dataset_info
from subgraph_utils import get_reconstruction_subgraphs_for_node
from torch_geometric.data import Batch
from ensemble import get_ensemble_model


def prepare_reconstruction_data(data, num_hops=2, delete_ratio=0.5, max_samples=10):
    """Prepare reconstruction subgraphs for all nodes."""
    all_subgraphs = []
    subgraph_batch = []
    num_nodes = data.num_nodes

    print("Generating reconstruction subgraphs...")
    for node_idx in tqdm(range(num_nodes)):
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

        for subgraph in subgraphs:
            all_subgraphs.append(subgraph)
            subgraph_batch.append(node_idx)

    batch_data = Batch.from_data_list(all_subgraphs)
    subgraph_batch = torch.tensor(subgraph_batch, dtype=torch.long)

    hidden_size = 256
    weights = torch.ones((len(all_subgraphs), hidden_size), dtype=torch.float)

    return batch_data, subgraph_batch, weights


def train_base_models(args, data, dataset, device):
    """Train baseline and reconstruction models from scratch."""
    print("\n" + "="*70)
    print("Training Base Models")
    print("="*70)

    # Prepare reconstruction data
    print("\nPreparing reconstruction data...")
    batch_data, subgraph_batch, weights = prepare_reconstruction_data(
        data, args.num_hops, args.delete_ratio, args.max_samples
    )
    batch_data = batch_data.to(device)
    subgraph_batch = subgraph_batch.to(device)
    weights = weights.to(device)

    # Create baseline model
    print("\nTraining Baseline GCN...")
    baseline_model = NodeGCN(
        node_size=data.num_node_features,
        edge_size=data.edge_attr.size(1),
        hidden_size=args.hidden_size,
        out_size=dataset.num_classes,
        num_layers=args.num_layers
    ).to(device)

    baseline_optimizer = torch.optim.Adam(
        baseline_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Train baseline
    best_val_acc = 0
    patience_counter = 0
    data_device = data.to(device)

    for epoch in range(1, args.base_epochs + 1):
        baseline_model.train()
        baseline_optimizer.zero_grad()

        out = baseline_model(data_device.x, data_device.edge_index, data_device.edge_attr)
        loss = F.cross_entropy(out[data_device.train_mask], data_device.y[data_device.train_mask])

        loss.backward()
        baseline_optimizer.step()

        if epoch % 10 == 0:
            baseline_model.eval()
            with torch.no_grad():
                out = baseline_model(data_device.x, data_device.edge_index, data_device.edge_attr)
                pred = out.argmax(dim=1)
                val_acc = (pred[data_device.val_mask] == data_device.y[data_device.val_mask]).float().mean().item()

                print(f"Baseline Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    torch.save(baseline_model.state_dict(), 'best_baseline.pt')
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    print(f"Early stopping baseline at epoch {epoch}")
                    break

    # Load best baseline model
    baseline_model.load_state_dict(torch.load('best_baseline.pt'))

    # Create reconstruction model
    print("\nTraining Reconstruction GCN...")
    recon_model = NodeReconstructionGCN(
        node_size=data.num_node_features,
        edge_size=data.edge_attr.size(1),
        hidden_size=args.hidden_size,
        out_size=dataset.num_classes,
        num_layers=args.num_layers
    ).to(device)

    recon_optimizer = torch.optim.Adam(
        recon_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Train reconstruction
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(1, args.base_epochs + 1):
        recon_model.train()
        recon_optimizer.zero_grad()

        out = recon_model(
            batch_data.x, batch_data.edge_index, batch_data.edge_attr,
            batch_data.batch, weights, subgraph_batch
        )
        loss = F.cross_entropy(out[data_device.train_mask], data_device.y[data_device.train_mask])

        loss.backward()
        recon_optimizer.step()

        if epoch % 10 == 0:
            recon_model.eval()
            with torch.no_grad():
                out = recon_model(
                    batch_data.x, batch_data.edge_index, batch_data.edge_attr,
                    batch_data.batch, weights, subgraph_batch
                )
                pred = out.argmax(dim=1)
                val_acc = (pred[data_device.val_mask] == data_device.y[data_device.val_mask]).float().mean().item()

                print(f"Recon Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    torch.save(recon_model.state_dict(), 'best_recon.pt')
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    print(f"Early stopping reconstruction at epoch {epoch}")
                    break

    # Load best reconstruction model
    recon_model.load_state_dict(torch.load('best_recon.pt'))

    return baseline_model, recon_model, batch_data, subgraph_batch, weights


def train_ensemble(args, baseline_model, recon_model, batch_data, subgraph_batch,
                   weights, data, dataset, device):
    """Train ensemble model."""
    print("\n" + "="*70)
    print(f"Training {args.ensemble_type.upper()} Ensemble")
    print("="*70)

    # Create ensemble model
    ensemble_model = get_ensemble_model(
        args.ensemble_type,
        num_classes=dataset.num_classes,
        hidden_size=args.ensemble_hidden_size,
        dropout=args.dropout
    ).to(device)

    # Freeze base models
    for param in baseline_model.parameters():
        param.requires_grad = False
    for param in recon_model.parameters():
        param.requires_grad = False

    baseline_model.eval()
    recon_model.eval()

    # Only optimize ensemble parameters
    optimizer = torch.optim.Adam(
        ensemble_model.parameters(),
        lr=args.ensemble_lr,
        weight_decay=args.weight_decay
    )

    data_device = data.to(device)
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0

    print("\nTraining ensemble...")
    for epoch in range(1, args.ensemble_epochs + 1):
        ensemble_model.train()
        optimizer.zero_grad()

        # Get predictions from both models
        with torch.no_grad():
            logits1 = baseline_model(data_device.x, data_device.edge_index, data_device.edge_attr)
            logits2 = recon_model(
                batch_data.x, batch_data.edge_index, batch_data.edge_attr,
                batch_data.batch, weights, subgraph_batch
            )

        # Ensemble prediction
        ensemble_out = ensemble_model(logits1, logits2)

        # Compute loss on training nodes
        loss = F.cross_entropy(
            ensemble_out[data_device.train_mask],
            data_device.y[data_device.train_mask]
        )

        loss.backward()
        optimizer.step()

        # Evaluation
        if epoch % args.eval_freq == 0:
            ensemble_model.eval()
            with torch.no_grad():
                ensemble_out = ensemble_model(logits1, logits2)
                pred = ensemble_out.argmax(dim=1)

                train_acc = (pred[data_device.train_mask] == data_device.y[data_device.train_mask]).float().mean().item()
                val_acc = (pred[data_device.val_mask] == data_device.y[data_device.val_mask]).float().mean().item()
                test_acc = (pred[data_device.test_mask] == data_device.y[data_device.test_mask]).float().mean().item()

                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                      f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

                # Print ensemble weights if weighted ensemble
                if args.ensemble_type == 'weighted' and hasattr(ensemble_model, 'get_weights'):
                    w1, w2 = ensemble_model.get_weights()
                    print(f"  Weights: Baseline={w1:.3f}, Recon={w2:.3f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

    return best_val_acc, best_test_acc


def evaluate_individual_models(baseline_model, recon_model, batch_data, subgraph_batch,
                                weights, data, device):
    """Evaluate individual models for comparison."""
    print("\n" + "="*70)
    print("Individual Model Performance")
    print("="*70)

    data_device = data.to(device)

    # Evaluate baseline
    baseline_model.eval()
    with torch.no_grad():
        out = baseline_model(data_device.x, data_device.edge_index, data_device.edge_attr)
        pred = out.argmax(dim=1)
        baseline_test_acc = (pred[data_device.test_mask] == data_device.y[data_device.test_mask]).float().mean().item()

    print(f"Baseline GCN Test Accuracy: {baseline_test_acc:.4f}")

    # Evaluate reconstruction
    recon_model.eval()
    with torch.no_grad():
        out = recon_model(
            batch_data.x, batch_data.edge_index, batch_data.edge_attr,
            batch_data.batch, weights, subgraph_batch
        )
        pred = out.argmax(dim=1)
        recon_test_acc = (pred[data_device.test_mask] == data_device.y[data_device.test_mask]).float().mean().item()

    print(f"Reconstruction GCN Test Accuracy: {recon_test_acc:.4f}")

    return baseline_test_acc, recon_test_acc


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset, data = load_dataset(args.dataset, root=args.data_root)
    data = prepare_data(data, use_fixed_split=args.use_fixed_split, seed=args.seed)

    info = get_dataset_info(data)
    print("\nDataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Train base models
    baseline_model, recon_model, batch_data, subgraph_batch, weights = train_base_models(
        args, data, dataset, device
    )

    # Evaluate individual models
    baseline_acc, recon_acc = evaluate_individual_models(
        baseline_model, recon_model, batch_data, subgraph_batch, weights, data, device
    )

    # Train ensemble
    val_acc, test_acc = train_ensemble(
        args, baseline_model, recon_model, batch_data, subgraph_batch,
        weights, data, dataset, device
    )

    # Final results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Baseline GCN Test Accuracy:      {baseline_acc:.4f}")
    print(f"Reconstruction GCN Test Accuracy: {recon_acc:.4f}")
    print(f"Ensemble Test Accuracy:           {test_acc:.4f}")
    print(f"Improvement over Baseline:        {(test_acc - baseline_acc)*100:+.2f}%")
    print(f"Improvement over Reconstruction:  {(test_acc - recon_acc)*100:+.2f}%")
    print("="*70)

    return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble training for node classification')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'cs', 'physics'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--use_fixed_split', action='store_true', default=True)

    # Reconstruction arguments
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--delete_ratio', type=float, default=0.5)
    parser.add_argument('--max_samples', type=int, default=10)

    # Base model arguments
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--base_epochs', type=int, default=200)

    # Ensemble arguments
    parser.add_argument('--ensemble_type', type=str, default='weighted',
                        choices=['average', 'weighted', 'concat', 'attention', 'stacking'],
                        help='Type of ensemble method')
    parser.add_argument('--ensemble_hidden_size', type=int, default=128)
    parser.add_argument('--ensemble_epochs', type=int, default=100)
    parser.add_argument('--ensemble_lr', type=float, default=0.01)

    # Training arguments
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--patience', type=int, default=20)

    # Other arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--runs', type=int, default=1)

    args = parser.parse_args()

    if args.runs > 1:
        print(f"\nRunning {args.runs} experiments...\n")
        test_accs = []

        for run in range(args.runs):
            print(f"\n{'='*70}")
            print(f"Run {run + 1}/{args.runs} (seed: {args.seed + run})")
            print(f"{'='*70}")

            args.seed = args.seed + run
            test_acc = main(args)
            test_accs.append(test_acc)

        print(f"\n{'='*70}")
        print(f"Results over {args.runs} runs:")
        print(f"Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        print(f"{'='*70}")
    else:
        main(args)
