"""
Compare original vs improved reconstruction methods.

This script demonstrates why the improved version (keeping center node)
performs better than the original (potentially deleting center node).
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
import argparse
import numpy as np
from tqdm import tqdm

from layers.gnn import NodeReconstructionGCN
from data_utils import load_dataset, prepare_data, get_dataset_info
from subgraph_utils import get_reconstruction_subgraphs_for_node
from subgraph_utils_improved import (
    get_reconstruction_subgraphs_for_node_improved,
    get_reconstruction_subgraphs_strategic
)


def prepare_reconstruction_data_original(data, num_hops=2, delete_ratio=0.5, max_samples=10):
    """Original method: May delete center node."""
    print("Preparing reconstruction data (ORIGINAL - may delete center)...")
    all_subgraphs = []
    subgraph_batch = []
    center_deleted_count = 0

    for node_idx in tqdm(range(data.num_nodes)):
        subgraphs, center_indices = get_reconstruction_subgraphs_for_node(
            node_idx=node_idx,
            edge_index=data.edge_index,
            node_features=data.x,
            num_hops=num_hops,
            delete_ratio=delete_ratio,
            max_samples=max_samples,
            edge_attr=data.edge_attr,
            num_nodes=data.num_nodes
        )

        # Count how many times center node is deleted
        center_deleted_count += sum(1 for idx in center_indices if idx is None)

        for subgraph in subgraphs:
            all_subgraphs.append(subgraph)
            subgraph_batch.append(node_idx)

    print(f"  Center node deleted in {center_deleted_count}/{data.num_nodes * max_samples} subgraphs "
          f"({100*center_deleted_count/(data.num_nodes * max_samples):.1f}%)")

    batch_data = Batch.from_data_list(all_subgraphs)
    subgraph_batch = torch.tensor(subgraph_batch, dtype=torch.long)
    weights = torch.ones((len(all_subgraphs), 256), dtype=torch.float)

    return batch_data, subgraph_batch, weights


def prepare_reconstruction_data_improved(data, num_hops=2, delete_ratio=0.5, max_samples=10):
    """Improved method: NEVER deletes center node."""
    print("Preparing reconstruction data (IMPROVED - keeps center)...")
    all_subgraphs = []
    subgraph_batch = []
    center_deleted_count = 0

    for node_idx in tqdm(range(data.num_nodes)):
        subgraphs, center_indices = get_reconstruction_subgraphs_for_node_improved(
            node_idx=node_idx,
            edge_index=data.edge_index,
            node_features=data.x,
            num_hops=num_hops,
            delete_ratio=delete_ratio,
            max_samples=max_samples,
            edge_attr=data.edge_attr,
            num_nodes=data.num_nodes,
            keep_center=True  # KEY: Always keep center node
        )

        # Count how many times center node is deleted (should be 0!)
        center_deleted_count += sum(1 for idx in center_indices if idx is None)

        for subgraph in subgraphs:
            all_subgraphs.append(subgraph)
            subgraph_batch.append(node_idx)

    print(f"  Center node deleted in {center_deleted_count}/{data.num_nodes * max_samples} subgraphs "
          f"({100*center_deleted_count/(data.num_nodes * max_samples) if data.num_nodes * max_samples > 0 else 0:.1f}%)")

    batch_data = Batch.from_data_list(all_subgraphs)
    subgraph_batch = torch.tensor(subgraph_batch, dtype=torch.long)
    weights = torch.ones((len(all_subgraphs), 256), dtype=torch.float)

    return batch_data, subgraph_batch, weights


def prepare_reconstruction_data_strategic(data, num_hops=2, max_samples=10, strategy='degree'):
    """Strategic deletion method."""
    print(f"Preparing reconstruction data (STRATEGIC - {strategy})...")
    all_subgraphs = []
    subgraph_batch = []

    for node_idx in tqdm(range(data.num_nodes)):
        subgraphs, center_indices = get_reconstruction_subgraphs_strategic(
            node_idx=node_idx,
            edge_index=data.edge_index,
            node_features=data.x,
            num_hops=num_hops,
            max_samples=max_samples,
            edge_attr=data.edge_attr,
            num_nodes=data.num_nodes,
            strategy=strategy
        )

        for subgraph in subgraphs:
            all_subgraphs.append(subgraph)
            subgraph_batch.append(node_idx)

    batch_data = Batch.from_data_list(all_subgraphs)
    subgraph_batch = torch.tensor(subgraph_batch, dtype=torch.long)
    weights = torch.ones((len(all_subgraphs), 256), dtype=torch.float)

    return batch_data, subgraph_batch, weights


def train_and_evaluate(model, batch_data, subgraph_batch, weights, data, device, epochs=100):
    """Train model and return test accuracy."""
    data_device = data.to(device)
    batch_data = batch_data.to(device)
    subgraph_batch = subgraph_batch.to(device)
    weights = weights.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_acc = 0
    best_test_acc = 0
    patience = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        optimizer.zero_grad()

        out = model(
            batch_data.x, batch_data.edge_index, batch_data.edge_attr,
            batch_data.batch, weights, subgraph_batch
        )
        loss = F.cross_entropy(out[data_device.train_mask], data_device.y[data_device.train_mask])

        loss.backward()
        optimizer.step()

        # Evaluate
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(
                    batch_data.x, batch_data.edge_index, batch_data.edge_attr,
                    batch_data.batch, weights, subgraph_batch
                )
                pred = out.argmax(dim=1)

                val_acc = (pred[data_device.val_mask] == data_device.y[data_device.val_mask]).float().mean().item()
                test_acc = (pred[data_device.test_mask] == data_device.y[data_device.test_mask]).float().mean().item()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    patience = 0
                else:
                    patience += 1

                if patience >= 5:
                    break

    return best_test_acc


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}\n")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset, data = load_dataset(args.dataset, root=args.data_root)
    data = prepare_data(data, use_fixed_split=True, seed=args.seed)

    info = get_dataset_info(data)
    print("\nDataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n" + "="*80)
    print("COMPARING RECONSTRUCTION METHODS")
    print("="*80)

    results = {}

    # 1. Original method
    if not args.skip_original:
        print("\n[1/3] Original Method (may delete center node)")
        print("-" * 80)

        batch_data, subgraph_batch, weights = prepare_reconstruction_data_original(
            data, args.num_hops, args.delete_ratio, args.max_samples
        )

        model = NodeReconstructionGCN(
            node_size=data.num_node_features,
            edge_size=data.edge_attr.size(1),
            hidden_size=256,
            out_size=dataset.num_classes,
            num_layers=2
        ).to(device)

        print("Training...")
        acc = train_and_evaluate(model, batch_data, subgraph_batch, weights, data, device, epochs=args.epochs)
        results['Original'] = acc
        print(f"Test Accuracy: {acc:.4f}")

    # 2. Improved method (keep center)
    print("\n[2/3] Improved Method (always keep center node)")
    print("-" * 80)

    batch_data, subgraph_batch, weights = prepare_reconstruction_data_improved(
        data, args.num_hops, args.delete_ratio, args.max_samples
    )

    model = NodeReconstructionGCN(
        node_size=data.num_node_features,
        edge_size=data.edge_attr.size(1),
        hidden_size=256,
        out_size=dataset.num_classes,
        num_layers=2
    ).to(device)

    print("Training...")
    acc = train_and_evaluate(model, batch_data, subgraph_batch, weights, data, device, epochs=args.epochs)
    results['Improved (keep center)'] = acc
    print(f"Test Accuracy: {acc:.4f}")

    # 3. Strategic method
    print("\n[3/3] Strategic Method (degree-based deletion)")
    print("-" * 80)

    batch_data, subgraph_batch, weights = prepare_reconstruction_data_strategic(
        data, args.num_hops, args.max_samples, strategy='degree'
    )

    model = NodeReconstructionGCN(
        node_size=data.num_node_features,
        edge_size=data.edge_attr.size(1),
        hidden_size=256,
        out_size=dataset.num_classes,
        num_layers=2
    ).to(device)

    print("Training...")
    acc = train_and_evaluate(model, batch_data, subgraph_batch, weights, data, device, epochs=args.epochs)
    results['Strategic (degree)'] = acc
    print(f"Test Accuracy: {acc:.4f}")

    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for i, (method, acc) in enumerate(sorted_results, 1):
        marker = "★" if i == 1 else " "
        print(f"{marker} {i}. {method:30s} {acc:.4f}")

    if len(results) >= 2:
        best_method, best_acc = sorted_results[0]
        if 'Original' in results:
            improvement = (best_acc - results['Original']) * 100
            print(f"\nBest improvement: +{improvement:.2f}% over original")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
1. Keeping the center node is CRUCIAL for node classification
   - Center node contains the target node's features
   - Deleting it forces the model to guess without direct information

2. Strategic deletion can help by:
   - Focusing on structurally important nodes
   - Testing robustness to specific perturbations

3. Even with improvements, reconstruction may still underperform baseline
   - Baseline sees full graph (global context)
   - Reconstruction only sees k-hop neighborhood (local context)

RECOMMENDATION: Use ensemble methods to combine baseline + reconstruction
    """)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare reconstruction methods')

    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'cs', 'physics'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--delete_ratio', type=float, default=0.5)
    parser.add_argument('--max_samples', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--skip_original', action='store_true',
                        help='Skip original method (save time)')

    args = parser.parse_args()
    main(args)
