"""
Quick example demonstrating node classification with reconstruction.

This script shows how to use the reconstruction approach for node classification
on the Cora dataset.
"""

import torch
from data_utils import load_dataset, prepare_data, get_dataset_info
from subgraph_utils import get_reconstruction_subgraphs_for_node


def main():
    print("="*70)
    print("Node Classification with Reconstruction Conjecture - Quick Example")
    print("="*70)

    # Load dataset
    print("\n1. Loading Cora dataset...")
    dataset, data = load_dataset('cora', root='./data')
    data = prepare_data(data, use_fixed_split=True)

    # Print dataset info
    info = get_dataset_info(data)
    print("\nDataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Demonstrate k-hop neighborhood extraction
    print("\n2. Extracting 2-hop neighborhood for node 0...")
    node_idx = 0
    num_hops = 2

    from subgraph_utils import get_khop_neighborhood
    subset, sub_edge_index, mapping, edge_mask = get_khop_neighborhood(
        node_idx=node_idx,
        edge_index=data.edge_index,
        num_hops=num_hops,
        num_nodes=data.num_nodes
    )

    print(f"  Node {node_idx} has {len(subset)} nodes in its {num_hops}-hop neighborhood")
    print(f"  Center node maps to index {mapping.item()} in the subgraph")

    # Demonstrate reconstruction subgraph generation
    print("\n3. Generating reconstruction subgraphs for node 0...")
    subgraphs, center_indices = get_reconstruction_subgraphs_for_node(
        node_idx=node_idx,
        edge_index=data.edge_index,
        node_features=data.x,
        num_hops=2,
        delete_ratio=0.5,
        max_samples=5,
        edge_attr=data.edge_attr,
        num_nodes=data.num_nodes
    )

    print(f"  Generated {len(subgraphs)} reconstruction subgraphs")
    print(f"  Each subgraph has the following properties:")
    for i, (subgraph, center_idx) in enumerate(zip(subgraphs[:3], center_indices[:3])):
        print(f"\n  Subgraph {i+1}:")
        print(f"    - Nodes: {subgraph.num_nodes}")
        print(f"    - Edges: {subgraph.num_edges}")
        print(f"    - Center node index: {center_idx} (None if deleted)")

    # Show how to run baseline model
    print("\n" + "="*70)
    print("4. Running Models")
    print("="*70)

    print("\nTo train the baseline GCN model:")
    print("  python gcn.py --dataset cora --epochs 200 --runs 5")

    print("\nTo train the reconstruction-based GCN model:")
    print("  python deck-gcn.py --dataset cora --num_hops 2 --delete_ratio 0.5 --max_samples 10")

    print("\nFor more options, use --help:")
    print("  python gcn.py --help")
    print("  python deck-gcn.py --help")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
