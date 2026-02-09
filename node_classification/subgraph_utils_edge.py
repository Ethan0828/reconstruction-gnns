"""
Edge-deletion based subgraph utilities for node classification with reconstruction.

Key improvement: Delete EDGES instead of NODES
- Preserves all node features (including center node)
- Only modifies graph topology/connectivity
- Better for node classification tasks
"""

import torch
import numpy as np
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from itertools import combinations


def get_khop_neighborhood(node_idx, edge_index, num_hops=2, num_nodes=None):
    """
    Extract k-hop neighborhood subgraph for a given node.

    Args:
        node_idx: Index of the center node
        edge_index: Edge index of the full graph [2, num_edges]
        num_hops: Number of hops (default: 2)
        num_nodes: Total number of nodes in the graph

    Returns:
        subset: Node indices in the k-hop neighborhood
        sub_edge_index: Edge index of the subgraph
        mapping: Mapping from old node indices to new indices
        edge_mask: Boolean mask for edges in the subgraph
    """
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes
    )
    return subset, sub_edge_index, mapping, edge_mask


def sample_edge_deletions(num_edges, delete_size, max_samples=10):
    """
    Sample random edge deletion combinations.

    Args:
        num_edges: Number of edges in the subgraph
        delete_size: Number of edges to delete
        max_samples: Maximum number of deletion samples

    Returns:
        List of tuples, each containing edge indices to delete
    """
    if delete_size >= num_edges or num_edges == 0:
        return []

    # Calculate total possible combinations
    from math import comb
    try:
        total_combs = comb(num_edges, delete_size)
    except:
        # Fallback for older Python versions
        total_combs = 1
        for i in range(delete_size):
            total_combs *= (num_edges - i)
            total_combs //= (i + 1)

    # Sample random combinations
    num_samples = min(max_samples, total_combs)

    all_edges = list(range(num_edges))
    sampled_deletions = []

    if total_combs <= max_samples:
        # Return all possible combinations
        sampled_deletions = list(combinations(all_edges, delete_size))
    else:
        # Randomly sample combinations
        seen = set()
        max_attempts = max_samples * 10  # Prevent infinite loop
        attempts = 0

        while len(sampled_deletions) < num_samples and attempts < max_attempts:
            deletion = tuple(sorted(np.random.choice(all_edges, delete_size, replace=False)))
            if deletion not in seen:
                seen.add(deletion)
                sampled_deletions.append(deletion)
            attempts += 1

    return sampled_deletions


def create_edge_deleted_subgraph(edge_index, node_features, edges_to_keep,
                                  edge_attr=None):
    """
    Create a subgraph by keeping only specified edges.
    ALL NODES ARE PRESERVED - only edges are removed.

    Args:
        edge_index: Edge index [2, num_edges]
        node_features: Node feature matrix [num_nodes, num_features]
        edges_to_keep: List or tensor of edge indices to keep
        edge_attr: Optional edge attributes

    Returns:
        Data object representing the edge-deleted subgraph
    """
    if len(edges_to_keep) == 0:
        # Return graph with nodes but no edges
        return Data(
            x=node_features,
            edge_index=torch.zeros((2, 0), dtype=torch.long)
        )

    # Select edges to keep
    sub_edge_index = edge_index[:, edges_to_keep]

    # Select edge attributes if present
    sub_edge_attr = None
    if edge_attr is not None:
        sub_edge_attr = edge_attr[edges_to_keep]

    # Create Data object (all node features preserved)
    data = Data(
        x=node_features,
        edge_index=sub_edge_index
    )

    if sub_edge_attr is not None:
        data.edge_attr = sub_edge_attr

    return data


def get_reconstruction_subgraphs_for_node_edge_deletion(
    node_idx, edge_index, node_features,
    num_hops=2, delete_ratio=0.5, max_samples=10,
    edge_attr=None, num_nodes=None,
    exclude_center_edges=True
):
    """
    Generate reconstruction subgraphs by DELETING EDGES (not nodes).

    This is better for node classification because:
    1. All node features are preserved (including center node)
    2. Only graph topology is modified
    3. Tests robustness to missing connections

    Args:
        node_idx: Index of the target node
        edge_index: Edge index of the full graph [2, num_edges]
        node_features: Node features of the full graph [num_nodes, num_features]
        num_hops: Number of hops for neighborhood (default: 2)
        delete_ratio: Ratio of edges to delete (default: 0.5)
        max_samples: Maximum number of subgraph samples
        edge_attr: Optional edge attributes
        num_nodes: Total number of nodes in the graph
        exclude_center_edges: If True, never delete edges connected to center node (recommended)

    Returns:
        subgraphs: List of Data objects (edge-deleted subgraphs)
        center_node_indices: Index of center node in each subgraph (always same, since nodes not relabeled)
    """
    # Extract k-hop neighborhood
    subset, sub_edge_index, mapping, edge_mask = get_khop_neighborhood(
        node_idx, edge_index, num_hops, num_nodes
    )

    # Extract features and edge attributes for the neighborhood
    sub_node_features = node_features[subset]
    sub_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    num_edges = sub_edge_index.size(1)
    center_idx_in_subgraph = mapping.item()

    if num_edges == 0:
        # No edges in neighborhood, return as is
        data = Data(x=sub_node_features, edge_index=sub_edge_index)
        return [data], [center_idx_in_subgraph]

    # Identify which edges are connected to center node
    if exclude_center_edges:
        # Find edges connected to center node
        center_edge_mask = (sub_edge_index[0] == center_idx_in_subgraph) | \
                          (sub_edge_index[1] == center_idx_in_subgraph)

        # Edges that can be deleted (not connected to center)
        deletable_edge_indices = torch.where(~center_edge_mask)[0].tolist()
        protected_edge_indices = torch.where(center_edge_mask)[0].tolist()
    else:
        # All edges can be deleted
        deletable_edge_indices = list(range(num_edges))
        protected_edge_indices = []

    if len(deletable_edge_indices) == 0:
        # No edges to delete, return original neighborhood
        data = Data(x=sub_node_features, edge_index=sub_edge_index)
        if sub_edge_attr is not None:
            data.edge_attr = sub_edge_attr
        return [data], [center_idx_in_subgraph]

    # Determine number of edges to delete
    delete_size = max(1, int(len(deletable_edge_indices) * delete_ratio))

    # Sample deletion combinations
    deletion_samples = sample_edge_deletions(len(deletable_edge_indices), delete_size, max_samples)

    if len(deletion_samples) == 0:
        # Can't create valid deletions, return original
        data = Data(x=sub_node_features, edge_index=sub_edge_index)
        if sub_edge_attr is not None:
            data.edge_attr = sub_edge_attr
        return [data], [center_idx_in_subgraph]

    # Create subgraphs by deleting edges
    subgraphs = []
    center_node_indices = []

    for deleted_edge_positions in deletion_samples:
        # Convert positions in deletable list to actual edge indices
        edges_to_delete_actual = [deletable_edge_indices[i] for i in deleted_edge_positions]

        # Determine edges to keep
        all_edges = set(range(num_edges))
        edges_to_keep = sorted(list(all_edges - set(edges_to_delete_actual)))

        # Create subgraph (all nodes preserved, only edges removed)
        subgraph_data = create_edge_deleted_subgraph(
            sub_edge_index,
            sub_node_features,
            edges_to_keep,
            sub_edge_attr
        )

        subgraphs.append(subgraph_data)
        # Center node index remains the same (no relabeling)
        center_node_indices.append(center_idx_in_subgraph)

    return subgraphs, center_node_indices


def get_reconstruction_subgraphs_strategic_edge_deletion(
    node_idx, edge_index, node_features,
    num_hops=2, max_samples=10,
    edge_attr=None, num_nodes=None,
    strategy='random'
):
    """
    Generate reconstruction subgraphs with strategic edge deletion.

    Strategies:
        - 'random': Random edge deletion (default)
        - 'high_degree': Delete edges connected to high-degree nodes first
        - 'low_degree': Delete edges connected to low-degree nodes first
        - 'distance': Delete edges farther from center first
        - 'mixed': Mix of different strategies

    Args:
        node_idx: Index of the target node
        edge_index: Edge index of the full graph
        node_features: Node features
        num_hops: Number of hops for neighborhood
        max_samples: Maximum number of subgraph samples
        edge_attr: Optional edge attributes
        num_nodes: Total number of nodes
        strategy: Edge deletion strategy

    Returns:
        subgraphs: List of subgraph Data objects
        center_node_indices: Indices of center node in each subgraph
    """
    # Extract k-hop neighborhood
    subset, sub_edge_index, mapping, edge_mask = get_khop_neighborhood(
        node_idx, edge_index, num_hops, num_nodes
    )

    sub_node_features = node_features[subset]
    sub_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
    neighborhood_size = len(subset)
    center_idx = mapping.item()
    num_edges = sub_edge_index.size(1)

    if num_edges == 0:
        data = Data(x=sub_node_features, edge_index=sub_edge_index)
        return [data], [center_idx]

    # Compute edge priorities based on strategy
    edge_priorities = np.zeros(num_edges)

    if strategy == 'high_degree':
        # Delete edges connected to high-degree nodes
        from torch_geometric.utils import degree
        node_degrees = degree(sub_edge_index[0], num_nodes=neighborhood_size)

        for edge_idx in range(num_edges):
            src, dst = sub_edge_index[:, edge_idx]
            # Priority = sum of degrees of connected nodes
            edge_priorities[edge_idx] = node_degrees[src] + node_degrees[dst]

    elif strategy == 'low_degree':
        # Delete edges connected to low-degree nodes
        from torch_geometric.utils import degree
        node_degrees = degree(sub_edge_index[0], num_nodes=neighborhood_size)

        for edge_idx in range(num_edges):
            src, dst = sub_edge_index[:, edge_idx]
            edge_priorities[edge_idx] = -(node_degrees[src] + node_degrees[dst])

    elif strategy == 'distance':
        # Delete edges farther from center (using simple heuristic)
        for edge_idx in range(num_edges):
            src, dst = sub_edge_index[:, edge_idx]
            # Edges not connected to center have higher priority
            if src == center_idx or dst == center_idx:
                edge_priorities[edge_idx] = -1.0  # Low priority
            else:
                # Higher priority for edges farther away
                edge_priorities[edge_idx] = abs(src - center_idx) + abs(dst - center_idx)

    elif strategy == 'mixed':
        # Combine degree and distance
        from torch_geometric.utils import degree
        node_degrees = degree(sub_edge_index[0], num_nodes=neighborhood_size)

        for edge_idx in range(num_edges):
            src, dst = sub_edge_index[:, edge_idx]
            degree_score = (node_degrees[src] + node_degrees[dst]).item()
            distance_score = 0 if (src == center_idx or dst == center_idx) else 1
            edge_priorities[edge_idx] = 0.5 * degree_score + 0.5 * distance_score

    else:  # 'random'
        edge_priorities = np.random.rand(num_edges)

    # Create subgraphs by progressively deleting top-priority edges
    subgraphs = []
    center_node_indices = []

    delete_ratios = np.linspace(0.1, 0.7, max_samples)

    for delete_ratio in delete_ratios:
        delete_size = max(1, int(num_edges * delete_ratio))

        # Select edges to delete based on priority
        sorted_edge_indices = np.argsort(-edge_priorities)  # High priority first
        edges_to_delete = sorted_edge_indices[:delete_size].tolist()

        # Edges to keep
        all_edges = set(range(num_edges))
        edges_to_keep = sorted(list(all_edges - set(edges_to_delete)))

        # Create subgraph
        subgraph_data = create_edge_deleted_subgraph(
            sub_edge_index,
            sub_node_features,
            edges_to_keep,
            sub_edge_attr
        )

        subgraphs.append(subgraph_data)
        center_node_indices.append(center_idx)

    return subgraphs, center_node_indices


def compare_node_vs_edge_deletion(node_idx, edge_index, node_features, num_hops=2, num_nodes=None):
    """
    Helper function to visualize the difference between node and edge deletion.

    Returns statistics about both approaches.
    """
    # Node deletion
    from subgraph_utils import get_reconstruction_subgraphs_for_node

    node_subgraphs, node_centers = get_reconstruction_subgraphs_for_node(
        node_idx, edge_index, node_features,
        num_hops=num_hops, delete_ratio=0.5, max_samples=10,
        num_nodes=num_nodes
    )

    # Edge deletion
    edge_subgraphs, edge_centers = get_reconstruction_subgraphs_for_node_edge_deletion(
        node_idx, edge_index, node_features,
        num_hops=num_hops, delete_ratio=0.5, max_samples=10,
        num_nodes=num_nodes
    )

    stats = {
        'node_deletion': {
            'num_subgraphs': len(node_subgraphs),
            'center_preserved': sum(1 for c in node_centers if c is not None),
            'center_deleted': sum(1 for c in node_centers if c is None),
            'avg_nodes': np.mean([sg.num_nodes for sg in node_subgraphs]),
            'avg_edges': np.mean([sg.num_edges for sg in node_subgraphs]),
        },
        'edge_deletion': {
            'num_subgraphs': len(edge_subgraphs),
            'center_preserved': len(edge_subgraphs),  # Always preserved
            'center_deleted': 0,  # Never deleted
            'avg_nodes': np.mean([sg.num_nodes for sg in edge_subgraphs]),
            'avg_edges': np.mean([sg.num_edges for sg in edge_subgraphs]),
        }
    }

    return stats


if __name__ == '__main__':
    print("Edge Deletion vs Node Deletion for Reconstruction")
    print("=" * 60)
    print("\nKey Advantages of Edge Deletion:")
    print("  ✓ Preserves all node features (including center node)")
    print("  ✓ Only modifies graph connectivity/topology")
    print("  ✓ Better for node classification tasks")
    print("  ✓ No information loss from deleted nodes")
    print("\nUsage:")
    print("  from subgraph_utils_edge import get_reconstruction_subgraphs_for_node_edge_deletion")
    print("  subgraphs, center_indices = get_reconstruction_subgraphs_for_node_edge_deletion(...)")
