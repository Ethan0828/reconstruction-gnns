"""
Improved subgraph utilities for node classification with reconstruction.

Key improvements:
1. NEVER delete the center node (most important!)
2. Smarter sampling strategies
3. Optional learned weighting
"""

import torch
import numpy as np
from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_geometric.data import Data
from itertools import combinations


def get_reconstruction_subgraphs_for_node_improved(
    node_idx, edge_index, node_features,
    num_hops=2, delete_ratio=0.5, max_samples=10,
    edge_attr=None, num_nodes=None,
    keep_center=True  # KEY IMPROVEMENT
):
    """
    Generate reconstruction subgraphs for a single node (IMPROVED VERSION).

    Key improvement: The center node is NEVER deleted (keep_center=True by default).
    This preserves the target node's features while still testing reconstruction
    from its neighborhood structure.

    Args:
        node_idx: Index of the target node
        edge_index: Edge index of the full graph [2, num_edges]
        node_features: Node features of the full graph [num_nodes, num_features]
        num_hops: Number of hops for neighborhood (default: 2)
        delete_ratio: Ratio of nodes to delete (default: 0.5)
        max_samples: Maximum number of subgraph samples
        edge_attr: Optional edge attributes
        num_nodes: Total number of nodes in the graph
        keep_center: If True, never delete the center node (RECOMMENDED)

    Returns:
        subgraphs: List of Data objects (node-deleted subgraphs)
        center_node_indices: New index of center node in each subgraph (always present if keep_center=True)
    """
    # Extract k-hop neighborhood
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes
    )

    # Extract features and edge attributes for the neighborhood
    sub_node_features = node_features[subset]
    sub_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    neighborhood_size = len(subset)
    center_idx_in_subgraph = mapping.item()

    # Determine number of nodes to delete
    if keep_center:
        # Delete from OTHER nodes only (exclude center)
        deletable_size = neighborhood_size - 1  # -1 for center node
        delete_size = max(1, int(deletable_size * delete_ratio))
        deletable_indices = [i for i in range(neighborhood_size) if i != center_idx_in_subgraph]
    else:
        # Original behavior: can delete any node
        delete_size = max(1, int(neighborhood_size * delete_ratio))
        deletable_indices = list(range(neighborhood_size))

    # Sample deletion combinations
    if len(deletable_indices) < delete_size:
        # Not enough nodes to delete, return full neighborhood
        data = Data(x=sub_node_features, edge_index=sub_edge_index)
        if sub_edge_attr is not None:
            data.edge_attr = sub_edge_attr
        return [data], [center_idx_in_subgraph]

    # Calculate total possible combinations
    from math import comb
    total_combs = comb(len(deletable_indices), delete_size)
    num_samples = min(max_samples, total_combs)

    # Sample combinations
    deletion_samples = []
    if total_combs <= max_samples:
        # Return all possible combinations
        deletion_samples = list(combinations(deletable_indices, delete_size))
    else:
        # Randomly sample combinations
        seen = set()
        while len(deletion_samples) < num_samples:
            deletion = tuple(sorted(np.random.choice(deletable_indices, delete_size, replace=False)))
            if deletion not in seen:
                seen.add(deletion)
                deletion_samples.append(deletion)

    # Create subgraphs by deleting nodes
    subgraphs = []
    center_node_indices = []

    for nodes_to_delete in deletion_samples:
        # Determine nodes to keep
        all_nodes = set(range(neighborhood_size))
        nodes_to_keep = sorted(list(all_nodes - set(nodes_to_delete)))

        # Find new index of center node (should always be present if keep_center=True)
        try:
            center_node_new_idx = nodes_to_keep.index(center_idx_in_subgraph)
        except ValueError:
            # Center node was deleted (shouldn't happen if keep_center=True)
            center_node_new_idx = None

        # Create subgraph
        from torch_geometric.utils import subgraph as pyg_subgraph

        node_mask = torch.zeros(neighborhood_size, dtype=torch.bool)
        node_mask[nodes_to_keep] = True

        sub_sub_edge_index, sub_sub_edge_attr = pyg_subgraph(
            node_mask,
            sub_edge_index,
            sub_edge_attr,
            relabel_nodes=True,
            num_nodes=neighborhood_size
        )

        # Extract node features
        sub_sub_node_features = sub_node_features[nodes_to_keep]

        # Create Data object
        subgraph_data = Data(
            x=sub_sub_node_features,
            edge_index=sub_sub_edge_index
        )

        if sub_sub_edge_attr is not None:
            subgraph_data.edge_attr = sub_sub_edge_attr

        subgraphs.append(subgraph_data)
        center_node_indices.append(center_node_new_idx)

    return subgraphs, center_node_indices


def get_reconstruction_subgraphs_strategic(
    node_idx, edge_index, node_features,
    num_hops=2, max_samples=10,
    edge_attr=None, num_nodes=None,
    strategy='random'
):
    """
    Generate reconstruction subgraphs with different deletion strategies.

    Strategies:
        - 'random': Random node deletion (default)
        - 'degree': Delete high-degree nodes first (test robustness to hubs)
        - 'distance': Delete nodes farther from center first
        - 'mixed': Mix of different strategies

    Args:
        node_idx: Index of the target node
        edge_index: Edge index of the full graph
        node_features: Node features
        num_hops: Number of hops for neighborhood
        max_samples: Maximum number of subgraph samples
        edge_attr: Optional edge attributes
        num_nodes: Total number of nodes
        strategy: Deletion strategy

    Returns:
        subgraphs: List of subgraph Data objects
        center_node_indices: Indices of center node in each subgraph
    """
    # Extract k-hop neighborhood
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes
    )

    sub_node_features = node_features[subset]
    sub_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
    neighborhood_size = len(subset)
    center_idx = mapping.item()

    # Compute node priorities based on strategy
    if strategy == 'degree':
        # Compute degrees in subgraph
        row, col = sub_edge_index
        degree = torch.zeros(neighborhood_size)
        for i in range(neighborhood_size):
            degree[i] = ((row == i) | (col == i)).sum()

        # Delete high-degree nodes first (more challenging)
        priorities = degree.numpy()

    elif strategy == 'distance':
        # Use BFS to compute distances from center
        from collections import deque

        distances = np.full(neighborhood_size, float('inf'))
        distances[center_idx] = 0

        queue = deque([center_idx])
        adj_list = [[] for _ in range(neighborhood_size)]

        # Build adjacency list
        row, col = sub_edge_index
        for i in range(len(row)):
            adj_list[row[i].item()].append(col[i].item())
            adj_list[col[i].item()].append(row[i].item())

        # BFS
        while queue:
            node = queue.popleft()
            for neighbor in adj_list[node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)

        # Delete farther nodes first
        priorities = distances

    elif strategy == 'mixed':
        # Combine degree and distance
        # Compute degree
        row, col = sub_edge_index
        degree = torch.zeros(neighborhood_size)
        for i in range(neighborhood_size):
            degree[i] = ((row == i) | (col == i)).sum()

        # Compute distance (simplified)
        distances = np.abs(np.arange(neighborhood_size) - center_idx)

        # Normalize and combine
        degree_norm = degree.numpy() / (degree.max() + 1e-6)
        dist_norm = distances / (distances.max() + 1e-6)
        priorities = 0.5 * degree_norm + 0.5 * dist_norm

    else:  # 'random'
        priorities = np.random.rand(neighborhood_size)

    # Never delete center node
    priorities[center_idx] = -float('inf')

    # Create subgraphs by progressively deleting top-priority nodes
    subgraphs = []
    center_node_indices = []

    delete_ratios = np.linspace(0.1, 0.7, max_samples)

    for delete_ratio in delete_ratios:
        delete_size = max(1, int((neighborhood_size - 1) * delete_ratio))

        # Select nodes to delete (excluding center)
        sorted_indices = np.argsort(-priorities)  # High priority first
        nodes_to_delete = [idx for idx in sorted_indices[:delete_size] if idx != center_idx]

        # Nodes to keep
        all_nodes = set(range(neighborhood_size))
        nodes_to_keep = sorted(list(all_nodes - set(nodes_to_delete)))

        # Create subgraph
        node_mask = torch.zeros(neighborhood_size, dtype=torch.bool)
        node_mask[nodes_to_keep] = True

        from torch_geometric.utils import subgraph as pyg_subgraph
        sub_sub_edge_index, sub_sub_edge_attr = pyg_subgraph(
            node_mask,
            sub_edge_index,
            sub_edge_attr,
            relabel_nodes=True,
            num_nodes=neighborhood_size
        )

        sub_sub_node_features = sub_node_features[nodes_to_keep]

        subgraph_data = Data(
            x=sub_sub_node_features,
            edge_index=sub_sub_edge_index
        )

        if sub_sub_edge_attr is not None:
            subgraph_data.edge_attr = sub_sub_edge_attr

        # Find center node's new index
        try:
            center_node_new_idx = nodes_to_keep.index(center_idx)
        except ValueError:
            center_node_new_idx = None

        subgraphs.append(subgraph_data)
        center_node_indices.append(center_node_new_idx)

    return subgraphs, center_node_indices


def get_adaptive_k_hop(node_idx, edge_index, num_nodes, min_k=1, max_k=3, target_size=50):
    """
    Adaptively determine k based on node degree and neighborhood size.

    High-degree nodes: Use smaller k (already well-connected)
    Low-degree nodes: Use larger k (need more context)

    Args:
        node_idx: Node index
        edge_index: Edge index
        num_nodes: Total number of nodes
        min_k: Minimum k value
        max_k: Maximum k value
        target_size: Target neighborhood size

    Returns:
        Optimal k value for this node
    """
    from torch_geometric.utils import degree

    # Compute node degree
    row, col = edge_index
    node_degree = degree(row, num_nodes=num_nodes)[node_idx].item()

    # Try different k values and find best fit
    for k in range(min_k, max_k + 1):
        subset, _, _, _ = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=k,
            edge_index=edge_index,
            num_nodes=num_nodes
        )

        if len(subset) >= target_size:
            return k

    return max_k
