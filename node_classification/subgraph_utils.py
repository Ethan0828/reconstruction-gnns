import torch
import numpy as np
from torch_geometric.utils import k_hop_subgraph, subgraph
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


def sample_node_deletions(num_nodes, delete_size, max_samples=10):
    """
    Sample random node deletion combinations.

    Args:
        num_nodes: Number of nodes in the subgraph
        delete_size: Number of nodes to delete (ℓ parameter)
        max_samples: Maximum number of deletion samples

    Returns:
        List of tuples, each containing node indices to delete
    """
    if delete_size >= num_nodes:
        return []

    # Calculate total possible combinations
    total_combs = 1
    for i in range(delete_size):
        total_combs *= (num_nodes - i)
        total_combs //= (i + 1)

    # Sample random combinations
    num_samples = min(max_samples, total_combs)

    all_nodes = list(range(num_nodes))
    sampled_deletions = []

    if total_combs <= max_samples:
        # Return all possible combinations
        sampled_deletions = list(combinations(all_nodes, delete_size))
    else:
        # Randomly sample combinations
        seen = set()
        while len(sampled_deletions) < num_samples:
            deletion = tuple(sorted(np.random.choice(all_nodes, delete_size, replace=False)))
            if deletion not in seen:
                seen.add(deletion)
                sampled_deletions.append(deletion)

    return sampled_deletions


def create_node_deleted_subgraph(edge_index, node_features, nodes_to_keep,
                                   edge_attr=None, original_num_nodes=None):
    """
    Create a subgraph by keeping only specified nodes.

    Args:
        edge_index: Edge index [2, num_edges]
        node_features: Node feature matrix [num_nodes, num_features]
        nodes_to_keep: List or tensor of node indices to keep
        edge_attr: Optional edge attributes
        original_num_nodes: Original number of nodes

    Returns:
        Data object representing the node-deleted subgraph
    """
    if len(nodes_to_keep) == 0:
        # Return empty graph
        return Data(
            x=torch.zeros((1, node_features.size(1)), dtype=node_features.dtype),
            edge_index=torch.zeros((2, 0), dtype=torch.long)
        )

    # Create mask for nodes to keep
    if original_num_nodes is None:
        original_num_nodes = node_features.size(0)

    node_mask = torch.zeros(original_num_nodes, dtype=torch.bool)
    node_mask[nodes_to_keep] = True

    # Extract subgraph
    sub_edge_index, sub_edge_attr = subgraph(
        node_mask,
        edge_index,
        edge_attr,
        relabel_nodes=True,
        num_nodes=original_num_nodes
    )

    # Extract node features
    sub_node_features = node_features[nodes_to_keep]

    # Create Data object
    data = Data(
        x=sub_node_features,
        edge_index=sub_edge_index
    )

    if sub_edge_attr is not None:
        data.edge_attr = sub_edge_attr

    return data


def get_reconstruction_subgraphs_for_node(node_idx, edge_index, node_features,
                                          num_hops=2, delete_ratio=0.5,
                                          max_samples=10, edge_attr=None,
                                          num_nodes=None):
    """
    Generate reconstruction subgraphs for a single node.

    Process:
    1. Extract k-hop neighborhood around the node
    2. Create multiple node-deleted subgraphs from this neighborhood
    3. Return all subgraphs for aggregation

    Args:
        node_idx: Index of the target node
        edge_index: Edge index of the full graph [2, num_edges]
        node_features: Node features of the full graph [num_nodes, num_features]
        num_hops: Number of hops for neighborhood (default: 2)
        delete_ratio: Ratio of nodes to delete (default: 0.5)
        max_samples: Maximum number of subgraph samples
        edge_attr: Optional edge attributes
        num_nodes: Total number of nodes in the graph

    Returns:
        subgraphs: List of Data objects (node-deleted subgraphs)
        center_node_new_idx: New index of the center node in each subgraph
                             (None if center node is deleted)
    """
    # Extract k-hop neighborhood
    subset, sub_edge_index, mapping, edge_mask = get_khop_neighborhood(
        node_idx, edge_index, num_hops, num_nodes
    )

    # Extract features and edge attributes for the neighborhood
    sub_node_features = node_features[subset]
    sub_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    neighborhood_size = len(subset)

    # Determine number of nodes to delete
    delete_size = max(1, int(neighborhood_size * delete_ratio))

    # Sample deletion combinations
    deletion_samples = sample_node_deletions(neighborhood_size, delete_size, max_samples)

    # If no valid deletions, return the full neighborhood
    if len(deletion_samples) == 0:
        data = Data(x=sub_node_features, edge_index=sub_edge_index)
        if sub_edge_attr is not None:
            data.edge_attr = sub_edge_attr
        return [data], [mapping.item()]

    # Create subgraphs by deleting nodes
    subgraphs = []
    center_node_indices = []

    for nodes_to_delete in deletion_samples:
        # Determine nodes to keep
        all_nodes = set(range(neighborhood_size))
        nodes_to_keep = sorted(list(all_nodes - set(nodes_to_delete)))

        # Check if center node is kept
        if mapping.item() in nodes_to_delete:
            # Center node is deleted in this subgraph
            center_node_new_idx = None
        else:
            # Find new index of center node
            center_node_new_idx = nodes_to_keep.index(mapping.item())

        # Create subgraph
        nodes_to_keep_global = subset[nodes_to_keep]
        subgraph_data = create_node_deleted_subgraph(
            sub_edge_index,
            sub_node_features,
            nodes_to_keep,
            sub_edge_attr,
            neighborhood_size
        )

        subgraphs.append(subgraph_data)
        center_node_indices.append(center_node_new_idx)

    return subgraphs, center_node_indices
