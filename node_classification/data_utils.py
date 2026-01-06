import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.transforms import NormalizeFeatures
import os


def load_dataset(name, root='./data'):
    """
    Load standard node classification datasets.

    Supported datasets:
    - Cora: Citation network, 2708 nodes, 7 classes
    - CiteSeer: Citation network, 3327 nodes, 6 classes
    - PubMed: Citation network, 19717 nodes, 3 classes
    - Computers: Amazon co-purchase network, 13752 nodes, 10 classes
    - Photo: Amazon co-purchase network, 7650 nodes, 8 classes
    - CS: Coauthor network, 18333 nodes, 15 classes
    - Physics: Coauthor network, 34493 nodes, 5 classes

    Args:
        name: Dataset name
        root: Root directory for data storage

    Returns:
        dataset: PyG dataset object
        data: Single graph data object
    """
    name = name.lower()

    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(
            root=os.path.join(root, 'Planetoid'),
            name=name.capitalize(),
            transform=NormalizeFeatures()
        )
    elif name in ['computers', 'photo']:
        dataset = Amazon(
            root=os.path.join(root, 'Amazon'),
            name=name.capitalize(),
            transform=NormalizeFeatures()
        )
    elif name in ['cs', 'physics']:
        dataset = Coauthor(
            root=os.path.join(root, 'Coauthor'),
            name=name.upper(),
            transform=NormalizeFeatures()
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    data = dataset[0]

    # Add edge attributes if not present (use ones)
    if not hasattr(data, 'edge_attr') or data.edge_attr is None:
        data.edge_attr = torch.ones((data.edge_index.size(1), 1), dtype=torch.float)

    return dataset, data


def get_dataset_info(data):
    """
    Get information about a dataset.

    Args:
        data: PyG data object

    Returns:
        Dictionary with dataset statistics
    """
    info = {
        'num_nodes': data.num_nodes,
        'num_edges': data.edge_index.size(1),
        'num_features': data.num_node_features,
        'num_classes': len(torch.unique(data.y)),
        'has_train_mask': hasattr(data, 'train_mask'),
        'has_val_mask': hasattr(data, 'val_mask'),
        'has_test_mask': hasattr(data, 'test_mask'),
    }

    if hasattr(data, 'train_mask'):
        info['num_train'] = data.train_mask.sum().item()
        info['num_val'] = data.val_mask.sum().item()
        info['num_test'] = data.test_mask.sum().item()

    return info


def create_train_val_test_masks(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Create random train/val/test masks for datasets that don't have them.

    Args:
        num_nodes: Total number of nodes
        train_ratio: Ratio of training nodes
        val_ratio: Ratio of validation nodes
        seed: Random seed

    Returns:
        train_mask, val_mask, test_mask
    """
    torch.manual_seed(seed)

    indices = torch.randperm(num_nodes)

    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train + num_val]] = True
    test_mask[indices[num_train + num_val:]] = True

    return train_mask, val_mask, test_mask


def prepare_data(data, use_fixed_split=True, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Prepare data for training.

    Args:
        data: PyG data object
        use_fixed_split: Use dataset's fixed split if available
        train_ratio: Training ratio if creating new split
        val_ratio: Validation ratio if creating new split
        seed: Random seed

    Returns:
        data: Prepared data object with masks
    """
    if not use_fixed_split or not hasattr(data, 'train_mask'):
        train_mask, val_mask, test_mask = create_train_val_test_masks(
            data.num_nodes, train_ratio, val_ratio, seed
        )
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    return data
