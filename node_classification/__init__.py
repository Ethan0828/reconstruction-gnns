"""
Node Classification with Reconstruction Conjecture

This package provides implementations for node classification using the reconstruction conjecture.
"""

from .layers.gnn import (
    NodeGCN,
    NodeReconstructionGCN,
    NodeGINE,
    NodeReconstructionGINE
)

from .data_utils import (
    load_dataset,
    prepare_data,
    get_dataset_info
)

from .subgraph_utils import (
    get_khop_neighborhood,
    get_reconstruction_subgraphs_for_node
)

from .ensemble import (
    AverageEnsemble,
    WeightedAverageEnsemble,
    ConcatenationEnsemble,
    AttentionEnsemble,
    StackingEnsemble,
    get_ensemble_model
)

__all__ = [
    'NodeGCN',
    'NodeReconstructionGCN',
    'NodeGINE',
    'NodeReconstructionGINE',
    'load_dataset',
    'prepare_data',
    'get_dataset_info',
    'get_khop_neighborhood',
    'get_reconstruction_subgraphs_for_node',
    'AverageEnsemble',
    'WeightedAverageEnsemble',
    'ConcatenationEnsemble',
    'AttentionEnsemble',
    'StackingEnsemble',
    'get_ensemble_model',
]
