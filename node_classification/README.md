# Node Classification with Reconstruction Conjecture

This directory contains implementations for node classification using the reconstruction conjecture approach. Instead of predicting labels for entire graphs, we adapt the reconstruction conjecture to predict labels for individual nodes.

## Overview

### Reconstruction Conjecture for Node Classification

The key idea is to apply the reconstruction conjecture at the node level:

1. **For each node v**:
   - Extract its k-hop neighborhood (default: k=2)
   - Create multiple node-deleted subgraphs from this neighborhood
   - Process each subgraph through a GNN
   - Aggregate subgraph representations to obtain a node embedding
   - Predict the node's label

2. **Hypothesis**: If the k-hop neighborhood's vertex-deleted deck contains sufficient information to reconstruct the local structure, it should also contain sufficient information for node classification.

### Comparison with Graph-Level Reconstruction

| Aspect | Graph-Level (Original) | Node-Level (This Implementation) |
|--------|----------------------|----------------------------------|
| Input | Entire graph | k-hop neighborhood per node |
| Subgraph creation | Delete nodes from graph | Delete nodes from neighborhood |
| Aggregation | One embedding per graph | One embedding per node |
| Prediction | Graph label | Node label |
| Task | Graph classification | Node classification |

## Files

- `layers/gnn.py`: GNN layer implementations for node classification
  - `NodeGCN`: Baseline GCN for node classification
  - `NodeReconstructionGCN`: Reconstruction-based GCN for node classification
  - `NodeGINE`: Baseline GIN for node classification
  - `NodeReconstructionGINE`: Reconstruction-based GIN for node classification

- `subgraph_utils.py`: Utilities for extracting k-hop neighborhoods and creating node-deleted subgraphs
  - `get_khop_neighborhood()`: Extract k-hop neighborhood around a node
  - `sample_node_deletions()`: Sample random node deletion combinations
  - `create_node_deleted_subgraph()`: Create subgraph by removing nodes
  - `get_reconstruction_subgraphs_for_node()`: Main function to generate reconstruction subgraphs

- `data_utils.py`: Dataset loading and preparation utilities
  - `load_dataset()`: Load standard node classification datasets
  - `prepare_data()`: Prepare train/val/test splits

- `gcn.py`: Baseline GCN training script (standard approach)
- `deck-gcn.py`: Reconstruction-based GCN training script (reconstruction approach)

## Datasets

We support the following standard node classification benchmarks:

### Citation Networks (Planetoid)
- **Cora**: 2,708 nodes, 7 classes (Machine Learning papers)
- **CiteSeer**: 3,327 nodes, 6 classes (Scientific papers)
- **PubMed**: 19,717 nodes, 3 classes (Diabetes-related papers)

### Co-purchase Networks (Amazon)
- **Computers**: 13,752 nodes, 10 classes
- **Photo**: 7,650 nodes, 8 classes

### Coauthor Networks
- **CS**: 18,333 nodes, 15 classes (Computer Science)
- **Physics**: 34,493 nodes, 5 classes

## Installation

Requirements:
```bash
pip install torch
pip install torch-geometric
pip install numpy
pip install tqdm
```

## Usage

### Baseline GCN (Standard Approach)

Train a standard GCN on the full graph:

```bash
cd node_classification
python gcn.py --dataset cora --epochs 200 --hidden_size 256 --num_layers 2
```

Options:
- `--dataset`: Dataset name (cora, citeseer, pubmed, computers, photo, cs, physics)
- `--epochs`: Number of training epochs (default: 200)
- `--hidden_size`: Hidden layer size (default: 256)
- `--num_layers`: Number of GNN layers (default: 2)
- `--lr`: Learning rate (default: 0.01)
- `--weight_decay`: Weight decay (default: 5e-4)
- `--seed`: Random seed (default: 42)
- `--runs`: Number of runs with different seeds (default: 5)

Example with multiple runs:
```bash
python gcn.py --dataset cora --runs 5
```

### Reconstruction-based GCN (DECK Approach)

Train a reconstruction-based GCN using k-hop neighborhoods:

```bash
cd node_classification
python deck-gcn.py --dataset cora --num_hops 2 --delete_ratio 0.5 --max_samples 10
```

Additional options for reconstruction:
- `--num_hops`: Number of hops for neighborhood extraction (default: 2)
- `--delete_ratio`: Ratio of nodes to delete from neighborhood (default: 0.5)
- `--max_samples`: Maximum number of subgraph samples per node (default: 10)

Example configurations:

**Small neighborhoods (faster, less context):**
```bash
python deck-gcn.py --dataset cora --num_hops 1 --delete_ratio 0.3 --max_samples 5
```

**Large neighborhoods (slower, more context):**
```bash
python deck-gcn.py --dataset cora --num_hops 3 --delete_ratio 0.5 --max_samples 15
```

**Comparison experiment:**
```bash
# Baseline
python gcn.py --dataset cora --runs 5 --seed 42

# Reconstruction
python deck-gcn.py --dataset cora --runs 5 --seed 42 --num_hops 2 --delete_ratio 0.5
```

## Implementation Details

### Baseline Model (gcn.py)

The baseline model is a standard GCN that:
1. Processes the full graph
2. Applies multiple GNN layers
3. Produces node embeddings
4. Predicts node labels directly

### Reconstruction Model (deck-gcn.py)

The reconstruction model follows these steps:

1. **Preprocessing** (once before training):
   ```python
   for each node v in graph:
       neighborhood = extract_k_hop_neighborhood(v, k=2)
       subgraphs = create_node_deleted_subgraphs(neighborhood, delete_ratio=0.5, max_samples=10)
       store subgraphs for node v
   ```

2. **Training** (each epoch):
   ```python
   # Batch all subgraphs together
   batch_data = batch_all_subgraphs()

   # Process through GNN
   subgraph_embeddings = GNN(batch_data)

   # Aggregate subgraphs to node embeddings
   node_embeddings = aggregate_by_node(subgraph_embeddings)

   # Predict labels
   predictions = MLP(node_embeddings)
   ```

### Key Parameters

- **num_hops (k)**: Controls the neighborhood size
  - k=1: Direct neighbors only
  - k=2: 2-hop neighbors (recommended)
  - k=3: 3-hop neighbors (may be too large for some datasets)

- **delete_ratio (ℓ)**: Fraction of neighborhood nodes to delete
  - 0.3: Delete 30% of nodes (more subgraphs kept intact)
  - 0.5: Delete 50% of nodes (balanced, recommended)
  - 0.7: Delete 70% of nodes (more aggressive reconstruction)

- **max_samples**: Number of different subgraphs per node
  - Small (5-10): Faster training, less diverse
  - Medium (10-20): Good balance
  - Large (20+): More comprehensive but slower

## Expected Results

Typical accuracy on Cora dataset:

| Method | Validation Acc | Test Acc |
|--------|---------------|----------|
| Baseline GCN | 79-82% | 78-81% |
| Reconstruction GCN (k=2, ℓ=0.5) | 76-80% | 75-79% |

Note: Reconstruction methods may have slightly lower accuracy but provide insights into local structure importance.

## Computational Complexity

### Baseline GCN
- Time per epoch: O(|E| × d × L)
- Memory: O(|V| × d + |E|)

where |V| = nodes, |E| = edges, d = hidden dimension, L = layers

### Reconstruction GCN
- Preprocessing: O(|V| × k × S × |E_local|)
- Time per epoch: O(S × |V| × |E_local| × d × L)
- Memory: O(S × |V| × |E_local|)

where S = max_samples, k = num_hops, |E_local| = average edges in k-hop neighborhood

**Trade-off**: Reconstruction methods are computationally more expensive but provide a principled way to study local graph structure.

## Extending to Other Models

To implement reconstruction for other GNN architectures:

1. **Create baseline model** in `layers/gnn.py`:
   ```python
   class NodeYourModel(torch.nn.Module):
       def forward(self, x, edge_index, edge_attr):
           # Process full graph
           # Return node predictions
   ```

2. **Create reconstruction model**:
   ```python
   class NodeReconstructionYourModel(torch.nn.Module):
       def forward(self, x, edge_index, edge_attr, batch, weights, subgraph_batch):
           # Process batched subgraphs
           # Aggregate to nodes
           # Return node predictions
   ```

3. **Create training script** following the pattern in `deck-gcn.py`

## Citation

If you use this code, please cite the original reconstruction conjecture work and this implementation:

```bibtex
@article{reconstruction-gnns,
  title={Graph Neural Networks with the Reconstruction Conjecture},
  note={Node classification extension}
}
```

## Future Work

Potential extensions:
- Adaptive neighborhood selection (different k per node)
- Learned aggregation weights (instead of uniform)
- Attention-based subgraph aggregation
- Hierarchical reconstruction (multi-scale neighborhoods)
- Mini-batch training for large graphs
- Graph sampling techniques for scalability
