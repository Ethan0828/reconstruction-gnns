# Graph Neural Networks with Reconstruction Conjecture

This repository implements Graph Neural Networks (GNNs) based on the **reconstruction conjecture** for both **graph-level** and **node-level** tasks.

## Overview

The reconstruction conjecture states that a graph can be reconstructed from the collection of all its vertex-deleted subgraphs (the "deck"). This repository explores using this principle for:

1. **Graph-level tasks** (original implementation): Classify entire graphs using their vertex-deleted subgraphs
2. **Node-level tasks** (new implementation): Classify individual nodes using reconstruction on their k-hop neighborhoods

## Repository Structure

```
reconstruction-gnns/
├── cycles/              # Graph-level: k-cycle detection task
├── multitask/           # Graph-level: Multi-task graph property prediction
├── csl/                 # Graph-level: CSL benchmark (graph isomorphism)
├── node_classification/ # Node-level: Node classification tasks (NEW)
└── README.md
```

## Graph-Level Tasks (Original)

### Datasets

- **Cycles**: Detect presence of k-cycles (k=4,6,8) in graphs
- **Multitask**: Predict multiple graph properties simultaneously
- **CSL**: Graph isomorphism benchmark (10-class classification)

### Data Download

For cycles dataset: https://www.dropbox.com/sh/zb0mzy27t648719/AABr3QyYzz2388Npa6079Qtma?dl=0

### Quick Start

```bash
# Navigate to task directory
cd cycles

# Train baseline GCN on full graphs
python gcn.py

# Train reconstruction-based GCN (DECK approach)
python deck-n-l-gcn.py
```

## Node-Level Tasks (NEW)

### Overview

Adapts the reconstruction conjecture to node classification:

1. For each node, extract its k-hop neighborhood
2. Create multiple node-deleted subgraphs from the neighborhood
3. Process subgraphs with GNN and aggregate to get node embeddings
4. Predict node labels

### Supported Datasets

**Citation Networks:**
- Cora (2,708 nodes, 7 classes)
- CiteSeer (3,327 nodes, 6 classes)
- PubMed (19,717 nodes, 3 classes)

**Co-purchase Networks:**
- Computers (13,752 nodes, 10 classes)
- Photo (7,650 nodes, 8 classes)

**Coauthor Networks:**
- CS (18,333 nodes, 15 classes)
- Physics (34,493 nodes, 5 classes)

### Quick Start

```bash
# Navigate to node classification directory
cd node_classification

# Install requirements
pip install -r requirements.txt

# Run quick example
python example.py

# Train baseline GCN (standard approach)
python gcn.py --dataset cora --epochs 200 --runs 5

# Train reconstruction-based GCN (DECK approach)
python deck-gcn.py --dataset cora --num_hops 2 --delete_ratio 0.5 --max_samples 10
```

### Key Parameters for Node Classification

- `--num_hops`: Neighborhood size (1, 2, or 3 hops)
- `--delete_ratio`: Fraction of nodes to delete (0.3 to 0.7)
- `--max_samples`: Number of subgraphs per node (5 to 20)

See `node_classification/README.md` for detailed documentation.

## Key Differences: Graph-Level vs Node-Level

| Aspect | Graph-Level | Node-Level |
|--------|------------|------------|
| **Input** | Entire graph | k-hop neighborhood per node |
| **Subgraphs** | Delete nodes from graph | Delete nodes from neighborhood |
| **Aggregation** | One embedding per graph | One embedding per node |
| **Prediction** | Graph label | Node label |
| **Datasets** | Synthetic cycles, CSL | Cora, CiteSeer, PubMed, etc. |

## Model Architectures

Both implementations support multiple GNN architectures:

- **GCN** (Graph Convolutional Networks)
- **GIN/GINE** (Graph Isomorphism Networks with Edge features)
- **PNA** (Principal Neighborhood Aggregation) - graph-level only

## Requirements

```bash
pip install torch>=1.13.0
pip install torch-geometric>=2.3.0
pip install numpy>=1.21.0
pip install tqdm>=4.64.0
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{reconstruction-gnns,
  title={Graph Neural Networks with the Reconstruction Conjecture},
  note={Extended with node classification}
}
```

## License

See individual directories for specific licensing information.
