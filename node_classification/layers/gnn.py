import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, Dropout
from torch_geometric.nn import global_mean_pool, MessagePassing, BatchNorm
from torch_geometric.utils import degree


class GCNConv(MessagePassing):
    """Graph Convolutional Network layer with edge attributes."""
    def __init__(self, emb_dim, dim1, dim2):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(dim1, dim2)
        self.root_emb = torch.nn.Embedding(1, dim2)
        self.bond_encoder = Sequential(Linear(emb_dim, dim2), nn.ReLU(), Linear(dim2, dim2))

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + \
               F.relu(x + self.root_emb.weight) * 1./deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GINConv(MessagePassing):
    """Graph Isomorphism Network layer with edge attributes."""
    def __init__(self, emb_dim, dim1, dim2):
        super(GINConv, self).__init__(aggr="add")
        self.bond_encoder = Sequential(Linear(emb_dim, dim1), nn.ReLU(), Linear(dim1, dim1))
        self.mlp = Sequential(Linear(dim1, dim1), nn.ReLU(), Linear(dim1, dim2))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class NodeGCN(torch.nn.Module):
    """
    Baseline GCN for node classification.
    Processes the full graph and outputs node-level predictions.
    """
    def __init__(self, node_size, edge_size, hidden_size, out_size, num_layers):
        super(NodeGCN, self).__init__()

        self.convs = ModuleList([GCNConv(edge_size, node_size, hidden_size)])
        self.batch_norms = ModuleList([BatchNorm(hidden_size)])
        for _ in range(num_layers - 1):
            conv = GINConv(edge_size, hidden_size, hidden_size)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_size))

        # Node-level prediction layers
        self.fc1 = Linear(num_layers * hidden_size, hidden_size)
        self.fc2 = Linear(hidden_size, out_size)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for node classification.

        Args:
            x: Node features [num_nodes, node_size]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_size]

        Returns:
            Node predictions [num_nodes, out_size]
        """
        x_lst = []

        for conv, bn in zip(self.convs, self.batch_norms):
            x = F.relu(conv(x, edge_index, edge_attr))
            x = bn(x)
            x_lst.append(x)

        # Concatenate all layer outputs
        x = torch.cat(x_lst, dim=-1)

        # Node-level prediction (no pooling)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)


class NodeReconstructionGCN(torch.nn.Module):
    """
    Reconstruction-based GCN for node classification.

    For each node:
    1. Extract 2-hop neighborhood
    2. Create multiple node-deleted subgraphs
    3. Process each subgraph with GNN
    4. Aggregate subgraph representations
    5. Predict node label
    """
    def __init__(self, node_size, edge_size, hidden_size, out_size, num_layers):
        super(NodeReconstructionGCN, self).__init__()

        self.non_linearity = nn.ReLU()

        # GNN layers
        self.convs = ModuleList([GCNConv(edge_size, node_size, hidden_size)])
        self.batch_norms = ModuleList([BatchNorm(hidden_size)])
        for _ in range(num_layers - 1):
            conv = GINConv(edge_size, hidden_size, hidden_size)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_size))

        # Subgraph-level processing
        self.fc0 = Linear(hidden_size * num_layers, hidden_size)

        # Node-level prediction after aggregation
        self.fc1 = Linear(hidden_size, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size)
        self.pred = Linear(hidden_size, out_size)

    def forward(self, x, edge_index, edge_attr, batch, weights, subgraph_batch):
        """
        Forward pass for reconstruction-based node classification.

        Args:
            x: Node features of all subgraphs [total_nodes, node_size]
            edge_index: Edge indices of all subgraphs [2, total_edges]
            edge_attr: Edge attributes of all subgraphs [total_edges, edge_size]
            batch: Batch assignment for nodes in subgraphs [total_nodes]
            weights: Weights for each subgraph [num_subgraphs, hidden_size]
            subgraph_batch: Batch assignment mapping subgraphs to nodes [num_subgraphs]

        Returns:
            Node predictions [num_nodes, out_size]
        """
        x_lst = []

        # Apply GNN layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = F.relu(conv(x, edge_index, edge_attr))
            x = bn(x)
            x_lst.append(x)

        # Concatenate all layer outputs
        x = torch.cat(x_lst, dim=-1)

        # Pool to subgraph-level representations
        x = global_mean_pool(x, batch)

        # Process subgraph representations
        x = self.non_linearity(self.fc0(x))

        # Weight each subgraph
        x = x * weights

        # Aggregate subgraphs to node-level (mean aggregation)
        from torch_geometric.nn import global_add_pool
        x = global_add_pool(x, subgraph_batch)
        norm = global_add_pool(weights, subgraph_batch)
        x = x / norm  # Mean aggregation

        # Node-level prediction
        x = self.non_linearity(self.fc1(x))
        x = self.non_linearity(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.pred(x)


class NodeGINE(torch.nn.Module):
    """
    Baseline GIN for node classification.
    """
    def __init__(self, node_size, edge_size, hidden_size, out_size, num_layers):
        super(NodeGINE, self).__init__()

        self.convs = ModuleList([GINConv(edge_size, node_size, hidden_size)])
        self.batch_norms = ModuleList([BatchNorm(hidden_size)])
        for _ in range(num_layers - 1):
            conv = GINConv(edge_size, hidden_size, hidden_size)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_size))

        self.fc1 = Linear(num_layers * hidden_size, hidden_size)
        self.fc2 = Linear(hidden_size, out_size)

    def forward(self, x, edge_index, edge_attr):
        x_lst = []

        for conv, bn in zip(self.convs, self.batch_norms):
            x = F.relu(conv(x, edge_index, edge_attr))
            x = bn(x)
            x_lst.append(x)

        x = torch.cat(x_lst, dim=-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)


class NodeReconstructionGINE(torch.nn.Module):
    """
    Reconstruction-based GIN for node classification.
    """
    def __init__(self, node_size, edge_size, hidden_size, out_size, num_layers):
        super(NodeReconstructionGINE, self).__init__()

        self.non_linearity = nn.ReLU()

        self.convs = ModuleList([GINConv(edge_size, node_size, hidden_size)])
        self.batch_norms = ModuleList([BatchNorm(hidden_size)])
        for _ in range(num_layers - 1):
            conv = GINConv(edge_size, hidden_size, hidden_size)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_size))

        self.fc0 = Linear(hidden_size * num_layers, hidden_size)
        self.fc1 = Linear(hidden_size, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size)
        self.pred = Linear(hidden_size, out_size)

    def forward(self, x, edge_index, edge_attr, batch, weights, subgraph_batch):
        x_lst = []

        for conv, bn in zip(self.convs, self.batch_norms):
            x = F.relu(conv(x, edge_index, edge_attr))
            x = bn(x)
            x_lst.append(x)

        x = torch.cat(x_lst, dim=-1)
        x = global_mean_pool(x, batch)
        x = self.non_linearity(self.fc0(x))

        x = x * weights
        from torch_geometric.nn import global_add_pool
        x = global_add_pool(x, subgraph_batch)
        norm = global_add_pool(weights, subgraph_batch)
        x = x / norm

        x = self.non_linearity(self.fc1(x))
        x = self.non_linearity(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.pred(x)
