import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter_mean, scatter_add, scatter_std, scatter_max
import random
from torch_geometric.nn import (GATConv, SAGPooling, LayerNorm, global_add_pool, Set2Set, global_mean_pool, NNConv, GlobalAttention, GINEConv,  GINConv)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GatherModel(nn.Module):
    """
    基于 GIN 的分子编码器
    """

    def __init__(self,
                 node_input_dim=56,
                 node_hidden_dim=49,
                 num_layers=3,
                 dropout=0.0,
                 output_type='graph'):
        super(GatherModel, self).__init__()

        self.num_layers = num_layers
        self.output_type = output_type
        self.dropout = dropout

        # 初始线性变换（可选，但推荐）
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)

        # GIN 层：每一层包含一个 MLP + GINConv
        self.gin_layers = nn.ModuleList()
        for i in range(num_layers):
            # 每个 GIN 层的 MLP：通常 2 层 MLP
            mlp = nn.Sequential(
                nn.Linear(node_hidden_dim, node_hidden_dim),
                nn.BatchNorm1d(node_hidden_dim),
                nn.ReLU(),
                nn.Linear(node_hidden_dim, node_hidden_dim),
                nn.BatchNorm1d(node_hidden_dim),
                nn.ReLU()
            )
            self.gin_layers.append(GINConv(mlp, train_eps=True))

        # 全局池化（图级别表示）
        if output_type == 'graph':
            self.pooling = global_mean_pool  # 也可换为 global_add_pool / Set2Set

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, g):
        """
        参数:
            g: PyG Data or Batch 对象
        返回:
            图级或节点级嵌入
        """
        x, edge_index, batch = g.x, g.edge_index, g.batch

        # 初始投影
        x = self.lin0(x)  # [N, hidden_dim]

        # 多层 GIN 传播
        for gin in self.gin_layers:
            x = gin(x, edge_index)
            x = self.dropout_layer(x)
            x = F.relu(x)

        if self.output_type == 'node':
            return x

        elif self.output_type == 'graph':
            graph_emb = self.pooling(x, batch)  # [batch_size, node_hidden_dim]
            return graph_emb


class PISGNN(nn.Module):
    def __init__(self,
                 node_input_dim=56,
                 node_hidden_dim=49,
                 mlp_layers = 3,
                 mlp_dims = [105, 74, 1],
                 num_heads=4,
                 gat_layers=3,
                 final_dropout=0.05
                 ):
        super(PISGNN, self).__init__()

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.solute_gather = GatherModel().to(device)
        self.solvent_gather = GatherModel().to(device)

        self.solute_gather = GatherModel(
            node_input_dim=node_input_dim,
            node_hidden_dim=node_hidden_dim,
            num_layers=gat_layers,  # 这里其实是 GIN 层数
            dropout=final_dropout,
            output_type='graph'
        ).to(device)

        self.solvent_gather = GatherModel(
            node_input_dim=node_input_dim,
            node_hidden_dim=node_hidden_dim,
            num_layers=gat_layers,
            dropout=final_dropout,
            output_type='graph'
        ).to(device)

        self.init_model()
        self.dropout = nn.Dropout(p=0.05)
        self.mlp_layers = mlp_layers
        mlp_layer_1 = self.node_hidden_dim * 2 + 1
        mlp_dims_complete = [mlp_layer_1] + mlp_dims

        # MLP for A
        self.mlp_a = nn.ModuleList()
        self.batch_norms_mlp_a = nn.ModuleList()
        for layer in range(self.mlp_layers):
            self.mlp_a.append(nn.Linear(mlp_dims_complete[layer], mlp_dims_complete[layer + 1]))
            self.batch_norms_mlp_a.append(nn.BatchNorm1d(mlp_dims_complete[layer + 1]))
        self.mlp_final_a = nn.Linear(mlp_dims_complete[-1], 1)
        self.num_step_set2set = 4
        self.set2set_pos_solute = Set2Set(196, self.num_step_set2set)


    def compute_a(self, x):
        A = x
        for layer in range(self.mlp_layers):
            A = self.mlp_a[layer](A)
            if layer != self.mlp_layers - 1:
                A = self.batch_norms_mlp_a[layer](A)
                A = F.leaky_relu(A)
                A = self.dropout(A)
        return A


    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, data):
        solute = data[0]
        solvent = data[1]
        solute_len = data[2]
        solvent_len = data[3]
        tm = data[4].float()

        solute.x = solute.x.float()
        solvent.x = solvent.x.float()
        if solute.edge_attr is not None:
            solute.edge_attr = solute.edge_attr.float()
        if solvent.edge_attr is not None:
            solvent.edge_attr = solvent.edge_attr.float()

        _solute_features = self.solute_gather(solute)
        _solvent_features = self.solvent_gather(solvent)
        # _solute_features = self.set2set_pos_solute(_solute_features, solute.batch)
        # _solvent_features= self.set2set_pos_solute( _solvent_features, solvent.batch)
        final_features = torch.cat((_solute_features, _solvent_features, tm), 1)
        pred = self.compute_a(final_features)

        return pred

