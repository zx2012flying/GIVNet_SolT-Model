import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter_mean, scatter_add, scatter_std, scatter_max
import random
from torch_geometric.nn import (GATConv, SAGPooling, LayerNorm, global_add_pool, Set2Set, global_mean_pool, NNConv, GlobalAttention, GINEConv)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AtomAttention(nn.Module):
    def __init__(self, feature_dim):
        super(AtomAttention, self).__init__()
        self.attn = nn.Linear(feature_dim * 2, 1)

    def forward(self, solute_features, solvent_features):

        solute_expanded = solute_features.unsqueeze(1).expand(-1, solvent_features.size(0), -1)
        solvent_expanded = solvent_features.unsqueeze(0).expand(solute_features.size(0), -1, -1)
        combined_features = torch.cat((solute_expanded, solvent_expanded), dim=-1)
        attention_logits = self.attn(combined_features).squeeze(-1)
        attention_weights = F.softmax(attention_logits, dim=-1)

        return attention_weights


class MolecularAttention(nn.Module):
    def __init__(self, mol_feature_dim):
        super(MolecularAttention, self).__init__()
        self.attn = nn.Linear(mol_feature_dim * 2, 1)

    def forward(self, solute_mol_features, solvent_mol_features):

        solute_expanded = solute_mol_features.unsqueeze(1).expand(-1, solvent_mol_features.size(0), -1)
        solvent_expanded = solvent_mol_features.unsqueeze(0).expand(solute_mol_features.size(0), -1, -1)
        combined_features = torch.cat((solute_expanded, solvent_expanded), dim=-1)
        attention_logits = self.attn(combined_features).squeeze(-1)
        attention_weights = F.softmax(attention_logits, dim=-1)

        return attention_weights


class GatherModel(nn.Module):
    def __init__(self, is_bipartite=False):
        super(GatherModel, self).__init__()
        self.num_layer  = 4
        self.drop_ratio = 0.0
        self.conv_dim   = 49
        self.JK         = 'sum'
        self.neurons_message = 3
        self.device     = device
        self.node_dim   = 10
        self.edge_dim   = 10
        self.convs       = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        conv_dim = 49

        self.linatoms = nn.Linear(56, conv_dim).to(device)

        # GNN layers
        for layer in range(self.num_layer):
            neurons_message = self.neurons_message
            mes_nn = nn.Sequential(
                nn.Linear(self.edge_dim, neurons_message),
                nn.ReLU(),
                nn.Linear(neurons_message, conv_dim**2)
            ).to(device)
            self.convs.append(NNConv(conv_dim, conv_dim, mes_nn, aggr='mean').to(device))

            self.batch_norms.append(nn.BatchNorm1d(conv_dim).to(device))

    def forward(self, batched_data):
        if batched_data.x is None:
            batched_data.x = torch.zeros(batched_data.num_nodes, self.node_dim).to(self.device)
        if batched_data.edge_attr is None:
            batched_data.edge_attr = torch.zeros(batched_data.edge_index.size(1), self.edge_dim).to(self.device)
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        x = F.leaky_relu(self.linatoms(x))

        # GNN layers
        x_list = [x]
        for layer in range(self.num_layer):
            x = self.convs[layer](x_list[layer], edge_index, edge_attr)
            x = self.batch_norms[layer](x)

            # Remove activation function from last layer
            if layer == self.num_layer - 1 and False:
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.leaky_relu(x), self.drop_ratio, training=self.training)
            x_list.append(x)

        if self.JK == "last":
            x = x_list[-1]
        elif self.JK == "sum":
            x = 0
            for layer in range(self.num_layer):
                x += x_list[layer]
        elif self.JK == "mean":
            x = 0
            for layer in range(self.num_layer):
                x += x_list[layer]
            x = x / self.num_layer
        return x



class PISGNN(nn.Module):
    def __init__(self,
                 node_input_dim=56,
                 edge_input_dim=10,
                 node_hidden_dim=49,
                 edge_hidden_dim=56,
                 num_step_message_passing=6,
                 num_step_set2_set=4,
                 mlp_layers = 3,

                 mlp_dims = [105, 74, 1]):
        super(PISGNN, self).__init__()

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing

        self.solute_gather = GatherModel().to(device)
        self.solvent_gather = GatherModel().to(device)

        self.num_step_set2set = num_step_set2_set

        self.neg_predictor = nn.Linear(self.node_hidden_dim * 2, 1)

        self.rand_predictor = nn.Linear(self.node_hidden_dim * 10, 1)

        self.set2set_pos_solute = Set2Set(self.node_hidden_dim * 2, self.num_step_set2set)

        self.init_model()
        self.dropout = nn.Dropout(p=0.05)
        self.mlp_layers = mlp_layers

        mlp_layer_1 = self.node_hidden_dim * 8
        mlp_dims_complete = [mlp_layer_1] + mlp_dims

        # MLP for A
        self.mlp_a = nn.ModuleList()
        self.batch_norms_mlp_a = nn.ModuleList()
        for layer in range(self.mlp_layers):
            self.mlp_a.append(nn.Linear(mlp_dims_complete[layer], mlp_dims_complete[layer + 1]))
            self.batch_norms_mlp_a.append(nn.BatchNorm1d(mlp_dims_complete[layer + 1]))
        self.mlp_final_a = nn.Linear(mlp_dims_complete[-1], 1)

        # MLP for B
        self.mlp_b = nn.ModuleList()
        self.batch_norms_mlp_b = nn.ModuleList()
        for layer in range(self.mlp_layers):
            self.mlp_b.append(nn.Linear(mlp_dims_complete[layer], mlp_dims_complete[layer + 1]))
            self.batch_norms_mlp_b.append(nn.BatchNorm1d(mlp_dims_complete[layer + 1]))
        self.mlp_final_b = nn.Linear(mlp_dims_complete[-1], 1)

        self.set2set_solute = Set2Set(self.node_hidden_dim * 2, self.num_step_set2set)
        self.set2set_solvent = Set2Set(self.node_hidden_dim * 2, self.num_step_set2set)
        self.set2set_solute_2d = Set2Set(self.node_hidden_dim * 2, self.num_step_set2set)
        self.set2set_solvent_2d = Set2Set(self.node_hidden_dim * 2, self.num_step_set2set)

        self.compressor = nn.Sequential(
            nn.Linear(self.node_hidden_dim * 2, self.node_hidden_dim),
            nn.BatchNorm1d(self.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.node_hidden_dim, 1))
        self.imap = nn.Linear(2 * node_hidden_dim, 1)


        self.tau = 1.0

        self.solute_update_mlp = nn.Sequential(
            nn.Linear(self.node_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 49))
        self.solvent_update_mlp = nn.Sequential(
            nn.Linear(self.node_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 49))

        self.solute_atom_update_mlp = nn.Sequential(
            nn.Linear(self.node_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 49))
        self.solvent_atom_update_mlp = nn.Sequential(
            nn.Linear(self.node_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 49))

        self.atom_attn = AtomAttention(feature_dim=self.node_hidden_dim)
        self.mol_attn = MolecularAttention(mol_feature_dim=self.node_hidden_dim)
        self.EM_num = 2


    def compute_a(self, x):
        A = x
        for layer in range(self.mlp_layers):
            A = self.mlp_a[layer](A)
            if layer != self.mlp_layers - 1:
                A = self.batch_norms_mlp_a[layer](A)
                A = F.leaky_relu(A)
                A = self.dropout(A)
        A = self.mlp_final_a(A)
        return A

    def compute_b(self, x):
        B = x
        for layer in range(self.mlp_layers):
            B = self.mlp_b[layer](B)
            if layer != self.mlp_layers - 1:
                B = self.batch_norms_mlp_b[layer](B)
                B = F.leaky_relu(B)
                B = self.dropout(B)
        B = self.mlp_final_b(B)
        return B

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def compress(self, solute_features):

        p = self.compressor(solute_features)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()

        return gate_inputs, p


    def interaction(self, solute_features, solvent_features, solute_len, solvent_len,solute, solvent):

        # --- Initializations ---
        solute_features = F.normalize(solute_features, dim=1)
        solvent_features = F.normalize(solvent_features, dim=1)

        len_map = torch.mm(solute_len.t(), solvent_len).to_dense().float()

        # Get initial molecular features by averaging atomic features
        solute_mol_features = scatter_mean(solute_features, solute.batch, dim=0)
        solvent_mol_features = scatter_mean(solvent_features, solvent.batch, dim=0)

        # --- EM Iteration Loop ---
        for em_iter in range(self.EM_num):
            with torch.no_grad():

                X1 = solute_features.unsqueeze(0)
                Y1 = solvent_features.unsqueeze(1)
                X2 = X1.repeat(solvent_features.shape[0], 1, 1)
                Y2 = Y1.repeat(1, solute_features.shape[0], 1)
                Z = torch.cat([X2, Y2], -1)
                atom_sim_matrix = torch.tanh(self.imap(Z)).squeeze(2).t()

                X1 = solute_mol_features.unsqueeze(0)
                Y1 = solvent_mol_features.unsqueeze(1)
                X2 = X1.repeat(solvent_mol_features.shape[0], 1, 1)
                Y2 = Y1.repeat(1, solute_mol_features.shape[0], 1)
                Z = torch.cat([X2, Y2], -1)
                mol_sim_matrix = torch.tanh(self.imap(Z)).squeeze(2).t()

                solute_batch_idx = solute.batch.unsqueeze(1).expand(-1, solvent.batch.size(0))
                solvent_batch_idx = solvent.batch.unsqueeze(0).expand(solute.batch.size(0), -1)
                expanded_mol_sim = mol_sim_matrix[solute_batch_idx, solvent_batch_idx]

                atom_attn = self.atom_attn(solute_features, solvent_features)
                mol_attn_matrix = self.mol_attn(solute_mol_features,solvent_mol_features)
                expanded_mol_attn = mol_attn_matrix[solute_batch_idx, solvent_batch_idx]
                combined_sim = atom_attn * atom_sim_matrix + expanded_mol_attn * expanded_mol_sim

                interaction_prob = combined_sim * len_map
                interaction_prob = F.softmax(interaction_prob, dim=1)

            solute_prime_from_solvent = torch.matmul(interaction_prob, solvent_features)
            solvent_prime_from_solute = torch.matmul(interaction_prob.t(), solute_features)

            mol_interaction_prob = F.softmax(mol_sim_matrix, dim=1)
            solute_mol_prime = torch.matmul(mol_interaction_prob, solvent_mol_features)
            solvent_mol_prime = torch.matmul(mol_interaction_prob.t(), solute_mol_features)

            solute_features = solute_features + self.solute_atom_update_mlp(solute_prime_from_solvent)
            solvent_features = solvent_features + self.solvent_atom_update_mlp(solvent_prime_from_solute)

            solute_mol_features = solute_mol_features +self.solute_update_mlp(solute_mol_prime)
            solvent_mol_features = solvent_mol_features + self.solvent_update_mlp(solvent_mol_prime)

            solute_features = F.normalize(solute_features, dim=1)
            solvent_features = F.normalize(solvent_features, dim=1)
            solute_mol_features = F.normalize(solute_mol_features, dim=1)
            solvent_mol_features = F.normalize(solvent_mol_features, dim=1)

        expanded_solute_mol_final = solute_mol_features[solute.batch]
        expanded_solvent_mol_final = solvent_mol_features[solvent.batch]

        solute_final = torch.cat([
            solute_features,
            expanded_solute_mol_final
        ], dim=1)

        solvent_final = torch.cat([
            solvent_features,
            expanded_solvent_mol_final
        ], dim=1)

        return solute_final, solvent_final, interaction_prob


    def forward(self, data):
        solute = data[0]
        solvent = data[1]
        solute_len = data[2]
        solvent_len = data[3]
        tm = data[4]

        _solute_features = self.solute_gather(solute)
        _solvent_features = self.solvent_gather(solvent)

        solute_features, solvent_features, ret_interaction_map = self.interaction(_solute_features, _solvent_features, solute_len,
                                                             solvent_len, solute, solvent )

        # ====================================================================
        # Causal Disentanglement
        # ====================================================================
        lambda_pos, _ = self.compress(solute_features)
        lambda_pos = lambda_pos.reshape(-1, 1)
        lambda_neg = 1 - lambda_pos

        shortcut_solute_features = lambda_neg * solute_features

        solute_mean = scatter_mean(solute_features.detach(), solute.batch, dim=0)[solute.batch]
        causal_solute_features = lambda_pos * solute_features + lambda_neg * solute_mean


        causal_solute_s2s = self.set2set_pos_solute(causal_solute_features, solute.batch)
        shortcut_solute_s2s = global_mean_pool(shortcut_solute_features, solute.batch)
        solvent_s2s = self.set2set_solvent(solvent_features, solvent.batch)

        causal_pair_features = torch.cat((causal_solute_s2s, solvent_s2s), 1)
        A = self.compute_a(causal_pair_features)
        B = self.compute_b(causal_pair_features)
        causal_predictions = A + B / tm

        neg_predictions = self.neg_predictor(shortcut_solute_s2s)

        batch_size = causal_solute_s2s.shape[0]
        random_indices = torch.randperm(batch_size)
        random_shortcut_solute_s2s = shortcut_solute_s2s[random_indices]
        random_intervention_features = torch.cat((causal_solute_s2s, solvent_s2s, random_shortcut_solute_s2s), 1)
        random_predictions = self.rand_predictor(random_intervention_features)


        return causal_predictions, A, B,  neg_predictions, random_predictions, ret_interaction_map

