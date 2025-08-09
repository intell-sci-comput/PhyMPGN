import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from src.utils.utils import build_net
from src.utils.padding import h_padding


class MPNNBlock(nn.Module):
    def __init__(self, mpnn_layers, mpnn_num):
        super(MPNNBlock, self).__init__()
        self.phi_layers = mpnn_layers[0]
        self.gamma_layers = mpnn_layers[1]
        self.mpnn_num = mpnn_num
        self.nets = self.build_block()

    def build_block(self):
        nets = nn.ModuleList()
        for i in range(self.mpnn_num):
            nets.append(MPNNLayer(self.phi_layers, self.gamma_layers))
        return nets

    def forward(self, graph):
        """
        :param graph: Data(y=[bxn, 2], pos=[bxn, 2], edge_index=[2, bxn], batch=[bxn])
        :return:
        """
        h = graph.state_node
        for mpnn in self.nets[:-1]:
            h = h + mpnn(h, graph.state_edge, graph.edge_index)
            # padding
            h = h_padding(h, graph)

        h = self.nets[-1](h, graph.state_edge, graph.edge_index)
        return h  # [bxn, features]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class MPNNLayer(MessagePassing):
    def __init__(self, phi_layers, gamma_layers):
        super(MPNNLayer, self).__init__(aggr='mean', flow='target_to_source')
        self.phi = build_net(phi_layers)
        self.gamma = build_net(gamma_layers)

    def forward(self, hidden_node, hidden_edge, edge_index):
        return self.propagate(edge_index, h=hidden_node, hidden_edge=hidden_edge)

    def message(self, h_i, h_j, hidden_edge):
        phi_input = torch.cat([h_i, h_j-h_i, hidden_edge], dim=1)  # (e, h_features*2 + hidden_edge_features)
        return self.phi(phi_input)  # (e, phi_out_features)

    def update(self, aggr, h):
        gamma_input = torch.cat([h, aggr], dim=1)  # (num_nodes, h_features + phi_out_features)
        out = self.gamma(gamma_input)  # (num_nodes, gamma_out_features)
        return out

    def __str__(self):
        return '(phi): \n{}\n(gamma):\n{}'.format(self.phi.__str__(), self.gamma.__str__())