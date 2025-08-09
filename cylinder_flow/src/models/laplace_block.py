import torch
import torch.nn as nn

from .mpnn_block import MPNNLayer


class LaplaceBlock(nn.Module):
    def __init__(self, enc_dim, h_dim, out_dim):
        super(LaplaceBlock, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(enc_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim)
        )
        self.processor = LaplaceProcessor(
            mpnn_layers=[
                [h_dim * 2 + 3, h_dim, h_dim],
                [h_dim * 2, h_dim, h_dim]
            ],
            mpnn_num=3
        )
        self.decoder = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, out_dim)
        )

    def cal_mesh_laplace(self, graph):
        laplace_matrix = torch.block_diag(*graph.laplace_matrix.unbind())
        laplace = laplace_matrix @ graph.y  # (bn, 2)
        return laplace

    def forward(self, graph):
        h = self.encoder(torch.cat((graph.y, graph.pos), dim=-1))
        edge_attr = graph.edge_attr[:, :3]
        h = self.processor(h, edge_attr, graph.edge_index)
        out = self.decoder(h)
        out = graph.d_vector * out

        out = out + self.cal_mesh_laplace(graph)
        return out
        # return self.cal_mesh_laplace(graph)

    def count_parameters(self):
        return sum([param.nelement() for param in self.parameters()])


class LaplaceProcessor(nn.Module):
    def __init__(self, mpnn_layers, mpnn_num):
        super(LaplaceProcessor, self).__init__()
        self.phi_layers = mpnn_layers[0]
        self.gamma_layers = mpnn_layers[1]
        self.mpnn_num = mpnn_num
        self.nets = self.build_block()

    def build_block(self):
        nets = nn.ModuleList()
        for i in range(self.mpnn_num):
            nets.append(MPNNLayer(self.phi_layers, self.gamma_layers))
        return nets

    def forward(self, h, edge_attr, edge_index):
        """
        :param graph: Data(y=[bxn, 2], pos=[bxn, 2], edge_index=[2, bxn], batch=[bxn])
        :return:
        """
        for mpnn in self.nets:
            h = h + mpnn(h, edge_attr, edge_index)
        return h  # [bxn, features]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
