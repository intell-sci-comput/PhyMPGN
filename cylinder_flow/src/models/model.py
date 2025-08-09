import torch
import torch.nn as nn

from .mpnn_block import MPNNBlock
from .encoder_decoder import Encoder, Decoder
from src.utils.padding import graph_padding
from src.datasets.data import Graph
from src.models.laplace_block import LaplaceBlock


class Model(nn.Module):
    def __init__(self, encoder_config, mpnn_block_config, decoder_config,
                 laplace_block_config, dtype, device, integral):
        super(Model, self).__init__()
        self.dtype = dtype
        self.device = device

        self.node_encoder = Encoder(encoder_config['node_encoder_layers'])
        self.edge_encoder = Encoder(encoder_config['edge_encoder_layers'])
        self.mpnn_block = MPNNBlock(
            mpnn_layers=mpnn_block_config['mpnn_layers'],
            mpnn_num=mpnn_block_config['mpnn_num']
        )
        self.decoder = Decoder(decoder_config['node_decoder_layers'])
        self.laplace_block = LaplaceBlock(
            enc_dim=laplace_block_config['in_dim'],
            h_dim=laplace_block_config['h_dim'],
            out_dim=laplace_block_config['out_dim']
        )

        update_fn = {
            1: self.update_euler,
            2: self.update_rk2,
            4: self.update_rk4
        }
        self.update = update_fn[integral]

    def forward(self, graph, steps):
        """

        Args:
            graph (Graph): Graph(y=(bn, d), pos=(bn, 2), edge_index=(2, be),
                laplace_matrix=(b, n, n))
            steps (int): steps of roll-out
        Returns:

        """
        loss_states = [graph.y] # [bn, 2]
        # unroll for 1 step
        graph_next = self.update(graph)
        loss_states.append(graph_next.y)

        graph = graph_next.detach()
        # graph = graph_next
        # unroll for steps-1
        for step in range(steps - 1):
            graph_next = self.update(graph)
            loss_states.append(graph_next.y)
            graph = graph_next

        # [t, bn, 2]
        loss_states = torch.stack(loss_states, dim=0)
        return torch.index_select(loss_states, 1, graph.truth_index)

    def get_temporal_diff(self, graph):
        """

        Args:
            graph (Graph): Graph(y=(bn, d), pos=(bn, 2), edge_index=(2, be),
                laplace_matrix=(b, n, n))

        Returns:

        """
        # (bn, 2+2+1+4) -> (bn, h)
        node_type = torch.nn.functional.one_hot(graph.node_type)
        graph.state_node = self.node_encoder(
            torch.cat((graph.y, graph.pos, node_type), dim=-1))
        # store dirichlet value
        if hasattr(graph, 'dirichlet_index'):
            graph.dirichlet_h_value = torch.index_select(
                graph.state_node, 0, graph.dirichlet_index)
            graph.inlet_h_value = torch.index_select(
                graph.state_node, 0, graph.inlet_index)

        rel_state = graph.y[graph.edge_index[1, :]] - \
                    graph.y[graph.edge_index[0, :]]  # (be, 2)
        # (be, 5) -> (be, h)
        graph.state_edge = self.edge_encoder(
            torch.cat((rel_state, graph.edge_attr), dim=-1))
        mpnn_out = self.mpnn_block(graph)  # (bn, h)
        decoder_out = self.decoder(mpnn_out)  # (bn, 2)

        # laplace
        laplace = self.laplace_block(graph) # (bn, 2)

        # gt_e5_index = laplace > 1.0e+3
        # gt_e5 = torch.sum(gt_e5_index, dim=0)
        # le_e5_index = laplace < -1.0e+4
        # le_e5 = torch.sum(le_e5_index, dim=0)
        # gt_pos_u = graph.pos[gt_e5_index[:, 0]]
        # flag_u = on_boundary(gt_pos_u)
        # flag_u_true = torch.sum(flag_u)
        # gt_pos_v = graph.pos[gt_e5_index[:, 1]]
        # flag_v = on_boundary(gt_pos_v)
        # flag_v_true = torch.sum(flag_v)
        #
        # plot_meshcolor(
        #     U=laplace.detach().cpu().numpy(),
        #     pos=graph.pos.detach().cpu().numpy(),
        #     tri=graph.face.permute(1, 0).detach().cpu().numpy()
        # )
        # fig, ax = plt.subplots(2, 1)
        # ax[0].scatter(graph.pos[gt_e5_index[:, 0], 0].detach().cpu().numpy(),
        #               graph.pos[gt_e5_index[:, 0], 1].detach().cpu().numpy())
        # ax[0].set_title('u laplace gt e+3')
        # ax[1].scatter(graph.pos[gt_e5_index[:, 1], 0].detach().cpu().numpy(),
        #               graph.pos[gt_e5_index[:, 1], 1].detach().cpu().numpy())
        # ax[1].set_title('v laplace gt e+3')
        # ax[0].set_xlim(0, 16)
        # ax[0].set_ylim(0, 8)
        # ax[0].set_aspect('equal')
        # ax[1].set_xlim(0, 16)
        # ax[1].set_ylim(0, 8)
        # ax[1].set_aspect('equal')
        # plt.show()

        # (bn, 1) * (bn, 2) + (bn, 2) -> (bn, 2)
        u_m, rho, D, mu = graph.u_m, graph.rho, graph.r * 2, graph.mu  # (bn, 1)
        Re = rho * D * u_m / mu  # (bn, 1)
        out = 1 / Re * laplace + decoder_out
        # out = decoder_out
        # laplace_min = laplace.min()
        # laplace_max = laplace.max()
        # laplace_mean = laplace.mean()
        # tmp_min = tmp.min()
        # tmp_max = tmp.max()
        # deco_out_min = decoder_out.min()
        # deco_out_max = decoder_out.max()
        # deco_out_mean = decoder_out.mean()
        # out_min = out.min()
        # out_max = out.max()
        # out = self.mu * laplace + decoder_out
        return out


    def update_euler(self, graph):
        out = self.get_temporal_diff(graph)
        graph.y = graph.y + out * graph.dt
        # padding
        graph_padding(graph)

        return graph

    def update_rk2(self, graph):

        U0 = graph.y
        K1 = self.get_temporal_diff(graph)  # (bn, 2)
        U1 = U0 + K1 * graph.dt  # (bn, 2) + (bn, 2) * (bn, 1) -> (bn, 2)
        graph.y = U1
        # padding
        graph_padding(graph)

        K2 = self.get_temporal_diff(graph)
        graph.y = U0 + K1 * graph.dt / 2 + K2 * graph.dt / 2
        # padding
        graph_padding(graph)

        return graph

    def update_rk4(self, graph):
        # stage 1
        U0 = graph.y
        K1 = self.get_temporal_diff(graph)

        # stage 2
        U1 = U0 + K1 * graph.dt / 2.
        graph.y = U1
        # padding
        graph_padding(graph)
        K2 = self.get_temporal_diff(graph)

        # stage 3
        U2 = U0 + K2 * graph.dt / 2.
        graph.y = U2
        # padding
        graph_padding(graph)
        K3 = self.get_temporal_diff(graph)

        # stage 4
        U3 = U0 + K3 * graph.dt
        graph.y = U3
        # padding
        graph_padding(graph)
        K4 = self.get_temporal_diff(graph)

        U4 = U0 + (K1 + 2 * K2 + 2 * K3 + K4) * graph.dt / 6.
        graph.y = U4
        # padding
        graph_padding(graph)

        return graph

    def count_parameters(self):
        total = sum([param.nelement() for param in self.parameters()])
        mpnn = self.mpnn_block.count_parameters()
        laplace = self.laplace_block.count_parameters()
        # laplace = 0
        return total, mpnn, laplace


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.loss_func = nn.MSELoss()

    def forward(self, U_pred, truth, mask=None):
        """
        :param U_pred: [t, n, 2]
        :param truth: [t, n, 2]
        :return:
        """
        pred1 = U_pred[1]  # [bxn, 2]
        predn = U_pred[-1]
        new_pred = torch.stack((pred1, predn), dim=0)

        truth1 = truth[1]
        truthn = truth[-1]
        new_truth = torch.stack((truth1, truthn), dim=0)

        if mask is None:
            return self.loss_func(new_pred, new_truth)
        else:
            return self.loss_func(new_pred[:, mask], new_truth[:, mask])


class AllMSELoss(nn.Module):
    def __init__(self):
        super(AllMSELoss, self).__init__()
        self.loss_func = nn.MSELoss()

    def forward(self, U_pred, truth, mask=None):
        if mask is None:
            return self.loss_func(U_pred, truth)
        else:
            return self.loss_func(U_pred[:, mask], truth[:, mask])