import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.interpolate import CloughTocher2DInterpolator
from matplotlib.animation import FuncAnimation

from src.utils.utils import cal_cur_time_corre, correlation



def plot_origin_graph(graph, bdry_pos, bdry_element):

    fig, ax = plt.subplots()
    for line in bdry_element:
        ax.plot(bdry_pos[line, 0], bdry_pos[line, 1], color='black', linewidth=1)
    ax.set_aspect('equal')
    ax.set_axis_off()

    pos = torch.index_select(graph.pos, 0, graph.truth_index)
    ax.scatter(pos[:, 0], pos[:, 1], s=3, c='black')

    return fig


def plot_mesh(graph):
    fig, ax = plt.subplots()
    tris = graph.face.permute(1, 0)  # (3, n_tri) -> (n_tri, 3)
    plt.triplot(graph.pos[:, 0], graph.pos[:, 1], tris)
    ax.set_aspect('equal')
    return fig


def draw_loss():
    epochs = 1000
    train_idx = np.arange(1, epochs + 1)
    val_idx = np.arange(0, epochs + 1, 10)
    val_idx[0] = 1

    train_loss = np.load('loss/loss90-tr-1.npy')
    val_loss1 = np.load('loss/loss90-val1-1.npy')
    val_loss2 = np.load('loss/loss90-val2-1.npy')

    plt.semilogy(train_idx, train_loss, label='train')
    plt.semilogy(val_idx, val_loss1, label='val old ic')
    plt.semilogy(val_idx, val_loss2, label='val new ic')
    plt.title('Train Loss')
    plt.legend()
    plt.show()


def plot_average_time_correlation(pred_list, truth_list, save_path=None):
    """
    Parameters
    ----------
    pred_list: List, each element is a np.ndarray, shape [t, n, d]
    truth_list: List, each element is a np.ndarray, shape [t, n, d]
    save_path

    Returns
    -------

    """
    corr_data = []
    for i in range(len(pred_list)):
        pred = pred_list[i]
        truth = truth_list[i]
        coef_list = cal_cur_time_corre(pred, truth)
        corr_data.append(np.array(coef_list))

    corr = np.mean(corr_data, axis=0)  # [b, t] -> [t,]
    fig, ax = plt.subplots()
    ax.plot(corr)
    ax.set_xlabel('time step')
    ax.set_title('Correlation')

    if save_path is not None:
        fig.savefig(save_path, dpi=300)

    return corr


def cal_mean_correlation(pred_list, truth_list):
    corr_data = []
    for i in range(len(pred_list)):
        pred = pred_list[i]
        truth = truth_list[i]
        coef = correlation(pred, truth)
        corr_data.append(coef)

    corr = np.mean(corr_data)
    return corr


def plot_meshcolor_evolution(coarse_pos, coarse_U_pred, coarse_U_gt, fine_pos, fine_tri, bdry_pos, bdry_elem, time_step, save_path=None):
    """

    Args:
        coarse_pos (np.ndarray): shape [n_c, two]
        coarse_U_pred (np.ndarray): shape [t, n_c, d]
        coarse_U_gt (np.ndarray): shape [t, n_c, d]
        fine_pos (np.ndarray): shape [n_f, two]
        fine_tri (np.ndarray): shape [n_tri, three], triangles
        dolphin_bdry_pos (np.ndarray): shape [n_d, 2]
        dolphin_bdry_elem (np.ndarray): shape [n_l, 2]
        time_step (int):
        save_path (str:
    """
    t = coarse_U_pred.shape[0]
    rows = 4
    cols = t // time_step + 1
    t_steps = np.arange(0, t, time_step)
    t_steps = np.concatenate((t_steps, [t - 1]))
    fig, ax = plt.subplots(rows, cols, figsize=(11, 4))
    fig.text(0.1, 0.8, 'Ref. u', ha='center', va='center')
    fig.text(0.1, 0.6, 'Ours. u', ha='center', va='center')
    fig.text(0.1, 0.4, 'Ref. v', ha='center', va='center')
    fig.text(0.1, 0.2, 'Ours. v', ha='center', va='center')
    tri = Triangulation(x=fine_pos[:, 0], y=fine_pos[:, 1], triangles=fine_tri)

    for i, t in enumerate(t_steps):
        func_u_gt = CloughTocher2DInterpolator(points=(coarse_pos[:, 0], coarse_pos[:, 1]), values=coarse_U_gt[t, :, 0])
        fine_u_gt = func_u_gt(fine_pos[:, 0], fine_pos[:, 1])  # [n_f,]
        ax[0, i].tripcolor(tri, fine_u_gt, shading='gouraud')
        func_u_pred = CloughTocher2DInterpolator(points=(coarse_pos[:, 0], coarse_pos[:, 1]), values=coarse_U_pred[t, :, 0])
        fine_u_pred = func_u_pred(fine_pos[:, 0], fine_pos[:, 1])  # [n_f,]
        ax[1, i].tripcolor(tri, fine_u_pred, shading='gouraud',
                           vmin=fine_u_gt.min(), vmax=fine_u_gt.max())

        func_v_gt = CloughTocher2DInterpolator(points=(coarse_pos[:, 0], coarse_pos[:, 1]), values=coarse_U_gt[t, :, 1])
        fine_v_gt = func_v_gt(fine_pos[:, 0], fine_pos[:, 1])
        ax[2, i].tripcolor(tri, fine_v_gt, shading='gouraud')
        func_v_pred = CloughTocher2DInterpolator(points=(coarse_pos[:, 0], coarse_pos[:, 1]), values=coarse_U_pred[t, :, 1])
        fine_v_pred = func_v_pred(fine_pos[:, 0], fine_pos[:, 1])  # [n_f,]
        ax[3, i].tripcolor(tri, fine_v_pred, shading='gouraud',
                           vmin=fine_v_gt.min(), vmax=fine_v_gt.max())

        ax[0, i].set_title('t={}'.format(t))

    for i in range(rows):
        for j in range(cols):
            for line in bdry_elem:
                ax[i, j].plot(bdry_pos[line, 0], bdry_pos[line, 1], color='black', linewidth=0.2)
            ax[i, j].set_aspect('equal')
            ax[i, j].set_axis_off()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)

    return fig


def plot_meshcolor_animation(coarse_pos, coarse_U_pred, coarse_U_gt, fine_pos,
                             fine_tri, bdry_pos, bdry_elem, save_path):

    rows = 2
    cols = 2
    fig, ax = plt.subplots(rows, cols, figsize=(6, 3))
    plt.subplots_adjust(bottom=0.01, right=0.99, hspace=0., wspace=0.)
    # fig.text(0.1, 0.7, 'Ref.', fontsize=18, ha='center', va='center')
    # fig.text(0.1, 0.25, 'Ours.', fontsize=18, ha='center', va='center')
    tri = Triangulation(x=fine_pos[:, 0], y=fine_pos[:, 1], triangles=fine_tri)

    def update(t):
        for i in range(rows):
            for j in range(cols):
                ax[i, j].clear()
                ax[i, j].set_aspect('equal')
                for line in bdry_elem:
                    ax[i, j].plot(bdry_pos[line, 0], bdry_pos[line, 1],
                                  color='black', linewidth=0.2)
                ax[i, j].set_axis_off()
                ax[i, j].invert_yaxis()

        func_u_gt = CloughTocher2DInterpolator(
            points=(coarse_pos[:, 0], coarse_pos[:, 1]),
            values=coarse_U_gt[t, :, 0])
        fine_u_gt = func_u_gt(fine_pos[:, 0], fine_pos[:, 1])  # [n_f,]
        ax[0, 0].tripcolor(tri, fine_u_gt, shading='gouraud')
        func_u_pred = CloughTocher2DInterpolator(
            points=(coarse_pos[:, 0], coarse_pos[:, 1]),
            values=coarse_U_pred[t, :, 0])
        fine_u_pred = func_u_pred(fine_pos[:, 0], fine_pos[:, 1])  # [n_f,]
        ax[1, 0].tripcolor(tri, fine_u_pred, shading='gouraud',
                           vmin=fine_u_gt.min(), vmax=fine_u_gt.max())

        func_v_gt = CloughTocher2DInterpolator(
            points=(coarse_pos[:, 0], coarse_pos[:, 1]),
            values=coarse_U_gt[t, :, 1])
        fine_v_gt = func_v_gt(fine_pos[:, 0], fine_pos[:, 1])
        ax[0, 1].tripcolor(tri, fine_v_gt, shading='gouraud')
        func_v_pred = CloughTocher2DInterpolator(
            points=(coarse_pos[:, 0], coarse_pos[:, 1]),
            values=coarse_U_pred[t, :, 1])
        fine_v_pred = func_v_pred(fine_pos[:, 0], fine_pos[:, 1])  # [n_f,]
        ax[1, 1].tripcolor(tri, fine_v_pred, shading='gouraud',
                           vmin=fine_v_gt.min(), vmax=fine_v_gt.max())
        fig.suptitle('t={}'.format(t), fontsize=20)

    animation = FuncAnimation(fig, update, frames=np.arange(0, coarse_U_pred.shape[0], 20),
                              interval=80)
    animation.save(save_path, writer='pillow')

# todo
def plot_target_animation(coarse_pos, coarse_U_gt, fine_pos,
                             fine_tri, bdry_pos, bdry_elem, save_path):

    fig, ax = plt.subplots()
    plt.margins(0, 0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_alpha(0)
    tri = Triangulation(x=fine_pos[:, 0], y=fine_pos[:, 1], triangles=fine_tri)

    def update(t):
        ax.clear()
        ax.set_aspect('equal')
        for line in bdry_elem:
            ax.plot(bdry_pos[line, 0], bdry_pos[line, 1],
                          color='black', linewidth=0.2)
        ax.set_axis_off()
        ax.invert_yaxis()

        func_u_gt = CloughTocher2DInterpolator(
            points=(coarse_pos[:, 0], coarse_pos[:, 1]),
            values=coarse_U_gt[t, :, 0])
        fine_u_gt = func_u_gt(fine_pos[:, 0], fine_pos[:, 1])  # [n_f,]
        ax.tripcolor(tri, fine_u_gt, shading='gouraud')

    animation = FuncAnimation(fig, update, frames=np.arange(0, coarse_U_gt.shape[0], 20),
                              interval=80)
    animation.save(save_path, writer='pillow')



def plot_train_loss(epochs, tr_loss, val_loss, save_path=None):
    fig = plt.figure()
    train_idx = np.arange(1, epochs + 1)
    val_idx = np.arange(0, epochs + 1, 10)
    val_idx[0] = 1
    plt.semilogy(train_idx, tr_loss, label='train')
    plt.semilogy(val_idx, val_loss, label='val')
    plt.title('Train Loss')
    plt.legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)


def plot_trans_graph(graph, bdry_pos, bdry_element, save_path=None):
    """
    plot the points in the transformed graph, the special points will be marked
    as red.

    Args:
        graph (torch_geometric.data.Data): the transformed graph
        bdry_pos (np.ndarray): the boundary position of the compute domain
        save_path (str): save path of the figure
    Returns:

    """
    fig, ax = plt.subplots()
    for line in bdry_element:
        ax.plot(bdry_pos[line, 0], bdry_pos[line, 1], color='black', linewidth=1)
    ax.set_aspect('equal')
    ax.set_axis_off()

    ax.scatter(graph.pos[:, 0], graph.pos[:, 1], s=3, c='black')
    if hasattr(graph, 'periodic_src_index'):
        ax.scatter(graph.pos[graph.periodic_src_index, 0],
                   graph.pos[graph.periodic_src_index, 1], s=3, c='g')
        ax.scatter(graph.pos[graph.periodic_tgt_index, 0],
                   graph.pos[graph.periodic_tgt_index, 1], s=3, c='g')
    if hasattr(graph, 'neumann_src_index'):
        ax.scatter(graph.pos[graph.neumann_src_index, 0],
                   graph.pos[graph.neumann_src_index, 1], s=3, c='r')
        ax.scatter(graph.pos[graph.neumann_tgt_index, 0],
                   graph.pos[graph.neumann_tgt_index, 1], s=3, c='r')
    if hasattr(graph, 'dirichlet_index'):
        ax.scatter(graph.pos[graph.dirichlet_index, 0],
                   graph.pos[graph.dirichlet_index, 1], s=3, c='b')

    if hasattr(graph, 'inlet_index'):
        ax.scatter(graph.pos[graph.inlet_index, 0],
                   graph.pos[graph.inlet_index, 1], s=3, c='y')

    return fig


def plot_meshcolor_evolution_origin(fine_U, fine_pos, fine_tri, bdry_pos,
                                    bdry_elem, time_step, save_path=None):
    """

       Args:
           fine_U (np.ndarray): shape [t, n_f, 2]
           fine_pos (np.ndarray): shape [n_f, 2]
           fine_tri (np.ndarray): shape [n_tri, 3], triangles
           bdry_pos (np.ndarray): shape [n_d, 2]
           bdry_elem (np.ndarray): shape [n_l, 2]
           time_step (int):
           save_path (str:
       """
    t = fine_U.shape[0]
    rows = 2
    cols = t // time_step + 1
    t_steps = np.arange(0, t, time_step)
    t_steps = np.concatenate((t_steps, [t - 1]))
    fig, ax = plt.subplots(rows, cols, figsize=(11, 2))
    fig.text(0.1, 0.7, 'Ref. u', ha='center', va='center')
    fig.text(0.1, 0.3, 'Ref. v', ha='center', va='center')
    tri = Triangulation(x=fine_pos[:, 0], y=fine_pos[:, 1], triangles=fine_tri)

    for i, t in enumerate(t_steps):
        ax[0, i].tripcolor(tri, fine_U[t, :, 0], shading='gouraud', cmap='bwr')
        ax[1, i].tripcolor(tri, fine_U[t, :, 1], shading='gouraud', cmap='bwr')
        ax[0, i].set_title('t={}'.format(t))

    for i in range(rows):
        for j in range(cols):
            for line in bdry_elem:
                ax[i, j].plot(bdry_pos[line, 0],
                              bdry_pos[line, 1], color='black',
                              linewidth=0.2)
            # ax[i, j].plot(domains[:, 0], domains[:, 1], color='black', \
            # linewidth=0.2)
            ax[i, j].set_aspect('equal')
            ax[i, j].set_axis_off()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)


def plot_meshcolor(U, pos, tri, save_path=None):
    """

       Args:
           U (np.ndarray): shape [n, 2]
           pos (np.ndarray): shape [n, 2]
           tri (np.ndarray): shape [n_tri, 3], triangles
       """
    fig, ax = plt.subplots(2, 1)
    tri = Triangulation(x=pos[:, 0], y=pos[:, 1], triangles=tri)

    map0 = ax[0].tripcolor(tri, U[:, 0], shading='gouraud', cmap='bwr')
    map1 = ax[1].tripcolor(tri, U[:, 1], shading='gouraud', cmap='bwr')
    fig.colorbar(map0, ax=ax[0])
    fig.colorbar(map1, ax=ax[1])
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

    if save_path is not None:
        fig.savefig(save_path, dpi=300)

    plt.show()


def plot_meshcolor_single(coarse_pos, coarse_U_pred, coarse_U_gt, fine_pos, fine_tri, bdry_pos, bdry_elem, save_path=None):
    """

    Args:
        coarse_pos:
        coarse_U_pred: shape (n, 2)
        coarse_U_gt:
        fine_pos:
        fine_tri:
        bdry_pos:
        bdry_elem:
        time_step:
        save_path:

    Returns:

    """
    tri = Triangulation(x=fine_pos[:, 0], y=fine_pos[:, 1], triangles=fine_tri)

    func_u_gt = CloughTocher2DInterpolator(
        points=(coarse_pos[:, 0], coarse_pos[:, 1]),
        values=coarse_U_gt[:, 0])
    fine_u_gt = func_u_gt(fine_pos[:, 0], fine_pos[:, 1])  # [n_f,]
    fig, ax = plt.subplots()
    plt.margins(0, 0)
    ax.tripcolor(tri, fine_u_gt, shading='gouraud')
    for line in bdry_elem:
        ax.plot(bdry_pos[line, 0], bdry_pos[line, 1], color='black',
                      linewidth=0.2)
    ax.set_aspect('equal')
    ax.set_axis_off()
    if save_path is not None:
        fig.savefig(save_path.format('u_gt'), dpi=300, bbox_inches='tight', pad_inches=0)

    func_u_pred = CloughTocher2DInterpolator(
        points=(coarse_pos[:, 0], coarse_pos[:, 1]),
        values=coarse_U_pred[:, 0])
    fine_u_pred = func_u_pred(fine_pos[:, 0], fine_pos[:, 1])  # [n_f,]
    fig, ax = plt.subplots()
    plt.margins(0, 0)
    ax.tripcolor(tri, fine_u_pred, shading='gouraud',
                 vmin=fine_u_gt.min(), vmax=fine_u_gt.max())
    for line in bdry_elem:
        ax.plot(bdry_pos[line, 0], bdry_pos[line, 1], color='black',
                      linewidth=0.2)
    ax.set_aspect('equal')
    ax.set_axis_off()
    if save_path is not None:
        fig.savefig(save_path.format('u_prd'), dpi=300, bbox_inches='tight', pad_inches=0)

    func_v_gt = CloughTocher2DInterpolator(
        points=(coarse_pos[:, 0], coarse_pos[:, 1]),
        values=coarse_U_gt[:, 1])
    fine_v_gt = func_v_gt(fine_pos[:, 0], fine_pos[:, 1])  # [n_f,]
    fig, ax = plt.subplots()
    plt.margins(0, 0)
    ax.tripcolor(tri, fine_v_gt, shading='gouraud')
    for line in bdry_elem:
        ax.plot(bdry_pos[line, 0], bdry_pos[line, 1], color='black',
                      linewidth=0.2)
    ax.set_aspect('equal')
    ax.set_axis_off()
    if save_path is not None:
        fig.savefig(save_path.format('v_gt'), dpi=300, bbox_inches='tight', pad_inches=0)

    func_v_pred = CloughTocher2DInterpolator(
        points=(coarse_pos[:, 0], coarse_pos[:, 1]),
        values=coarse_U_pred[:, 1])
    fine_v_pred = func_v_pred(fine_pos[:, 0], fine_pos[:, 1])  # [n_f,]
    fig, ax = plt.subplots()
    plt.margins(0, 0)
    ax.tripcolor(tri, fine_v_pred, shading='gouraud',
                 vmin=fine_v_gt.min(), vmax=fine_v_gt.max())
    for line in bdry_elem:
        ax.plot(bdry_pos[line, 0], bdry_pos[line, 1], color='black',
                      linewidth=0.2)
    ax.set_aspect('equal')
    ax.set_axis_off()
    if save_path is not None:
        fig.savefig(save_path.format('v_prd'), dpi=300, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    draw_loss()