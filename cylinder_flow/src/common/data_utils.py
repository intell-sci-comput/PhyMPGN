import numpy as np
from sklearn.neighbors import KDTree
import torch


def load_txt_data(data_path):
    data = np.loadtxt(data_path, comments='%')  # [n, 2+2*t]
    pos = data[:, :2]  # [n, 2]
    u = data[:, 2::2]  # [n, t]
    v = data[:, 3::2]  # [n, t]
    U = np.stack((u, v), axis=-1)  # [n, t, 2]
    U = U.transpose((1, 0, 2))  # [t, n, 2]
    return pos, U


def load_mesh(mesh_path, node_lines):
    pos = np.loadtxt(mesh_path, max_rows=node_lines, comments='%')  # [m, 2]
    tri = np.loadtxt(mesh_path, skiprows=node_lines+11, dtype=np.int64)  # [ntri, 3]
    tri = tri - 1
    return pos, tri


def load_txt_bdry(mesh_path, node_lines):
    pos = np.loadtxt(mesh_path, comments='%', max_rows=node_lines)  # [n_b, p_d]
    element = np.loadtxt(mesh_path, skiprows=node_lines+11, dtype=np.int64)  # [n_l, 2]
    element = element - 1  # change index start from 1 to 0
    return pos, element


def find_correspond_indices(fine_pos, coarse_pos):
    """

    Args:
        fine_pos (np.ndarray):
        coarse_pos (np.ndarray):

    Returns:
        indices (np.ndarray): index of the element of pos2 in pos1
    """
    kd_tree = KDTree(fine_pos)
    indices = kd_tree.query(coarse_pos, return_distance=False).squeeze()  # []
    return indices


def compute_error(U_gt, U_pred):
    """
    :param U_gt: [t, n, 2]
    :param U_pred: [t, n, 2]
    :return:
    """
    # t = U_gt.shape[0]
    # U_gt = U_gt.reshape(t, -1)
    # U_pred = U_pred.reshape(t, -1)

    # nume = torch.linalg.norm(U_pred - U_gt, dim=1)
    # deno = torch.linalg.norm(U_gt, dim=1)

    nume = torch.linalg.norm(U_pred - U_gt)
    deno = torch.linalg.norm(U_gt)
    res = nume / deno

    return res


def correlation(u, truth):
    """

    Parameters
    ----------
    u: np.ndarray, shape [n, d]
    truth: np.ndarray, shape [n, d]

    Returns
    -------
    coef: float
    """
    u = u.reshape(1, -1)
    truth = truth.reshape(1, -1)
    u_truth = np.concatenate((u, truth), axis=0)
    coef = np.corrcoef(u_truth)[0][1]
    return coef


def cal_cur_time_corre(u, truth):
    """
    Parameters
    ----------
    u: np.ndarray, shape [t, n, d]
    truth: np.ndarray, shape [t, n, d]

    Returns
    -------
    coef_list: list
    """
    coef_list = []
    for i in range(u.shape[0]):
        # if i % 100 == 0:
        #     print(i)
        cur_truth = truth[i]
        cur_u = u[i]
        # cur_truth = truth[:i].squeeze(dim=)#accumulate
        cur_coef = correlation(cur_u, cur_truth)
        coef_list.append(cur_coef)
    return coef_list