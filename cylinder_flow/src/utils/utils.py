import io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
import enum

from src.common.data_utils import correlation


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    INLET = 2
    OUTLET = 3


def activation_func():
    return nn.ELU()


def build_net(layers, activation_end=False):
    net = nn.Sequential()
    layer_n = len(layers)

    assert layer_n >= 2

    for i in range(layer_n - 2):
        net.add_module('linear' + str(i), nn.Linear(layers[i], layers[i + 1]))
        net.add_module('activation' + str(i), activation_func())
    net.add_module('linear' + str(layer_n - 2), nn.Linear(layers[layer_n - 2], layers[layer_n - 1]))
    if activation_end:
        net.add_module('activation' + str(layer_n - 2), activation_func())
    return net

# def build_net(layers, activation_end=False):
#     net = nn.Sequential()
#     layer_n = len(layers)

#     assert layer_n >= 2

#     for i in range(layer_n - 2):
#         net.add_module(str(i * 2), nn.Linear(layers[i], layers[i + 1]))
#         net.add_module('activation' + str(i * 2), activation_func())
#     net.add_module(str((layer_n - 2) * 2), nn.Linear(layers[layer_n - 2], layers[layer_n - 1]))
#     if activation_end:
#         net.add_module('activation' + str((layer_n - 2) * 2), activation_func())
#     return net


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


def compute_armse(pred, truth):
    """
    Parameters
    ----------
    pred: torch.Tensor, shape [t, n, d]
    truth: torch.Tensor, shape [t, n, d]

    Returns
    -------
    armse: List
    """
    armses = []
    for i in range(pred.shape[0]):
        nume = torch.linalg.norm(pred[:i+1] - truth[:i+1])
        deno = torch.linalg.norm(truth[:i+1])
        res = nume / deno
        armses.append(res.item())

    return armses


def writer_add_fig(writer, tag):
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    image = Image.open(buffer)
    image_tensor = TF.to_tensor(image)
    writer.add_image(tag, image_tensor)


def on_boundary(pos):
    eps = torch.finfo(torch.float32).eps
    def on_circle(x, y):
        if torch.abs((x - 5.) ** 2 + (y - 4.) ** 2 - 1. ** 2) < eps:
            return True
        else:
            return False

    flag = []
    for i in range(pos.shape[0]):
        x, y = pos[i]
        if on_circle(x, y):
            flag.append(True)
        else:
            flag.append(False)

    return torch.tensor(flag)


def add_noise(truth, percentage=0.05):
    # shape of truth must be (n, 2)
    assert truth.shape[1]==2
    uv = [truth[:, 0:1], truth[:, 1:2]]
    uv_noi = []
    for truth in uv:
        R = torch.normal(mean=0.0, std=1.0, size=truth.shape)
        std_R = torch.std(R)          # std of samples
        std_T = torch.std(truth)
        noise = R * std_T / std_R * percentage
        uv_noi.append(truth+noise)
    return torch.cat(uv_noi, dim=1)