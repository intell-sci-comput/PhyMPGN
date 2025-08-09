from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import torch

from src.utils.utils import NodeType


@functional_transform('my_cartesian')
class MyCartesian(BaseTransform):
    r"""Saves the relative Cartesian coordinates of linked nodes in its edge
    attributes (functional name: :obj:`cartesian`).

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to the interval :math:`{[0, 1]}^D`.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        norm: bool = False
    ):
        self.norm = norm

    def __call__(self, data: Data) -> Data:
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cart = pos[col] - pos[row]
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart
        data.rel_pos = cart

        if self.norm and cart.numel() > 0:
            max_value = cart.abs().max()
            cart = cart / (2 * max_value) + 0.5

        if pseudo is not None:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(norm={self.norm}'


@functional_transform('my_distance')
class MyDistance(BaseTransform):
    r"""Saves the Euclidean distance of linked nodes in its edge attributes
    (functional name: :obj:`distance`).

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to the interval :math:`[0, 1]`. (default: :obj:`False`)
    """
    def __init__(self, norm: bool = False):
        self.norm = norm

    def __call__(self, data: Data) -> Data:
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
        data.distance = dist

        if self.norm and dist.numel() > 0:
            dist = dist / dist.max()

        if pseudo is not None:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = dist

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(norm={self.norm}'


@functional_transform('periodic')
class Periodic(BaseTransform):
    def __init__(self, distance):
        super(Periodic, self).__init__()
        self.y_min = 0.
        self.y_max = 8.
        self.distance = distance
        self.ghost_pos = None
        self.source_index = None
        self.target_index = None

    def is_none(self):
        return self.ghost_pos is None

    def __call__(self, data):
        if self.is_none():
            self.get_periodic_index(data)

        data.periodic_src_index = self.source_index  # [m,]
        data.periodic_tgt_index = self.target_index

        data.pos = torch.cat((data.pos, self.ghost_pos), dim=0)  # [n+m, t, d]
        ghost_y = torch.zeros((self.ghost_pos.shape[0], data.y.shape[1],
                               data.y.shape[2]))  # [m, t, d]
        data.y = torch.cat((data.y, ghost_y), dim=0)  # [n+m, t, d]

        return data

    def get_periodic_index(self, data):
        pos = data.pos
        ghost_pos = []
        tgt2src = {}
        target_count = pos.shape[0]

        for i in range(pos.shape[0]):
            x, y = pos[i]
            if 0 < y - self.y_min <= self.distance:
                ghost_x = x
                ghost_y = y + (self.y_max - self.y_min)
                ghost_pos.append([ghost_x, ghost_y])
                tgt2src[target_count] = i
                target_count += 1
            elif 0 < self.y_max - y <= self.distance:
                ghost_x = x
                ghost_y = y - (self.y_max - self.y_min)
                ghost_pos.append([ghost_x, ghost_y])
                tgt2src[target_count] = i
                target_count += 1

        self.ghost_pos = torch.tensor(ghost_pos)  # [m, 2]
        self.source_index = torch.tensor(list(tgt2src.values()),
                                         dtype=torch.long)
        self.target_index = torch.tensor(list(tgt2src.keys()), dtype=torch.long)


@functional_transform('dirichlet')
class Dirichlet(BaseTransform):
    def __init__(self):
        self.index = None

    def set_index(self, index):
        self.index = torch.tensor(index, dtype=torch.long)

    def __call__(self, data):
        data.dirichlet_index = self.index
        return data


@functional_transform('dirichlet_inlet')
class DirichletInlet(BaseTransform):
    def __init__(self):
        self.index = None

    def set_index(self, index):
        self.index = torch.tensor(index, dtype=torch.long)

    def __call__(self, data):
        data.inlet_index = self.index
        return data


@functional_transform('neumann')
class Neumann(BaseTransform):
    def __init__(self, distance, distance_circle):
        self.x0 = 0.5
        self.y0 = 0.3
        self.r = 0.2
        self.bdry = 0.5
        self.distance = distance
        self.distance_circle = distance_circle

        self.ghost_pos = None
        self.source_index = None
        self.target_index = None

    def is_none(self):
        return self.ghost_pos is None

    def __call__(self, data):
        if self.is_none():
            self.get_neumann_index(data)

        data.neumann_src_index = self.source_index
        data.neumann_tgt_index = self.target_index

        data.pos = torch.cat((data.pos, self.ghost_pos), dim=0)  # [n+m, t, d]
        ghost_y = torch.zeros((self.ghost_pos.shape[0], data.y.shape[1],
                               data.y.shape[2]))  # [m, t, d]
        data.y = torch.cat((data.y, ghost_y), dim=0)  # [n+m, t, d]
        return data

    def get_neumann_index(self, data):
        """
        get dirichlet index from data.pos. The function needs to be changed \
        according to location of the dirichlet boundary

        Args:
            data (torch_geometric.data.Data):

        Returns:
            dirichlet_index (np.ndarray):
        """
        pos = data.pos
        ghost_pos = []
        tgt2src = {}
        target_count = pos.shape[0]

        eps = torch.finfo(torch.float32).eps

        def near_circle(x, y, distance):
            if eps < torch.sqrt((x - self.x0)**2 + (y - self.y0)**2) - self.r \
                    <= distance:
                return True
            else:
                return False

        def on_circle(x, y):
            if torch.sqrt((x - self.x0)**2 + (y - self.y0)**2) - self.r \
                    <= eps:
                return True
            else:
                return False


        for i in range(pos.shape[0]):
            x1, y1 = pos[i]
            if 0 < self.bdry - x1 < self.distance and not on_circle(x1, y1):
                ghost_x = x1 + (self.bdry - x1) * 2
                ghost_y = y1
                ghost_pos.append([ghost_x, ghost_y])
                tgt2src[target_count] = i
                target_count += 1
            if near_circle(x1, y1, self.distance_circle):
                if torch.abs(x1 - self.x0) < eps:
                    ghost_x = x1
                    if y1 > self.y0 + self.r:
                        y2 = self.y0 + self.r
                    else:
                        y2 = self.y0 - self.r
                    ghost_y = 2 * y2 - y1
                else:
                    tan = (y1 - self.y0) / (x1 - self.x0)
                    cos_2 = 1 / (1 + tan**2)  # cos^2 = 1 / tan^2
                    # sin^2 = tan^2 / (1 + tan^2)
                    sin_2 =  tan**2 / (1 + tan**2)
                    if x1 - self.x0 > 0:
                        cos = torch.sqrt(cos_2)
                    else:
                        cos = - torch.sqrt(cos_2)
                    if y1 - self.y0 > 0:
                        sin = torch.sqrt(sin_2)
                    else:
                        sin = - torch.sqrt(sin_2)
                    ghost_x = 2 * (self.r * cos + self.x0) - x1
                    ghost_y = 2 * (self.r * sin + self.y0) - y1
                ghost_pos.append([ghost_x, ghost_y])
                tgt2src[target_count] = i
                target_count += 1

        self.ghost_pos = torch.tensor(ghost_pos)
        self.source_index = torch.tensor(list(tgt2src.values()),
                                         dtype=torch.long)
        self.target_index = torch.tensor(list(tgt2src.keys()),
                                         dtype=torch.long)

# todo remove
@functional_transform('mask_face')
class MaskFace(BaseTransform):
    def __init__(self):
        self.cylinder_index = None
        self.new_face_index = None

    def is_none(self):
        return self.new_face_index is None

    def set_cylinder_index(self, cylinder_index):
        self.cylinder_index = torch.tensor(cylinder_index, dtype=torch.long)

    def __call__(self, data):
        if self.is_none():
            self.new_face_index = self.cal_mask_face(data)

        data.face = data.face[:, self.new_face_index]
        return data

    def cal_mask_face(self, graph):
        on_circle_index = self.cylinder_index
        new_face_index = []
        for i in range(graph.face.shape[1]):
            if torch.isin(graph.face[:, i], on_circle_index).all():
                continue
            else:
                new_face_index.append(i)
        return torch.tensor(new_face_index)


@functional_transform('node_type_info')
class NodeTypeInfo(BaseTransform):
    def __init__(self):
        self.type_dict = None
        self.node_type = None

    def is_none(self):
        return self.node_type is None

    def set_type_dict(self, type_dict):
        self.type_dict = type_dict

    def __call__(self, data):
        if self.is_none():
            self.node_type = self.cal_node_type(data)

        data.node_type = self.node_type
        return data

    def cal_node_type(self, data):
        node_num = data.pos.shape[0]
        node_type = torch.ones(node_num, dtype=torch.long) * NodeType.NORMAL
        if hasattr(data, 'dirichlet_index'):
            node_type[data.dirichlet_index] = NodeType.OBSTACLE
        if hasattr(data, 'inlet_index'):
            node_type[data.inlet_index] = NodeType.INLET

        outlet_index = self.type_dict['outlet'][:]
        outlet_index = torch.tensor(outlet_index, dtype=torch.long)
        node_type[outlet_index] = NodeType.OUTLET
        return node_type
