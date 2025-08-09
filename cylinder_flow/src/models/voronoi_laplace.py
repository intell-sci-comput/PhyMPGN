"""
@reference: Discrete Laplace–Beltrami operators for shape analysis and
segmentation
"""
import torch
from shapely.geometry import MultiPoint
from tqdm import tqdm
import numpy as np


def compute_discrete_laplace(graph):
    """
    Compute discrete Laplace-Beltrami operator.

    Args:
        graph (torch_geometric.data.Data):

    Returns:
        L (torch.Tensor): shape (n, n), laplace matrix
    """
    W = compute_weight_matrix(graph)  # (n, n)
    v = torch.sum(W, dim=1)
    V = torch.diag(v)  # (n, n)
    A = V - W
    assert torch.isinf(A).any() == False
    assert torch.isnan(A).any() == False

    d = compute_d_vector(graph)
    d_inv = 1 / d
    d_inv[torch.isinf(d_inv)] = 0
    D_inv = torch.diag(d_inv)  # (n, n)
    assert torch.isinf(D_inv).any() == False
    assert torch.isnan(D_inv).any() == False

    L = D_inv @ A
    return -L, d_inv


# todo: optimize complexity
def compute_weight_matrix(graph):
    """
    Compute weight matrix of discrete Laplace-Beltrami operator proposed by
    Pinkall and Polthier.

    Args:
        graph:

    Returns:
        weights (torch.Tensor): shape (n, n), weight matrix
    """
    n = graph.pos.shape[0]
    e = graph.edge_index.shape[1]
    weights = torch.zeros((n, n))
    eps = torch.finfo(torch.float32).eps
    for e_i in tqdm(range(e)):
        edge = graph.edge_index[:, e_i]
        i, j = edge
        nodes = find_opposite_nodes(edge, graph.face)
        if len(nodes) != 0:
            p, q = nodes
            alpha = compute_opposite_angle([
                graph.pos[i], graph.pos[j], graph.pos[p]
            ])
            beta = compute_opposite_angle([
                graph.pos[i], graph.pos[j], graph.pos[q]
            ])
            if torch.isnan(alpha) or torch.isnan(beta):
                w = torch.tensor(0.)
            elif alpha < eps or beta < eps:
                w = torch.tensor(0.)
            else :
                w = (cot(alpha) + cot(beta)) / 2
            if torch.isnan(w):  # for debug
                print('weights nan, e_{}, n_{}-n_{}'.format(e_i, i, j))
            weights[i, j] = w
    return weights


# todo: optimize complexity
def compute_d_vector(graph):
    """
    Compute d matrix of discrete Laplace-Beltrami operator proposed by Meyer.

    Args:
        graph:

    Returns:
        d_vector (torch.Tensor): shape (n,), d_i is the area obtained by joining
            the circumcenters of the triangles around v_i.
    """
    d_vector = []
    n = graph.pos.shape[0]
    for i in tqdm(range(n)):
        tris = find_node_triangles(i, graph.face)
        area = compute_all_voronoi_area(graph.pos, tris)
        d_vector.append(area)
        if np.isnan(area): # for debug
            print('d nan, n_{}'.format(i))
    return torch.tensor(d_vector)


def find_opposite_nodes(edge, triangles):
    """
    Find the two opposite nodes of the edge in the triangles mesh.
    Args:
        edge (torch.Tensor): shape (2,), (v_i, v_j)
        triangles (torch.Tensor): shape (3, n_tri), each column is
            [v_p, v_q, v_r]

    Returns:
        nodes (List): null List if the two opposite nodes aren't found or
            (v_a, v_b)
    """
    nodes = []
    n_tri = triangles.shape[1]
    for i in range(n_tri):
        tri = triangles[:, i]
        is_subset = torch.all(torch.isin(edge, tri))
        if is_subset:
            diff = torch.masked_select(tri, ~torch.isin(tri, edge))
            nodes.append(diff.item())
    if len(nodes) == 1:
        nodes = []
    assert len(nodes) == 0 or len(nodes) == 2
    return nodes


def compute_opposite_angle(triangle):
    """
    Compute the opposite angle of edge ij in triangle.
    Args:
        triangle (List): len (3,), position three nodes (i, j, k)

    Returns:
        theta (torch.Tensor): shape (1,), the angle of vector ki and kj.
    """
    v_i, v_j, v_k = triangle[0], triangle[1], triangle[2]
    e_ki = v_k - v_i
    e_kj = v_k - v_j
    cos = torch.dot(e_ki, e_kj) / \
          (torch.linalg.vector_norm(e_ki) * torch.linalg.vector_norm(e_kj))
    # if cos > 1. or cos < -1., angle will be nan.
    angle = torch.acos(cos)
    return angle


def cot(theta):
    """
    cot = 1 / tan
    Args:
        theta:

    Returns:
        cot:
    """
    cot = 1 / torch.tan(theta)
    if cot < torch.finfo(torch.float32).eps:
        cot = torch.tensor(0.)
    return cot


def compute_tri_circumcenter(triangle):
    A, B, C = triangle[0], triangle[1], triangle[2]
    if A.shape[0] == 2:
        D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (
                    A[1] - B[1]))
        Ux = ((A[0] ** 2 + A[1] ** 2) * (B[1] - C[1]) + (
                    B[0] ** 2 + B[1] ** 2) * (C[1] - A[1]) + (
                          C[0] ** 2 + C[1] ** 2) * (A[1] - B[1])) / D
        Uy = ((A[0] ** 2 + A[1] ** 2) * (C[0] - B[0]) + (
                    B[0] ** 2 + B[1] ** 2) * (A[0] - C[0]) + (
                          C[0] ** 2 + C[1] ** 2) * (B[0] - A[0])) / D
        center = torch.tensor([Ux, Uy])
    else:
        AB = B - A
        AC = C - A
        AB_magnitude = torch.linalg.norm(AB)
        AC_magnitude = torch.linalg.norm(AC)

        # Calculate triangle normal
        N = torch.linalg.cross(AB, AC)
        N_magnitude = torch.linalg.norm(N)

        # Calculate circumcenter
        center = A + (AB_magnitude * torch.linalg.cross(
            AB_magnitude * AC - AC_magnitude * AB, N)) / (2 * N_magnitude ** 2)
    return center


def compute_voronoi_area(triangle):
    """
    Compute voronoi area.

    Args:
        triangle (array_like): shape (3, d), position of triangle nodes

    Returns:
        area (float):
    """
    A, B, C = triangle[0], triangle[1], triangle[2]

    AB = B - A
    AC = C - A
    cos_A = torch.dot(AB, AC) / \
               (torch.linalg.norm(AB) * torch.linalg.norm(AC))

    eps = torch.finfo(torch.float32).eps
    if np.abs(cos_A - 1.0) < eps:
        area = 0.
    elif cos_A < 0:  # A is obtuse
        area = 0.5 * MultiPoint(triangle).convex_hull.area
    else:
        circumcenter = compute_tri_circumcenter(triangle)
        MAB = (A + B) / 2
        MAC = (A + C) / 2
        area = MultiPoint([A, MAB, circumcenter, MAC]).convex_hull.area
    return area


def compute_all_voronoi_area(pos, tris):
    """
    Compute the circumcenters of triangles.
    Args:
        pos (torch.Tensor): shape (n, d), position of nodes
        tris (torch.Tensor): shape (n_tri, 3), each element is [v_i, v_j,
            v_k]. v_i is the index of nodes.

    Returns:
        area (torch.float): voronoi area of each triangle.
    """
    areas_sum = 0
    n_tri = tris.shape[0]
    for tri_i in range(n_tri):
        i, j, k = tris[tri_i]
        area = compute_voronoi_area([pos[i], pos[j], pos[k]])
        areas_sum += area
    return areas_sum


def find_node_triangles(node, triangles):
    """
    Find the triangles consists of the node
    Args:
        node (int): the index of node
        triangles (torch.Tensor): shape (3, n_tri), the triangles, each element
            is an index of node

    Returns:
        tri_indices (torch.Tensor): shape (m,), the index of triangles
    """
    tris = []
    n_tri = triangles.shape[1]
    for i in range(n_tri):
        triangle = triangles[:, i]
        if node in triangles[:, i]:
            # move node to the first location in triangle
            tri = torch.cat([torch.tensor([node]), triangle[triangle != node]])
            tris.append(tri)
    return torch.stack(tris)


def test():
    from torch_geometric.data import Data

    # test
    pos = torch.tensor([
        [0, 0], [1, 1], [1, 0], [1, -1], [0, -1],
        [-1, -1], [-1, 0], [-1, 1], [0, 1]
    ], dtype=torch.float32)
    edge = torch.tensor([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6,
         6, 6, 7, 7, 7, 8, 8, 8],
        [1, 2, 4, 6, 7, 8, 0, 2, 8, 0, 1, 3, 4, 2, 4, 0, 2, 3, 5, 6, 4, 6, 0, 4,
         5, 7, 0, 6, 8, 0, 1, 7],
    ])
    face = torch.tensor([
        [0, 0, 0, 0, 0, 0, 2, 4],
        [1, 2, 4, 6, 7, 8, 3, 5],
        [2, 4, 6, 7, 8, 1, 4, 6],
    ])

    graph = Data(pos=pos, edge_index=edge, face=face)
    L = compute_discrete_laplace(graph)
    print(L)


if __name__ == '__main__':
    test()

    # center = compute_tri_circumcenter(
    #     torch.tensor([[0, 0, 0],
    #      [0, 1, 0],
    #      [1, 0, 0]], dtype=torch.float32)
    # )
    # print(center)