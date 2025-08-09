
def periodic_padding(features, source_index, target_index):
    """

    Args:
        features (torch.Tensor): shape [n, ...], the origin features
        source_index (torch.Tensor): shape [m,]
        target_index (torch.Tensor): shape [m,]

    Returns:
        features (torch.Tensor): shape [n, ...], the padded features
    """
    features[target_index] = features[source_index]
    return features


def dirichlet_padding(features, padding_index, padding_value):
    if len(features.shape) == 3:
        #  (m, t, d)
        features[padding_index] = padding_value.unsqueeze(1)\
            .repeat(1, features.shape[1], 1)
    else:  # == 2
        features[padding_index] = padding_value
    return features


def neumann_padding(features, source_index, target_index):
    features[target_index] = features[source_index]
    return features


def graph_padding(graph, clone=False):

    if hasattr(graph, 'dirichlet_index'):
        graph.y = dirichlet_padding(graph.y, graph.dirichlet_index,
                                    graph.dirichlet_value)
    if hasattr(graph, 'inlet_index'):
        graph.y = dirichlet_padding(graph.y, graph.inlet_index,
                                    graph.inlet_value)
    if hasattr(graph, 'periodic_src_index'):
        graph.y = periodic_padding(graph.y, graph.periodic_src_index,
                                   graph.periodic_tgt_index)
    if hasattr(graph, 'neumann_src_index'):
        graph.y = neumann_padding(graph.y, graph.neumann_src_index,
                                  graph.neumann_tgt_index)

    if clone:
        graph.y = graph.y.clone()


def h_padding(h, graph):
    if hasattr(graph, 'dirichlet_index'):
        h = dirichlet_padding(h, graph.dirichlet_index,
                                    graph.dirichlet_h_value)
    if hasattr(graph, 'inlet_index'):
        h = dirichlet_padding(h, graph.inlet_index,
                                    graph.inlet_h_value)
    if hasattr(graph, 'periodic_src_index'):
        h = periodic_padding(h, graph.periodic_src_index,
                                   graph.periodic_tgt_index)
    if hasattr(graph, 'neumann_src_index'):
        h = neumann_padding(h, graph.neumann_src_index,
                                  graph.neumann_tgt_index)
    return h