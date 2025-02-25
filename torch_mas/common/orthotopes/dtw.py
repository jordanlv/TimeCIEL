import torch

import rust_dtw


def batch_dtw(activations, X, window_rate=0.1):
    """
    Return the cost matrix between each agent and each signal

    Args:
        activations: (tensor): (n_agents, seq_len_h, input_dim, 2)
        X (tensor) : (batch_size, seq_len_p, input_dim)

    Returns:
        cost_matrix (tensor) : (batch_size, n_agents)
        dist matrix (tensor) : (batch_size, n_agents, seq_len_p, seq_len_h)
        path_matrix (tensor) : (batch_size, n_agents, seq_len_p, seq_len_h)
    """
    window = int(X.shape[1] * window_rate)
    costs, dist, path = rust_dtw.batch_dtw_full(X.numpy(), activations.numpy(), window)
    return (
        torch.from_numpy(costs.copy()),
        torch.from_numpy(dist.copy()),
        torch.from_numpy(path[:, :, 1:, 1:].copy()),
    )


def batch_is_neighbor(activations, X, paths, side_length):
    """
    Args:
        activations (tensor): (n_agents, seq_len, input_dim, 2)
        X (tensor): (batch_size, seq_len, in_dim)
        paths (boolean tensor): (batch_size, n_agents, seq_len, seq_len)
        side_length (float): the maximum distance between the activation and the X
    Returns:
        tensor: (batch_size, n_agents)
    """
    neighbors = rust_dtw.batch_is_neighbor(
        activations.numpy(), X.numpy(), paths[:, :, 1:, 1:].numpy(), side_length
    )
    return torch.from_numpy(neighbors.copy())


def update_hypercube(hypercube, x, path, alpha):
    """Updates an hypercube towards x, modifying its sides so that the final volume
    is change by a factor alpha.

    Args:
        hypercube (Tensor): (n_dim, 2)
        x (Tensor): (seq_len, n_dim,)
        path (tensor): (seq_len_p,)
        alpha (Tensor): (1,)

    Returns:x
        Tensor: (n_dim, 2)
    """

    updated_hypercube = hypercube.clone().detach()
    low, high = updated_hypercube[:, 0], updated_hypercube[:, 1]

    dists = torch.where(
        x < low, low - x, torch.where(x > high, x - high, torch.zeros_like(x))
    )

    x_indices = torch.max((dists * path.unsqueeze(-1)), dim=0).indices
    x = x.gather(0, x_indices.unsqueeze(1)).squeeze(1)

    dims_mask = (x < high) & (x > low)

    should_extend = torch.where(alpha < 0.0, True, ~(dims_mask.all()))

    dims_mask = torch.where(dims_mask.all(), dims_mask, ~dims_mask)
    theta = torch.sum(dims_mask)
    diff = high - low
    new_high = diff * torch.pow(1 + alpha, 1 / theta) + low
    new_low = high - diff * torch.pow(1 + alpha, 1 / theta)

    dist_low = torch.abs(low - x)
    dist_high = torch.abs(high - x)
    mask = dist_high < dist_low

    high = torch.where(should_extend & (mask & dims_mask), new_high, high)
    low = torch.where(should_extend & (~mask & dims_mask), new_low, low)
    # print(torfch.stack([low, high], dim=-1).shape)
    return torch.stack([low, high], dim=-1)


_update_temporal_hypercube = torch.vmap(update_hypercube, in_dims=(0, None, 1, 0))


def update_temporal_hypercube(hypercubes, x, path, alphas):
    """Updates a batch of hypercubes towards x, modifying its sides so that the final volume
    is change by a factor alpha.

    Args:
        hypercubes (Tensor): (seq_len, n_dim, 2)
        x (Tensor): (seq_len, n_dim,)
        path (tensor): (seq_len_p, seq_len_h, 1)
        alphas (Tensor): (seq_len, 1)

    Returns:
        Tensor: (seq_len, n_dim, 2)
    """
    return _update_temporal_hypercube(hypercubes, x, path, alphas)


_batch_update_temporal_hypercube = torch.vmap(
    update_temporal_hypercube, in_dims=(0, None, 0, 0)
)


def batch_update_temporal_hypercube(hypercubes, x, path, alphas):
    """Updates a batch of hypercubes towards x, modifying its sides so that the final volume
    is change by a factor alpha.

    Args:
        hypercubes (Tensor): (batch_size, seq_len, n_dim, 2)
        x (Tensor): (seq_len, n_dim,)
        path (tensor): (batch_size, seq_len_p, seq_len_h, 1)
        alphas (Tensor): (batch_size, seq_len, 1)

    Returns:
        Tensor: (batch_size, seq_len, n_dim, 2)
    """

    return _batch_update_temporal_hypercube(hypercubes, x, path, alphas)


# def distance(h, x):
#     """
#     Compute the distance between a hypercube and a point

#     Args:
#         h (tensor): (input_dim, 2)
#         x (tensor): (input_dim,)

#     Returns:
#         tensor: (input_dim)
#     """

#     lower = h[:, 0]
#     upper = h[:, 1]

#     per_dim_distance = torch.where(
#         x < lower, lower - x, torch.where(x > upper, x - upper, torch.zeros_like(x))
#     )

#     return per_dim_distance


# def is_neighbor(activation, X, path, radius):
#     """
#     Check if the path is a valid neighbor of the activation

#     Args:
#         activation (tensor): (n_agents, seq_len, input_dim, 2)
#         X (tensor): (batch_size, seq_len, input_dim)
#         path (tensor): (batch_size, n_agents, seq_len, seq_len)
#         radius (float): the maximum distance between the activation and the X
#     Returns:
#         (tensor): (n_agents, batch_size)
#     """
#     batch_size, seq_len, input_dim = X.size()
#     n_agents = activation.size(0)

#     neighbor = torch.ones((n_agents, batch_size), dtype=torch.bool, device=X.device)
#     for i in range(n_agents):
#         for j in range(batch_size):
#             current_agent = activation[i]
#             current_path = path[j, i]
#             current_X = X[j]

#             indices = nonzero_indices(current_path)
#             for k in range(len(indices[0])):
#                 if (
#                     distance(current_agent[indices[0][k]], current_X[indices[1][k]])
#                     > radius
#                 ).all():
#                     neighbor[i, j] = False
#                     break

#     return neighbor


# def nonzero_indices(tensor):
#     """
#     Return the indices of the non zero elements of a tensor

#     Args:
#         tensor (tensor): (n_dim, n_dim)

#     Returns:
#         tuple:
#             - (tensor): (n_nonzero,)
#             - (tensor): (n_nonzero,)
#     """
#     rowindicies = []
#     colindicies = []
#     for i in range(tensor.size(0)):
#         for j in range(tensor.size(1)):
#             if tensor[i, j] != 0:
#                 rowindicies.append(i)
#                 colindicies.append(j)
#     return torch.tensor(rowindicies), torch.tensor(colindicies)
