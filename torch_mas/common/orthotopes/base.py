import torch


def create_hypercube(x, side_lengths):
    """Create an hypercube from its center and side lengths

    Args:
        x (Tensor): (n_dim)
        side_lengths (Tensor): (n_dim,)

    Returns:
        Tensor: (n_dim, 2)
    """
    lows = x - side_lengths / 2
    highs = x + side_lengths / 2
    hypercube = torch.stack([lows, highs], dim=-1)
    return hypercube


_create_temporal_hypercube = torch.vmap(create_hypercube, in_dims=(0, None))


def create_temporal_hypercube(x, side_lengths):
    """Create a temporal hypercube from its centers and side lengths

    Args:
        x (Tensor): (seq_len, n_dim)
        side_lengths (Tensor): (n_dim,)

    Returns:
        Tensor: (seq_len, n_dim, 2)
    """
    return _create_temporal_hypercube(x, side_lengths)


_batch_create_temporal_hypercube = torch.vmap(create_hypercube)


def batch_create_temporal_hypercube(x, side_lengths):
    """Create a batch of hypercubes from their center and side lengths

    Args:
        x (Tensor): (batch_size, seq_len, n_dim)
        side_lengths (Tensor): (batch_size, seq_len, n_dim)

    Returns:
        Tensor: (batch_size, seq_len, n_dim, 2)
    """
    return _batch_create_temporal_hypercube(x, side_lengths)


def intersect_point(hypercube, x):
    """Check if a point is contained in a hypercube

    Args:
        hypercube (Tensor): (n_dim, 2)
        x (Tensor): (n_dim,)

    Returns:
        BoolTensor: (1,)
    """
    return ((x < hypercube[:, 1]) & (x > hypercube[:, 0])).all()


_intersect_signal = torch.vmap(intersect_point, in_dims=(0, 0))


def intersect_signal(hypercubes, x):
    """Check in which hypercubes of a batch of hypercubes x is contained.

    Args:
        hypercube (Tensor): (seq_len, n_dim, 2)
        x (Tensor): (seq_len, n_dim,)

    Returns:
        BoolTensor: (seq_len, 1)
    """
    return _intersect_signal(hypercubes, x)


_batch_intersect_signal = torch.vmap(intersect_signal, in_dims=(0, None))


def batch_intersect_signal(hypercubes, x):
    """Check in which hypercubes of a batch of hypercubes each x of a batch is contained.

    Args:
        hypercubes (Tensor): (h_batch_size, seq_len, n_dim, 2)
        x (Tensor): (seq_len, n_dim)

    Returns:
        BoolTensor: (x_batch_size, seq_len, 1)
    """
    return _batch_intersect_signal(hypercubes, x)


_batch_intersect_signals = torch.vmap(batch_intersect_signal, in_dims=(None, 0))


def batch_intersect_signals(hypercubes, x):
    """Check in which hypercubes of a batch of hypercubes each x of a batch is contained.

    Args:
        hypercubes (Tensor): (h_batch_size, seq_len, n_dim, 2)
        x (Tensor): (x_batch_size, seq_len, n_dim)

    Returns:
        BoolTensor: (h_batch_size, x_batch_size, seq_len)
    """
    return _batch_intersect_signals(hypercubes, x)


def intersect_hypercube(hypercube1, hypercube2):
    """Check if two hypercubes intersect

    Args:
        hypercube1 (Tensor): (n_dim, 2)
        hypercube2 (Tensor): (n_dim, 2)

    Returns:
        BoolTensor: (1,)
    """
    max_start = torch.maximum(hypercube1[:, 0], hypercube2[:, 0])
    min_end = torch.minimum(hypercube1[:, 1], hypercube2[:, 1])
    return (max_start <= min_end).all()


_intersect_temporal_hypercube = torch.vmap(intersect_hypercube, in_dims=(0, 0))


def intersect_temporal_hypercube(hypercube1, hypercubes2):
    """Check which hypercube in a batch of hypercubes intersects with another hypercube.

    Args:
        hypercube1 (Tensor): (seq_len, n_dim, 2)
        hypercubes2 (Tensor): (seq_len, n_dim, 2)

    Returns:
        Tensor: (seq_len, 1)
    """
    return _intersect_temporal_hypercube(hypercube1, hypercubes2)


_batch_intersect_temporal_hypercube = torch.vmap(
    intersect_temporal_hypercube, in_dims=(None, 0)
)


def batch_intersect_temporal_hypercube(hypercube1, hypercubes2):
    """Check which hypercube in a batch of hypercubes intersects with another hypercube.

    Args:
        hypercube1 (Tensor): (h1_batch_size, seq_len, n_dim, 2)
        hypercubes2 (Tensor): (seq_len, n_dim, 2)

    Returns:
        Tensor: (h1_batch_size, seq_len, 1)
    """
    return _batch_intersect_temporal_hypercube(hypercube1, hypercubes2)


_batch_intersect_temporal_hypercubes = torch.vmap(
    batch_intersect_temporal_hypercube, in_dims=(0, None)
)


def batch_intersect_temporal_hypercubes(hypercube1, hypercubes2):
    """Check which hypercube in a batch of hypercubes intersects with another hypercube.

    Args:
        hypercube1 (Tensor): (h1_batch_size, seq_len, n_dim, 2)
        hypercubes2 (Tensor): (h2_batch_size, seq_len, n_dim, 2)

    Returns:
        Tensor: (h2_batch_size, h1_batch_size, seq_len)
    """
    return _batch_intersect_temporal_hypercubes(hypercube1, hypercubes2)


def update_hypercube(hypercube, x, alpha):
    """Updates an hypercube towards x, modifying its sides so that the final volume
    is change by a factor alpha.

    Args:
        hypercube (Tensor): (n_dim, 2)
        x (Tensor): (n_dim,)
        alpha (Tensor): (1,)

    Returns:x
        Tensor: (n_dim, 2)
    """
    updated_hypercube = hypercube.clone().detach()
    low, high = updated_hypercube[:, 0], updated_hypercube[:, 1]
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

    return torch.stack([low, high], dim=-1)


_update_temporal_hypercube = torch.vmap(update_hypercube, in_dims=(0, 0, 0))


def update_temporal_hypercube(hypercubes, x, alphas):
    """Updates a batch of hypercubes towards x, modifying its sides so that the final volume
    is change by a factor alpha.

    Args:
        hypercubes (Tensor): (seq_len, n_dim, 2)
        x (Tensor): (seq_len, n_dim,)
        alphas (Tensor): (seq_len, 1)

    Returns:
        Tensor: (seq_len, n_dim, 2)
    """
    return _update_temporal_hypercube(hypercubes, x, alphas)


_batch_update_temporal_hypercube = torch.vmap(
    update_temporal_hypercube, in_dims=(0, None, 0)
)


def batch_update_temporal_hypercube(hypercubes, x, alphas):
    """Updates a batch of hypercubes towards x, modifying its sides so that the final volume
    is change by a factor alpha.

    Args:
        hypercubes (Tensor): (batch_size, seq_len, n_dim, 2)
        x (Tensor): (seq_len, n_dim,)
        alphas (Tensor): (batch_size, seq_len, 1)

    Returns:
        Tensor: (batch_size, seq_len, n_dim, 2)
    """

    return _batch_update_temporal_hypercube(hypercubes, x, alphas)


def sides(hypercube):
    """Calculate the side lengths of an hypercube

    Args:
        hypercube (Tensor): (n_dim, 2)

    Returns:
        Tensor: (n_dim, 1)
    """
    return hypercube[:, 1] - hypercube[:, 0]


_batch_sides = torch.vmap(sides)


def batch_sides(hypercubes):
    """Calculate the side lengths of a batch of hypercubes

    Args:
        hypercubes (Tensor): (batch_size, n_dim, 2)

    Returns:
        Tensor: (batch_size, n_dim, 1)
    """
    return _batch_sides(hypercubes)


def volume(hypercube):
    """Calculate volume of a hypercube

    Args:
        hypercube (Tensor): (n_dim, 2)

    Returns:
        Tensor: (1,)
    """
    return torch.prod(hypercube[:, 1] - hypercube[:, 0])


_batch_volume = torch.vmap(volume)


def batch_volume(hypercubes):
    """Calculate volume of a batch of hypercubes

    Args:
        hypercubes (Tensor): (batch_size, n_dim, 2)

    Returns:
        Tensor: (batch_size, 1)
    """
    return _batch_volume(hypercubes)


def dist_point_to_border(hypercube, x):
    """Calculate the distance of a point to the edge of an hypercube

    Args:
        hypercube (Tensor): (n_dim, 2)
        x (Tensor): (n_dim,)

    Returns:
        Tensor: distance of the point to the edge of the specified hypercube
    """
    low, high = hypercube[:, 0], hypercube[:, 1]
    dist_to_low = torch.abs(x - low)
    dist_to_high = torch.abs(x - high)
    dists = torch.min(dist_to_low, dist_to_high)
    min_dist = torch.linalg.norm(dists)
    return min_dist


_dist_signal_to_border = torch.vmap(dist_point_to_border, in_dims=(0, 0))


def dist_signal_to_border(hypercubes, x):
    """Calculate the distance of a point to the edges of a batch of hypercubes

    Args:
        hypercubes (Tensor): (seq_len, n_dim, 2)
        x (Tensor): (seq_len, n_dim,)

    Returns:
        Tensor: (seq_len,)
    """
    return _dist_signal_to_border(hypercubes, x)


_batch_dist_signal_to_border = torch.vmap(dist_signal_to_border, in_dims=(0, None))


def batch_dist_signal_to_border(hypercubes, x):
    """Calculate the distance of a point to the edges of a batch of hypercubes

    Args:
        hypercubes (Tensor): (batch_size, seq_len, n_dim, 2)
        x (Tensor): (seq_len, n_dim,)

    Returns:
        Tensor: (batch_size,)
    """
    return _batch_dist_signal_to_border(hypercubes, x)


_batch_dist_signals_to_border = torch.vmap(
    batch_dist_signal_to_border, in_dims=(None, 0)
)


def batch_dist_signals_to_border(hypercubes, x):
    """Calculate the distance of each point of a batch of points to the edges of a batch of hypercubes

    Args:
        hypercubes (Tensor): (h_batch_size, seq_len, n_dim, 2)
        x (Tensor): (x_batch_size, seq_len, n_dim)

    Returns:
        Tensor: (x_batch_size, h_batch_size)
    """
    return _batch_dist_signals_to_border(hypercubes, x)
