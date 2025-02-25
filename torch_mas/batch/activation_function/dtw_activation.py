import torch
import copy

from . import BaseActivation

from ...common.orthotopes.base import (
    batch_intersect_signal,
    # batch_intersect_signals,
    # batch_create_temporal_hypercube,
    # batch_intersect_temporal_hypercubes,
    # batch_dist_signals_to_border,
)

from ...common.orthotopes.dtw import batch_update_temporal_hypercube, batch_is_neighbor

batch_update_temporal_hypercubes = torch.vmap(
    batch_update_temporal_hypercube, in_dims=(None, 0, 0, None)
)
batch_batch_intersect_signal = torch.vmap(batch_intersect_signal)
batch_batch_update_temporal_hypercube = torch.vmap(
    batch_update_temporal_hypercube, in_dims=(None, 0, 0, 0)
)


class DTWActivation(BaseActivation):

    def activated(self, costs):
        return costs[:, :] == 0

    def neighbors(self, X, paths, side_length):
        """
        Args:
            X (tensor): (batch_size, seq_len, in_dim)
            paths (boolean tensor): (batch_size, n_agents, seq_len, seq_len)
            side_length (float): the maximum distance between the activation and the X
        Returns:
            tensor: (batch_size, n_agents)
        """
        return batch_is_neighbor(self.orthotopes, X, paths, side_length)

    def update(self, X, agents_mask, bad, no_activated, path):
        batch_size, seq_len, _ = X.size()
        self.goods += (~bad).sum(0).view(self.n_agents, 1)
        self.bads += bad.sum(0).view(self.n_agents, 1)

        # print(agents_mask.shape, path.shape)

        alphas = torch.zeros(
            (batch_size, self.n_agents, seq_len), device=self.device
        )  # (batch_size, n_agents)

        alphas = torch.where(
            agents_mask.unsqueeze(-1)
            & no_activated.unsqueeze(-1).unsqueeze(-1)
            & ~bad.unsqueeze(-1),
            self.alpha,
            alphas,
        )
        alphas = torch.where(
            agents_mask.unsqueeze(-1)
            & ~no_activated.unsqueeze(-1).unsqueeze(-1)
            & bad.unsqueeze(-1),
            self.alpha,
            alphas,
        )

        updated_orthotopes = batch_batch_update_temporal_hypercube(
            self.orthotopes, X, path, alphas
        )  # (batch_size, n_agents, in_dim, 2)
        deltas = (
            updated_orthotopes - self.orthotopes
        )  # (batch_size, n_agents, in_dim, 2)
        deltas = deltas.sum(dim=0)  # (n_agents, in_dim, 2)
        self.orthotopes += deltas

    def immediate_expandable(self, X, path):
        """
        Calculate which agents should be expanded towards X
        Args:
            X (tensor): (batch_size, seq_len, in_dim)
            path (tensor): (batch_size, n_agents, seq_len, seq_len)
        Returns:
            tensor: (batch_size, n_agents)
        """

        batch_size, seq_len, input_dim = X.size()

        expanded_neighbors = batch_update_temporal_hypercubes(
            self.orthotopes,
            X,
            path,
            torch.full((self.n_agents, X.size(1)), self.alpha, device=self.device),
        )  # (batch_size, n_agents, in_dim, 2) possible shape of each agent for each x

        # This should take into account the path
        expanded_mask = batch_batch_intersect_signal(
            expanded_neighbors,  # (batch_size*n_agents, in_dim, 2)
            X,  # (batch_size, in_dim)
        )  # (batch_size, n_agents)
        return expanded_mask.all(dim=-1)
