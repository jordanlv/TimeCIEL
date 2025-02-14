import torch
import copy
from .activation_interface import ActivationInterface
from ...common.orthotopes.base import (
    batch_create_temporal_hypercube,
    batch_intersect_temporal_hypercubes,
    batch_intersect_signals,
    batch_update_temporal_hypercube,
    batch_dist_signals_to_border,
    create_hypercube,
)


class TimeActivation(ActivationInterface):
    def __init__(self, seq_len, input_dim, output_dim, alpha, device="cpu", **kwargs):
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.alpha = torch.tensor(alpha, device=self.device)

        self.orthotopes: torch.Tensor = torch.empty(
            0, seq_len, input_dim, 2, device=device
        )  # (n_agents, seq_len, input_dim, 2) Tensor of orthotopes

    @property
    def n_agents(self):
        return self.orthotopes.size(0)

    def create(self, X, side_lengths):
        lows = X - side_lengths / 2
        highs = X + side_lengths / 2

        orthotopes = torch.stack([lows, highs], dim=-1)
        self.orthotopes = torch.vstack([self.orthotopes, orthotopes])

    def activated(self, X):
        agents_mask = batch_intersect_signals(self.orthotopes, X)
        return agents_mask.all(dim=-1)

    def neighbors(self, X, side_length):

        neighborhood = create_hypercube(X, side_length)

        neighbor_mask = batch_intersect_temporal_hypercubes(
            self.orthotopes, neighborhood
        )

        return neighbor_mask.sum(dim=-1) > neighbor_mask.shape[-1] * 0.5

    def immediate_expandable(self, X, agents_mask):
        n_agents = torch.count_nonzero(agents_mask)
        expanded_neighbors = batch_update_temporal_hypercube(
            self.orthotopes[agents_mask],
            X.squeeze(0),
            torch.full((n_agents,), self.alpha),
        )
        expanded_mask = batch_intersect_signals(expanded_neighbors, X)
        return expanded_mask.any(dim=-1)

    def update(self, X, agents_mask, good, bad, no_activated=False):
        n_agents = (
            agents_mask.size(0)
            if isinstance(agents_mask, torch.LongTensor)
            else agents_mask.count_nonzero()
        )
        alphas = torch.zeros((n_agents, 1))
        if no_activated:
            alphas[~bad] = self.alpha
        else:
            alphas[bad] = -self.alpha

        updated_orthotopes = batch_update_temporal_hypercube(
            self.orthotopes[agents_mask], X.squeeze(0), alphas
        )
        self.orthotopes[agents_mask] = updated_orthotopes

    def dist_to_border(self, X, agents_mask):
        return batch_dist_signals_to_border(self.orthotopes[agents_mask], X)

    def clone(self):
        cloned_self = copy.copy(self)  # shallow copy
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(cloned_self, attr_name, attr_value.clone())
        return cloned_self
