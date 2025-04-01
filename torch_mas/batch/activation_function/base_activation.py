import torch
import copy

from .activation_interface import ActivationInterface
from ...common.orthotopes.base import (
    batch_intersect_signal,
    batch_intersect_signals,
    batch_create_temporal_hypercube,
    batch_intersect_temporal_hypercubes,
    batch_update_temporal_hypercube,
    batch_dist_signals_to_border,
)

batch_update_temporal_hypercubes = torch.vmap(
    batch_update_temporal_hypercube, in_dims=(None, 0, None)
)
batch_batch_intersect_signal = torch.vmap(batch_intersect_signal, in_dims=(0, None, 0))
batch_batch_update_temporal_hypercube = torch.vmap(
    batch_update_temporal_hypercube, in_dims=(None, 0, 0)
)


class BaseActivation(ActivationInterface):
    def __init__(self, input_dim, output_dim, alpha, seq_len, device="cpu", **kwargs):
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.alpha = torch.tensor(alpha, device=self.device)

        self.orthotopes: torch.Tensor = torch.empty(
            0, seq_len, input_dim, 2, device=device
        )  # (n_agents, seq_len, input_dim, 2) Tensor of orthotopes

        self.timesteps: torch.Tensor = torch.empty(
            0, seq_len, device=device, dtype=torch.bool
        )  # (n_agents, seq_len) Tensor of orthotopes

        self.goods: torch.Tensor = torch.empty(
            0,
            1,
            dtype=torch.long,
            device=device,
        )  # (n_agents, 1)
        self.bads: torch.Tensor = torch.empty(
            0,
            1,
            dtype=torch.long,
            device=device,
        )  # (n_agents, 1)

        self.used: torch.Tensor = torch.empty(
            0,
            1,
            dtype=torch.long,
            device=device,
        )  # (n_agents, 1)

    @property
    def n_agents(self):
        return self.orthotopes.size(0)

    def destroy(self, agents_mask):
        self.orthotopes = self.orthotopes[~agents_mask]
        self.timesteps = self.timesteps[~agents_mask]
        self.goods = self.goods[~agents_mask]
        self.bads = self.bads[~agents_mask]
        self.used = self.used[~agents_mask]

    def create(self, X, side_lengths, timesteps):
        lows = X - side_lengths / 2
        highs = X + side_lengths / 2

        orthotopes = torch.stack([lows, highs], dim=-1)
        self.orthotopes = torch.vstack([self.orthotopes, orthotopes])
        self.timesteps = torch.vstack([self.timesteps, timesteps])

        batch_size = X.size(0)
        goods = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        bads = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        used = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)
        self.goods = torch.vstack([self.goods, goods])
        self.bads = torch.vstack([self.bads, bads])
        self.used = torch.vstack([self.used, used])

    def activated(self, X):
        agents_mask = batch_intersect_signals(self.orthotopes, self.timesteps, X)

        self.used += (
            (agents_mask.sum(dim=-1) == self.timesteps.sum(dim=-1).unsqueeze(0))
            .sum(0)
            .unsqueeze(-1)
        )

        return agents_mask, agents_mask.sum(dim=-1) == self.timesteps.sum(
            dim=-1
        ).unsqueeze(0)

    def neighbors(self, X, side_length):
        neighborhood = batch_create_temporal_hypercube(
            X, side_length.repeat(X.size(0), 1, 1)
        )

        neighbor_mask = batch_intersect_temporal_hypercubes(
            neighborhood, self.orthotopes, self.timesteps
        )

        self.used += (
            (neighbor_mask.sum(dim=-1) == self.timesteps.sum(dim=-1).unsqueeze(0))
            .sum(0)
            .unsqueeze(-1)
        )

        return (
            neighbor_mask,
            neighbor_mask.sum(dim=-1) == self.timesteps.sum(dim=-1).unsqueeze(0),
        )

    def immediate_expandable(self, X):

        alphas = torch.zeros(
            (self.n_agents, X.size(1)), device=self.device
        )  # (batch_size, n_agents)

        alphas = torch.where(self.timesteps, self.alpha, 0)

        expanded_neighbors = batch_update_temporal_hypercubes(
            self.orthotopes,
            X,
            alphas,
        )  # (batch_size, n_agents, in_dim, 2) possible shape of each agent for each x

        expanded_mask = batch_batch_intersect_signal(
            expanded_neighbors,  # (batch_size*n_agents, in_dim, 2)
            self.timesteps,
            X,  # (batch_size, in_dim)
        )  # (batch_size, n_agents)
        return expanded_mask.all(dim=-1)

    def update(self, X, agents_mask, bad, no_activated, neighbors_hyperrectangle):
        batch_size, seq_len, _ = X.size()
        self.goods += (~bad).sum(0).view(self.n_agents, 1)
        self.bads += bad.sum(0).view(self.n_agents, 1)

        alphas = torch.zeros(
            (batch_size, self.n_agents, seq_len), device=self.device
        )  # (batch_size, n_agents)

        # TODO alpha = 0 when not in self.timesteps because useless to update them
        alphas = torch.where(
            agents_mask.unsqueeze(-1)
            & no_activated.unsqueeze(-1).unsqueeze(-1)
            & ~bad.unsqueeze(-1)
            & neighbors_hyperrectangle,
            self.alpha,
            alphas,
        )
        alphas = torch.where(
            agents_mask.unsqueeze(-1)
            & ~no_activated.unsqueeze(-1).unsqueeze(-1)
            & bad.unsqueeze(-1)
            & neighbors_hyperrectangle,
            -self.alpha,
            alphas,
        )

        updated_orthotopes = batch_batch_update_temporal_hypercube(
            self.orthotopes, X, alphas
        )  # (batch_size, n_agents, in_dim, 2)
        deltas = (
            updated_orthotopes - self.orthotopes
        )  # (batch_size, n_agents, in_dim, 2)
        deltas = deltas.sum(dim=0)  # (n_agents, in_dim, 2)
        self.orthotopes += deltas

    def dist_to_border(self, X, agents_mask):
        return batch_dist_signals_to_border(self.orthotopes[agents_mask], X)

    def clone(self):
        cloned_self = copy.copy(self)  # shallow copy
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(cloned_self, attr_name, attr_value.clone())
        return cloned_self
