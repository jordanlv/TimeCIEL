import torch

from abc import ABC, abstractmethod
from ..activation_function import ActivationInterface
from ..internal_model import InternalModelInterface


class LearningRule(ABC):
    @abstractmethod
    def __call__(
        self,
        X: torch.Tensor,
        validity: ActivationInterface,
        internal_model: InternalModelInterface,
        bad: torch.Tensor,
        activated: torch.BoolTensor,
        neighbors: torch.BoolTensor,
        n_activated: torch.Tensor,
        n_neighbors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class IfNoActivatedAndNoNeighbors(LearningRule):
    def __call__(
        self,
        X,
        validity,
        internal_model,
        bad,
        activated,
        neighbors,
        n_activated,
        n_neighbors,
        path,
    ):
        batch_size = X.size(0)
        device = X.device

        n_agents = validity.n_agents
        # solve incompetence 1
        mask_inc1 = (n_activated == 0) & (n_neighbors == 0)  # (batch_size,)
        agents_to_create = mask_inc1  # which points to use to create new agents
        activation_to_update = torch.zeros(
            (n_agents, batch_size), dtype=torch.bool, device=device
        )
        agents_to_destroy = torch.zeros((n_agents,), dtype=torch.bool, device=device)
        return (
            agents_to_create,
            activation_to_update,
            agents_to_destroy,
        )


class IfNoActivated(LearningRule):
    def __call__(
        self,
        X,
        validity,
        internal_model,
        bad,
        activated,
        neighbors,
        n_activated,
        n_neighbors,
        path,
    ):
        batch_size = X.size(0)
        device = X.device

        n_agents = validity.n_agents
        # solve incompetence 2
        mask_inc2 = (n_activated == 0) & (n_neighbors > 0)  # (batch_size,)
        if validity.n_agents > 0:
            immediate_expandables_masked = validity.immediate_expandable(
                X[mask_inc2],
                path[mask_inc2, :, :, :],
            )  # (batch_size, n_agents)

            immediate_expandables = torch.zeros(
                (batch_size, n_agents), dtype=torch.bool, device=device
            )
            immediate_expandables[mask_inc2, :] = immediate_expandables_masked

            n_expand_candidates = torch.sum(
                immediate_expandables, dim=-1
            )  # (batch_size,)

            activation_to_update = immediate_expandables.T
            agents_to_create = mask_inc2 & (n_expand_candidates == 0)
            agents_to_destroy = torch.zeros(n_agents, dtype=torch.bool, device=device)
            return (
                agents_to_create,
                activation_to_update,
                agents_to_destroy,
            )
        return (
            torch.zeros(batch_size, dtype=torch.bool, device=device),
            torch.zeros((n_agents, batch_size), dtype=torch.bool, device=device),
            torch.zeros((n_agents,), dtype=torch.bool, device=device),
        )


class IfActivated(LearningRule):
    def __call__(
        self,
        X,
        validity,
        internal_model,
        bad,
        activated,
        neighbors,
        n_activated,
        n_neighbors,
        path,
    ):
        batch_size = X.size(0)
        device = X.device

        n_agents = validity.n_agents
        # solve inaccuracy
        mask_inac = n_activated > 0
        if validity.n_agents > 0:
            activation_to_update = activated.T & bad
            agents_to_create = torch.zeros(batch_size, dtype=torch.bool, device=device)
            agents_to_destroy = torch.zeros(n_agents, dtype=torch.bool, device=device)
            return (
                agents_to_create,
                activation_to_update,
                agents_to_destroy,
            )
        return (
            torch.zeros(batch_size, dtype=torch.bool, device=device),
            torch.zeros((n_agents, batch_size), dtype=torch.bool, device=device),
            torch.zeros((n_agents,), dtype=torch.bool, device=device),
        )


class SimpleDestroy(LearningRule):
    def __init__(self, imbalance_th=20):
        self.imbalance_th = imbalance_th

    def __call__(
        self,
        X,
        validity,
        internal_model,
        bad,
        activated,
        neighbors,
        n_activated,
        n_neighbors,
        path,
    ):
        batch_size = X.size(0)
        device = X.device
        n_agents = validity.n_agents
        balanced = validity.bads - validity.goods
        agents_to_destroy = (balanced > self.imbalance_th) | (
            (validity.orthotopes[:, :, :, 1] - validity.orthotopes[:, :, :, 0])
            .prod(dim=-1)
            .mean(dim=-1)
            <= 0
        ).any(dim=-1)
        return (
            torch.zeros(batch_size, dtype=torch.bool, device=device),
            torch.zeros((n_agents, batch_size), dtype=torch.bool, device=device),
            agents_to_destroy.view(-1),
        )
