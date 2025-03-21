import torch

from typing import Callable
from ..activation_function import ActivationInterface
from ..internal_model import InternalModelInterface
from ...common.orthotopes.base import batch_sides


def mse_loss(y_pred: torch.FloatTensor, y: torch.FloatTensor):
    """Calculate the mean squared error

    Args:
        y_pred (FloatTensor): (n_predictions, output_dim)
        y (FloatTensor): (output_dim,)

    Returns:
        Tensor: (n_predictions, 1)
    """
    return ((y_pred - y) ** 2).mean(dim=-1)


class BaseTrainer:
    def __init__(
        self,
        activation: ActivationInterface,
        internal_model: InternalModelInterface,
        R: list | float,
        imprecise_th: float,
        bad_th: float,
        criterion: Callable = mse_loss,
        n_epochs: int = 10,
        device="cpu",
    ):
        self.activation = activation
        self.internal_model = internal_model
        self.criterion = criterion

        if isinstance(R, float):
            R = [R]
        self.R = torch.as_tensor(R, device=device)
        self.neighborhood_sides = torch.as_tensor(self.R, device=device)
        self.imprecise_th = imprecise_th
        self.bad_th = bad_th
        self.n_epochs = n_epochs
        self.device = device

    @property
    def n_agents(self):
        return self.activation.n_agents

    def create_agents(self, X, side_lengths):
        """Create agents

        Args:
            X (Tensor): (batch_size, n_dim)
            side_lengths (Tensor): (batch_size, n_dim)

        Returns:
            BoolTensor: (n_created, batch_size,)
        """
        created_idxs = torch.arange(0, X.size(0), dtype=torch.long) + self.n_agents
        self.activation.create(X, side_lengths)
        self.internal_model.create(X)

        return created_idxs

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor):
        neighborhood_agents = self.activation.neighbors(
            X, self.neighborhood_sides
        ).squeeze(0)
        n_neighbors = torch.count_nonzero(neighborhood_agents)
        activated_agents = self.activation.activated(X)
        n_activated = torch.count_nonzero(activated_agents)
        agents_to_update = torch.empty(0, device=self.device)
        if n_activated == 0 and n_neighbors == 0:
            created_idxs = self.create_agents(X, self.R)
            agents_to_update = torch.concat([agents_to_update, created_idxs])

        if n_activated == 0 and n_neighbors > 0:
            expanded_mask = self.activation.immediate_expandable(
                X, neighborhood_agents
            ).squeeze(-1)
            expanded_idxs = torch.arange(self.n_agents, device=self.device)[
                neighborhood_agents
            ][expanded_mask]
            activated_maturity = self.internal_model.maturity(expanded_idxs).squeeze(-1)
            expanded_idxs = expanded_idxs[activated_maturity]
            n_expand_candidates = len(expanded_idxs)
            if n_expand_candidates > 0:
                predictions = self.internal_model(X, expanded_idxs)
                score = self.criterion(predictions, y).squeeze(-1)  # (n_predictions,)
                good = score <= self.imprecise_th
                bad = score > self.bad_th

                self.activation.update(X, expanded_idxs, good, bad, no_activated=True)

                agents_to_update = torch.arange(self.n_agents, device=self.device)[
                    expanded_idxs
                ][~bad & ~good]
                if bad.all():
                    created_idxs = self.create_agents(X, self.R)
                    agents_to_update = torch.concat([agents_to_update, created_idxs])
            else:
                radius = self.R
                if n_neighbors > 1:
                    radius = batch_sides(
                        self.activation.orthotopes[neighborhood_agents]
                    ).mean()
                created_idxs = self.create_agents(X, radius)
                agents_to_update = torch.concat([agents_to_update, created_idxs])
        if n_activated > 0:
            agents_mask = activated_agents.squeeze(dim=-1)
            predictions = self.internal_model(X, agents_mask)
            score = self.criterion(predictions, y).squeeze(-1)  # (n_predictions,)
            activated_maturity = self.internal_model.maturity(agents_mask).squeeze(-1)

            good = score <= self.imprecise_th
            bad = score > self.bad_th

            self.activation.update(X, agents_mask, good, bad, no_activated=False)

            agents_to_update = torch.arange(self.n_agents, device=self.device)[
                agents_mask
            ][~bad & ~good | ~activated_maturity]
        if agents_to_update.size(0) > 0:
            self.internal_model.update(X, y, agents_to_update.long())

    def fit(self, dataset):
        n_samples = len(dataset)
        for _ in range(self.n_epochs):
            indices = torch.arange(n_samples)
            shuffled_indices = indices[torch.randperm(indices.size(0))]
            batches = shuffled_indices.split(1)
            for batch in batches:
                X, y = dataset[batch]
                self.partial_fit(X, y)

    def predict(self, X: torch.Tensor):
        batch_size = X.size(0)
        agents_mask = torch.ones(self.n_agents, dtype=torch.bool, device=self.device)

        res = torch.empty((batch_size,), device=self.device)
        y_hat = self.internal_model(X, agents_mask).squeeze(-1)

        # activated
        activated_mask = self.activation.activated(X)  # (batch_size, n_orthotopes)

        preds_activation = y_hat.clone()
        preds_activation[~activated_mask] = torch.nan

        res, _ = preds_activation.nanmedian(dim=0)

        non_pred_mask = ~activated_mask.all(dim=0)

        # neighbors
        neighbor_mask = self.activation.neighbors(X, self.neighborhood_sides).transpose(
            0, 1
        )
        preds_neighbors = y_hat.clone()

        preds_neighbors[~neighbor_mask] = torch.nan
        mask = non_pred_mask & neighbor_mask.sum(dim=0)
        res[mask], _ = preds_neighbors.nanmedian(dim=0)

        # closest
        distances = self.activation.dist_to_border(X[non_pred_mask], agents_mask).mean(
            dim=-1
        )
        closest_mask = torch.zeros_like(distances, dtype=torch.bool).scatter(
            1, distances.argsort()[:, :1], True
        )
        res[res.isnan()] = y_hat.transpose(0, 1)[closest_mask][res.isnan()]

        return res
