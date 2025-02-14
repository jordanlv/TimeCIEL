import torch

from typing import Callable
from ..activation_function import ActivationInterface
from ..internal_model import InternalModelInterface
from ...common.orthotopes.base import batch_intersect_signals
from .learning_rules import (
    LearningRule,
    IfActivated,
    IfNoActivated,
    IfNoActivatedAndNoNeighbors,
)


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
        bad_th: float,
        learning_rules: list[LearningRule] = [
            IfNoActivatedAndNoNeighbors(),
            IfNoActivated(),
            IfActivated(),
        ],
        criterion: Callable = mse_loss,
        n_epochs: int = 10,
        batch_size: int = 64,
        device="cpu",
    ):
        self.activation = activation
        self.internal_model = internal_model
        self.learning_rules = learning_rules
        self.criterion = criterion

        if isinstance(R, float):
            R = [R]
        self.R = torch.as_tensor(R, device=device)
        self.neighborhood_sides = torch.as_tensor(self.R, device=device)
        self.bad_th = bad_th
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

    @property
    def n_agents(self):
        return self.activation.n_agents

    def destroy_agents(self, agents_to_destroy):
        """Destroy Agents

        Args:
            agents_to_destroy (torch.BoolTensor): (n_agents,)
        """
        self.activation.destroy(agents_to_destroy)
        self.internal_model.destroy(agents_to_destroy)

    def create_agents(self, X, agents_to_create, side_lengths):
        """
        Create agents for time series signals using a greedy set cover approach.

        Each candidate agent is associated with a sequence of hypercubes (one per timestep).
        A candidate is said to cover a signal if, at every timestep, the candidate’s hypercube
        covers the signal’s coordinate at that timestep.

        Args:
            X (Tensor): (batch_size, seq_len, n_dim)
            agents_to_create (BoolTensor): (batch_size,)
            side_lengths (Tensor): (batch_size, seq_len, n_dim)

        Returns:
            BoolTensor: (n_created, batch_size)
        """
        batch_size = X.size(0)

        lows = X - side_lengths / 2
        highs = X + side_lengths / 2
        hypercubes = torch.stack(
            [lows, highs], dim=-1
        )  # (batch_size, seq_len, n_dim, 2)

        coverage_time = batch_intersect_signals(
            hypercubes, X
        )  # (batch_size, batch_size, seq_len)

        coverage_matrix = coverage_time.all(dim=-1)

        valid_candidate_mask = agents_to_create.unsqueeze(1)
        coverage_matrix = coverage_matrix & valid_candidate_mask

        signals_to_cover = agents_to_create.clone()
        current_coverage = torch.logical_not(signals_to_cover)

        selected_candidates = torch.zeros(
            batch_size, dtype=torch.bool, device=self.device
        )

        # Greedy loop: select the candidate that covers the most _new_ signals.
        while not current_coverage.all():
            union_coverage = (
                current_coverage.unsqueeze(0) | coverage_matrix
            )  # (batch_size, batch_size)
            new_coverage = union_coverage ^ current_coverage.unsqueeze(0)
            new_signals_count = new_coverage.sum(dim=1)  # (batch_size,)

            new_signals_count = torch.where(
                selected_candidates,
                torch.full_like(new_signals_count, -1),
                new_signals_count,
            )

            max_new = int(new_signals_count.max().item())
            if max_new <= 0:
                break

            selected_idx = int(torch.argmax(new_signals_count).item())
            selected_candidates[selected_idx] = True

            current_coverage = current_coverage | coverage_matrix[selected_idx]

        agents_selected = selected_candidates

        self.activation.create(X[agents_selected], side_lengths[agents_selected])
        self.internal_model.create(X[agents_selected])

        models_to_init = coverage_matrix[agents_selected]  # (n_created, batch_size)
        return models_to_init

    def feedbacks(self, propositions, scores, neighbors, n_neighbors):
        """_summary_

        Args:
            propositions (Tensor): (n_agents, batch_size, output_dim)
            scores (Tensor): (n_agents, batch_size)
            neighbors (Tensor): (n_agents, batch_size)
            n_neighbors (Tensor): (batch_size,)

        Returns:
            bad (n_agents, batch_size)
        """
        bad = scores > self.bad_th  # (n_agents, batch_size)

        return bad

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, can_create=True):
        batch_size, seq_len, input_dim = X.size()

        neighbors_hyperrectangle, neighbors_agents = self.activation.neighbors(
            X, self.neighborhood_sides
        )  # (batch_size, n_agents)
        n_neighbors = torch.count_nonzero(neighbors_agents, dim=-1)  # (batch_size,)
        _, activated_agents = self.activation.activated(X)  # (batch_size, n_agents)
        n_activated = torch.count_nonzero(activated_agents, dim=-1)  # (batch_size,)

        agents_to_predict = neighbors_agents.T.sum(-1) > 0
        predictions = self.internal_model(
            X, agents_to_predict
        )  # (n_agents_to_predict, batch_size, out_dim)
        propositions = torch.zeros(
            (self.n_agents, batch_size, predictions.size(-1)),
            device=self.device,
        )  # (n_agents, batch_size, out_dim)
        propositions[agents_to_predict] = predictions
        scores = self.criterion(propositions, y)  # (n_agents, batch_size)

        bad = self.feedbacks(propositions, scores, neighbors_agents.T, n_neighbors)

        agents_to_create = torch.zeros(
            (batch_size,), dtype=torch.bool, device=self.device
        )  # (batch_size,)
        hypercubes_to_update = torch.zeros(
            (self.n_agents, batch_size),
            dtype=torch.bool,
            device=self.device,
        )  # (n_agents, batch_size)
        models_to_update = torch.zeros(
            (self.n_agents, batch_size),
            dtype=torch.bool,
            device=self.device,
        )  # (n_agents, batch_size) batch points to use to update each agent
        agents_to_destroy = torch.zeros(
            (self.n_agents,), dtype=torch.bool, device=self.device
        )  # (batch_sizen_agents,)

        for learning_rule in self.learning_rules:
            (
                _agents_to_create,
                _activation_to_update,
                _agents_to_destroy,
            ) = learning_rule(
                X,
                self.activation,
                self.internal_model,
                bad,
                activated_agents,
                neighbors_agents,
                n_activated,
                n_neighbors,
            )

            agents_to_create |= _agents_to_create
            hypercubes_to_update |= _activation_to_update
            agents_to_destroy |= _agents_to_destroy

        if self.n_agents > 0:
            # update orthotopes
            no_activated = (n_activated == 0) & (n_neighbors > 0)
            self.activation.update(
                X,
                hypercubes_to_update.T,
                bad.T,
                no_activated,
                neighbors_hyperrectangle,
            )

        # create new agents

        # get mean over all seq_len,input_dim if neighbor else self.R
        diff = self.activation.orthotopes[..., 1] - self.activation.orthotopes[..., 0]

        activation_extanded = diff.unsqueeze(0)
        neighbors_expanded = neighbors_agents.unsqueeze(-1).unsqueeze(-1)

        masked_activation = activation_extanded * neighbors_expanded

        sum_masked = masked_activation.sum(dim=1)

        cond = n_neighbors > 1

        radius = torch.empty(batch_size, seq_len, input_dim, device=self.device)
        radius[cond] = (sum_masked / n_neighbors.view(batch_size, 1, 1))[cond]
        radius[~cond] = self.R.repeat(batch_size, 1, 1)[~cond]

        if can_create:
            models_to_init = self.create_agents(X, agents_to_create, radius)
            models_to_update = torch.zeros(
                (self.n_agents, batch_size), device=self.device, dtype=torch.bool
            )
            if models_to_init.size(0) > 0:
                models_to_update[-models_to_init.size(0) :, :] = models_to_init
            self.internal_model.update(X, y, models_to_update)

        # destroy agents
        _to_destroy = torch.zeros(self.n_agents, dtype=torch.bool)

        n_to_destroy = agents_to_destroy.sum()
        if n_to_destroy > 0:
            _to_destroy[: agents_to_destroy.size(0)] = agents_to_destroy
            self.destroy_agents(_to_destroy)

    def fit(self, dataset):
        n_samples = len(dataset)
        for _ in range(self.n_epochs):
            indices = torch.arange(n_samples)
            shuffled_indices = indices[torch.randperm(indices.size(0))]
            batches = shuffled_indices.split(self.batch_size)
            for batch in batches:
                X, y = dataset[batch]
                self.partial_fit(X, y)

    def predict(self, X: torch.Tensor):
        batch_size = X.size(0)
        agents_mask = torch.ones(self.n_agents, dtype=torch.bool, device=self.device)

        res = torch.empty((batch_size,), device=self.device)
        y_hat = self.internal_model(X, agents_mask).squeeze(-1).transpose(0, 1)

        # activated
        _, activated_mask = self.activation.activated(X)  # (batch_size, n_orthotopes)

        preds_activation = y_hat.clone()
        preds_activation[~activated_mask] = torch.nan

        res, _ = preds_activation.nanmedian(dim=-1)

        non_pred_mask = ~activated_mask.all(dim=-1)

        # neighbors
        _, neighbor_mask = self.activation.neighbors(X, self.neighborhood_sides)
        preds_neighbors = y_hat.clone()

        preds_neighbors[~neighbor_mask] = torch.nan

        mask = non_pred_mask & (neighbor_mask.sum(dim=-1) > 0)
        res[mask] = preds_neighbors.nanmedian(dim=-1).values[mask]

        # closest
        distances = self.activation.dist_to_border(X[non_pred_mask], agents_mask).mean(
            dim=-1
        )
        closest_mask = torch.zeros_like(distances, dtype=torch.bool).scatter(
            1, distances.argsort()[:, :1], True
        )
        res[res.isnan()] = y_hat[closest_mask][res.isnan()]

        return res
