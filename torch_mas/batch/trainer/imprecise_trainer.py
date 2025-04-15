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


class ImpreciseTrainer:
    def __init__(
        self,
        activation: ActivationInterface,
        internal_model: InternalModelInterface,
        R: list | float,
        bad_th: float,
        imprecise_th: float,
        kernel_size: list | int,
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

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]

        self.kernel_size = torch.as_tensor(kernel_size, device=device)

        self.neighborhood_sides = torch.as_tensor(self.R, device=device)
        self.imprecise_th = imprecise_th
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

        # TODO smart create agents which if 2 time series (of the same class)
        #  have a common part then agent is created at this emplacement

        batch_size = X[agents_to_create].size(0)

        kernels = self.kernel_size[
            torch.randint(0, self.kernel_size.size(0), (batch_size,))
        ]

        timesteps = torch.empty((batch_size,))

        for i, kernel in enumerate(kernels):
            timesteps[i] = torch.randint(0, X.size(1) - kernel, (1,))

        row_indices = torch.arange(X.size(1)).unsqueeze(1).repeat(1, batch_size).T

        bool_tensor = (row_indices >= timesteps.unsqueeze(-1)) & (
            row_indices < (timesteps + kernels).unsqueeze(-1)
        )

        self.activation.create(
            X[agents_to_create], side_lengths[agents_to_create], bool_tensor
        )
        self.internal_model.create(X[agents_to_create])

        models_to_init = torch.eye(X.size(0))
        models_to_init = models_to_init[agents_to_create]

        return models_to_init  # (n_created, batch_size)

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
        good = scores <= self.imprecise_th  # (n_agents, batch_size)
        bad = scores > self.bad_th  # (n_agents, batch_size)

        return good, bad

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, can_create=True):
        batch_size, seq_len, input_dim = X.size()

        neighbors_hyperrectangle, neighbors_agents = self.activation.neighbors(
            X, self.neighborhood_sides
        )  # (batch_size, n_agents)
        n_neighbors = torch.count_nonzero(neighbors_agents, dim=-1)  # (batch_size,)
        _, activated_agents = self.activation.activated(X)  # (batch_size, n_agents)
        n_activated = torch.count_nonzero(activated_agents, dim=-1)  # (batch_size,)
        maturity = self.internal_model.maturity(
            torch.ones(self.n_agents, dtype=torch.bool)
        )  # (n_agents, 1)
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

        good, bad = self.feedbacks(
            propositions, scores, neighbors_agents.T, n_neighbors
        )

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
                _models_to_update,
                _agents_to_destroy,
            ) = learning_rule(
                X,
                self.activation,
                self.internal_model,
                good,
                bad,
                activated_agents,
                neighbors_agents,
                n_activated,
                n_neighbors,
                maturity,
            )

            agents_to_create |= _agents_to_create
            hypercubes_to_update |= _activation_to_update
            models_to_update |= _models_to_update
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

        models_to_init = self.create_agents(X, agents_to_create, radius)
        models_to_update = torch.vstack([models_to_update, models_to_init])
        self.internal_model.update(X, y, models_to_update, self.activation.timesteps)

        # destroy agents
        _to_destroy = torch.zeros(self.n_agents, dtype=torch.bool)

        n_to_destroy = agents_to_destroy.sum()
        if n_to_destroy > 0:
            _to_destroy[: agents_to_destroy.size(0)] = agents_to_destroy
            self.destroy_agents(_to_destroy)

    def fit(self, dataset):
        self.agents_over_epoch = []
        n_samples = len(dataset)
        for e in range(self.n_epochs):
            indices = torch.arange(n_samples)
            shuffled_indices = indices[torch.randperm(indices.size(0))]
            batches = shuffled_indices.split(self.batch_size)
            for batch in batches:
                X, y = dataset[batch]
                self.partial_fit(X, y)

            # if e < self.n_epochs - 1:
            #     self.destroy_agents((self.activation.used < 1).squeeze())
            #     self.activation.used = torch.zeros_like(self.activation.used)

            self.agents_over_epoch.append(self.n_agents)

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
