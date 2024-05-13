import torch
import numpy as np
from torch_mas.agents import Agents
from torch_mas.hypercubes import *
from torch_mas.linear_models import *


class Head:
    def __init__(
        self,
        input_dim,
        output_dim,
        R: list | float,
        imprecise_th,
        bad_th,
        alpha,
        memory_length=20,
        n_epochs=10,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        if isinstance(R, float):
            R = [R]
        self.R = torch.FloatTensor(R)
        self.neighborhood_sides = torch.FloatTensor(self.R)
        self.imprecise_th = imprecise_th
        self.bad_th = bad_th
        self.alpha = alpha
        self.memory_length = memory_length
        self.n_epochs = n_epochs

        self._step = 0

    def score(self, y_pred: torch.FloatTensor, y: torch.FloatTensor):
        """Calculate the mean squared error

        Args:
            y_pred (FloatTensor): (n_predictions, output_dim)
            y (FloatTensor): (output_dim,)

        Returns:
            Tensor: (n_predictions, 1)
        """
        return ((y_pred - y) ** 2).mean(dim=-1)

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor):
        neighborhood_agents = self.agents.neighbors(X, self.neighborhood_sides)
        n_neighbors = torch.count_nonzero(neighborhood_agents)
        activated_agents = self.agents.activated(X.squeeze(0))
        n_activated = torch.count_nonzero(activated_agents)

        agents_to_update = torch.empty(0)

        if n_activated == 0 and n_neighbors == 0:
            created_idxs = self.agents.create_agents(X, self.R)
            agents_to_update = torch.concat([agents_to_update, created_idxs])

        if n_activated == 0 and n_neighbors > 0:
            expanded_neighbors = batch_update_hypercube(
                self.agents.hypercubes[neighborhood_agents],
                X.squeeze(0),
                torch.full((n_neighbors,), self.alpha),
            )
            expanded_mask = batch_intersect_point(expanded_neighbors, X)
            expanded_idxs = torch.arange(self.agents.n_agents)[neighborhood_agents][
                expanded_mask
            ]
            activated_maturity = self.agents.maturity(expanded_idxs).squeeze(-1)
            expanded_idxs = expanded_idxs[activated_maturity]
            n_expand_candidates = len(expanded_idxs)

            if n_expand_candidates > 0:
                predictions = self.agents.predict(X, expanded_idxs)
                score = self.score(predictions, y).squeeze(-1)  # (n_predictions,)
                good = score <= self.imprecise_th
                bad = score > self.bad_th

                alphas = torch.zeros((n_expand_candidates, 1))
                alphas[~bad] = self.alpha  # expansion
                self.agents.update_hypercube(
                    X, agents_idxs=expanded_idxs, alphas=alphas
                )

                agents_to_update = torch.arange(self.agents.n_agents)[expanded_idxs][
                    ~bad & ~good
                ]
                if bad.all():
                    created_idxs = self.agents.create_agents(X, self.R)
                    agents_to_update = torch.concat([agents_to_update, created_idxs])
            else:
                radius = self.R
                if n_neighbors > 1:
                    radius = batch_sides(
                        self.agents.hypercubes[neighborhood_agents]
                    ).mean(0)
                created_idxs = self.agents.create_agents(X, radius)
                agents_to_update = torch.concat([agents_to_update, created_idxs])
        if n_activated > 0:
            predictions = self.agents.predict(X, activated_agents)
            score = self.score(predictions, y).squeeze(-1)  # (n_predictions,)
            activated_maturity = self.agents.maturity(activated_agents).squeeze(-1)

            good = score <= self.imprecise_th
            bad = score > self.bad_th

            alphas = torch.zeros((n_activated, 1))
            alphas[bad & activated_maturity] = -self.alpha  # retraction
            self.agents.update_hypercube(X, agents_idxs=activated_agents, alphas=alphas)

            agents_to_update = torch.arange(self.agents.n_agents)[activated_agents][
                ~bad & ~good | ~activated_maturity
            ]
        if agents_to_update.size(0) > 0:
            self.agents.update_model(X, y, agents_to_update.long())

    def fit(self, dataset):
        self.agents = Agents(self.input_dim, self.output_dim, self.memory_length)

        n_samples = len(dataset)
        idxs = np.arange(0, n_samples)
        np.random.shuffle(idxs)

        for _ in range(self.n_epochs):
            for idx in idxs:
                X, y = dataset[torch.LongTensor([idx])]
                self.partial_fit(X, y)

    def predict(self, X):
        """Make a prediction

        Args:
            X (Tensor): (batch_size, input_dim)

        Returns:
            Tensor: (batch_size, output_dim)
        """
        agents_mask = torch.ones(self.agents.n_agents, dtype=torch.bool)
        neighborhoods = batch_create_hypercube(
            X,
            self.neighborhood_sides.expand(
                (X.size(0),) + self.neighborhood_sides.size()
            ),
        )
        neighborhood_mask = batch_intersect_hypercubes(
            neighborhoods, self.agents.hypercubes
        )
        maturity_mask = self.agents.maturity(agents_mask)
        activated_mask = batch_intersect_points(self.agents.hypercubes, X)
        mask = neighborhood_mask & maturity_mask.T

        y_hat = self.agents.predict(X, agents_mask).transpose(0, 1)
        W = mask.float().unsqueeze(-1)

        return (y_hat * W).sum(1) / W.sum(1)
