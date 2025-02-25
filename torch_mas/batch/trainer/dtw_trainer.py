import torch

from . import BaseTrainer
from torch_mas.common.orthotopes.dtw import batch_dtw
import numpy as np


class DTWTrainer(BaseTrainer):

    def partial_fit(self, X, y, can_create=True):

        batch_size, seq_len, input_dim = X.size()

        # Perform DTW
        costs, dist, path = batch_dtw(self.activation.orthotopes, X, 0.1)

        # print(costs.shape, dist.shape, path.shape)

        neighbors = self.activation.neighbors(X, path, self.R)  # (batch_size, n_agents)
        n_neighbors = torch.count_nonzero(neighbors, dim=-1)  # (batch_size,)
        activated = self.activation.activated(costs)  # (batch_size, n_agents)
        n_activated = torch.count_nonzero(activated, dim=-1)  # (batch_size,)

        agents_to_predict = neighbors.T.sum(-1) > 0

        predictions = self.internal_model(
            X, agents_to_predict
        )  # (n_agents_to_predict, batch_size, out_dim)

        propositions = torch.zeros(
            (self.n_agents, batch_size, predictions.size(-1)),
            device=self.device,
        )  # (n_agents, batch_size, out_dim)
        propositions[agents_to_predict] = predictions
        scores = self.criterion(propositions, y)  # (n_agents, batch_size)

        bad = self.feedbacks(propositions, scores, neighbors.T, n_neighbors)

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
                activated,
                neighbors,
                n_activated,
                n_neighbors,
                path,
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
                path,  # garder que les bons paths
            )

        # create new agents
        # get mean over all seq_len,input_dim if neighbor else self.R
        diff = self.activation.orthotopes[..., 1] - self.activation.orthotopes[..., 0]

        activation_extanded = diff.unsqueeze(0)
        neighbors_expanded = neighbors.unsqueeze(-1).unsqueeze(-1)

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

    def predict(self, X: torch.Tensor):
        batch_size = X.size(0)
        agents_mask = torch.ones(self.n_agents, dtype=torch.bool, device=self.device)

        res = torch.empty((batch_size,), device=self.device)
        y_hat = self.internal_model(X, agents_mask).squeeze(-1).transpose(0, 1)

        costs, dist, path = batch_dtw(self.activation.orthotopes, X, 0.1)

        # activated
        activated_mask = self.activation.activated(costs)  # (batch_size, n_orthotopes)

        preds_activation = y_hat.clone()
        preds_activation[~activated_mask] = torch.nan

        res, _ = preds_activation.nanmedian(dim=-1)

        non_pred_mask = ~activated_mask.all(dim=-1)

        # neighbors
        neighbor_mask = self.activation.neighbors(X, path, self.R)
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
