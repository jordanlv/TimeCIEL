import torch
import copy
from .model_interface import InternalModelInterface
from ...common.models.logreg import (
    batch_fit_logistic_regression,
    batch_predict_logistic_regression,
)


def _update_memory(X, y, mem_X, mem_y, mem_mask, batch_mask):
    """_summary_

    Args:
        X (Tensor): (batch_size, in_dim)
        y (Tensor): (batch_size, out_dim)
        mem_X (Tensor): (memory_length, in_dim)
        mem_y (Tensor): (memory_length, out_dim)
        mem_mask (Tensor): (memory_length,)
        agent_mask (Tensor): (batch_size,)
    """
    mem_length = mem_X.size(0)
    cat_masks = torch.cat([batch_mask, mem_mask])
    new_mem_X = torch.cat([X, mem_X])
    new_mem_y = torch.cat([y, mem_y])
    sorted_idxs = torch.argsort(cat_masks.float(), descending=True)

    return new_mem_X[sorted_idxs][:mem_length], new_mem_y[sorted_idxs][:mem_length]


_batch_update_memory = torch.vmap(_update_memory, in_dims=(None, None, 0, 0, 0, 0))


class Logreg(InternalModelInterface):
    def __init__(
        self,
        input_dim,
        seq_len,
        output_dim,
        n_class,
        memory_length,
        device="cpu",
        **kwargs
    ) -> None:
        self.device = device
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.memory_length = memory_length
        self.n_class = n_class

        self.feature_memories: torch.Tensor = torch.empty(
            0, memory_length, input_dim, dtype=torch.float, device=device
        )  # (n_agents, memory_length,) Tensor of features
        self.target_memories: torch.Tensor = torch.empty(
            0, memory_length, output_dim, dtype=torch.float, device=device
        )  # (n_agents, memory_length,) Tensor of targets
        self.memory_sizes: torch.Tensor = torch.empty(
            0, 1, dtype=torch.long, device=device
        )  # (n_agents, 1) Tensor of fill memory levels
        self.memory_ptr: torch.Tensor = torch.empty(
            0, 1, dtype=torch.long, device=device
        )  # (n_agents, 1) Tensor of fill memory levels

        self.models: torch.Tensor = torch.empty(
            0,
            input_dim + 1,
            n_class,
            dtype=torch.float,
            requires_grad=False,
            device=device,
        )  # (n_agents, input_dim+1, output_dim) Tensor of linear models

        self.base_prediction: torch.Tensor = torch.zeros(
            0, dtype=torch.float, requires_grad=False, device=device
        )  # (n_agents) Tensor of base prediction

    @property
    def n_agents(self):
        return self.models.size(0)

    def destroy(self, agents_mask):
        self.feature_memories = self.feature_memories[~agents_mask]
        self.target_memories = self.target_memories[~agents_mask]
        self.memory_sizes = self.memory_sizes[~agents_mask]
        self.memory_ptr = self.memory_ptr[~agents_mask]
        self.models = self.models[~agents_mask]
        self.base_prediction = self.base_prediction[~agents_mask]

    def create(self, X):
        batch_size = X.size(0)
        models = torch.zeros(
            (batch_size, self.input_dim + 1, self.n_class), device=self.device
        )
        base_prediction = torch.zeros(
            (batch_size,), dtype=torch.float, device=self.device
        )
        feature_memories = torch.zeros(
            (batch_size, self.memory_length, self.input_dim),
            dtype=torch.float,
            device=self.device,
        )
        target_memories = torch.zeros(
            (batch_size, self.memory_length, self.output_dim),
            dtype=torch.float,
            device=self.device,
        )
        memory_size = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        memory_ptr = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)

        self.models = torch.vstack([self.models, models])
        self.base_prediction = torch.vstack(
            [self.base_prediction.unsqueeze(-1), base_prediction.unsqueeze(-1)]
        ).squeeze(-1)
        self.feature_memories = torch.vstack([self.feature_memories, feature_memories])
        self.target_memories = torch.vstack([self.target_memories, target_memories])
        self.memory_sizes = torch.vstack([self.memory_sizes, memory_size])
        self.memory_ptr = torch.vstack([self.memory_ptr, memory_ptr])

    def maturity(self, agents_mask):
        return self.memory_sizes[agents_mask] > (self.input_dim + 1)

    def _update_memories(
        self, X: torch.Tensor, y: torch.Tensor, agent_mask: torch.BoolTensor
    ):
        mem_masks = (
            torch.arange(self.memory_length, device=self.device) < self.memory_sizes
        )  # (n_agents, memory_length)
        nb_to_add_per_agent = agent_mask.sum(-1)
        agents_to_update = nb_to_add_per_agent > 0
        self.memory_sizes = torch.clip(
            self.memory_sizes + nb_to_add_per_agent.view(-1, 1), max=self.memory_length
        )

        (
            self.feature_memories[agents_to_update],
            self.target_memories[agents_to_update],
        ) = _batch_update_memory(
            X,
            y,
            self.feature_memories[agents_to_update],
            self.target_memories[agents_to_update],
            mem_masks[agents_to_update],
            agent_mask[agents_to_update],
        )

        return agents_to_update

    def update(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        agent_mask: torch.BoolTensor,
        timesteps: torch.BoolTensor,
    ):

        # nb_agents_to_update = agent_mask.size(0)
        # update memory
        # X = X.unsqueeze(1).repeat(
        #     1, nb_agents_to_update, 1, 1
        # )  # (batch_size, n_agents, seq_len, input_dim)

        # print(X.shape)

        # X_masked = torch.where(
        #     timesteps.unsqueeze(0).unsqueeze(-1), X, torch.full_like(X, torch.nan)
        # )

        # print(X_masked.shape)

        # print(X_masked.nanmean(dim=2))

        # X devient un tensor de taille (batch_size, n_agents, input_dim)

        y = y.unsqueeze(-1)
        agents_to_update = self._update_memories(X.std(dim=1), y, agent_mask)

        # extract memories
        X = self.feature_memories[agents_to_update]
        y = self.target_memories[agents_to_update]

        base_prediction = torch.zeros(
            (self.base_prediction[agents_to_update].size(0)), device=self.device
        )
        models = torch.zeros(
            (self.models[agents_to_update].size(0), X.size(2) + 1, self.n_class),
            device=self.device,
        )

        has_different_classes = ~((y == y[:, 0:1, :]).all(dim=1).reshape(X.size(0)))

        # when agents have only 1 class in memory
        base_prediction[~has_different_classes] = y[~has_different_classes][
            :, 0
        ].squeeze()

        # when agents have all classes in memory
        if has_different_classes.sum() > 0:

            X_train = X[has_different_classes]
            y_train = y[has_different_classes]

            # update agents
            updated_models = batch_fit_logistic_regression(
                X_train, y_train, self.n_class, device=self.device
            )

            models[has_different_classes] = updated_models
            base_prediction[has_different_classes] = 0

        self.base_prediction[agents_to_update] = base_prediction
        self.models[agents_to_update] = models

    def __call__(self, X, agents_mask=None):
        y_pred = torch.empty(
            (self.models[agents_mask].size(0), X.size(0), 1),
            dtype=torch.float,
            device=self.device,
        )
        mask = self.base_prediction[agents_mask] == 0

        # when agents have only 1 class in memory
        base_prediction_agents = self.base_prediction[agents_mask]
        y_pred[~mask] = (
            base_prediction_agents[~mask]
            .unsqueeze(1)
            .unsqueeze(2)
            .repeat(1, X.size(0), 1)
        )

        # when agents have only all classes in memory
        models_agents = self.models[agents_mask]
        if models_agents[mask].size(0) > 0:
            y_pred[mask] = (
                batch_predict_logistic_regression(X.std(dim=1), models_agents[mask])
                .unsqueeze(-1)
                .float()
            )

        return y_pred

    def clone(self):
        cloned_self = copy.copy(self)  # shallow copy
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(cloned_self, attr_name, attr_value.clone())
        return cloned_self
