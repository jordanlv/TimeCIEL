import torch
import torch.nn as nn
import torch.optim as optim
import torch.func as func


def binary_cross_entropy(xx, y_true, weights, biases):

    logits = xx @ weights + biases  # Linear transformation
    y_pred = torch.argmax(torch.sigmoid(logits), dim=-1)

    return -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)).mean()


_batch_grad_binary_cross_entropy = func.vmap(
    func.grad(binary_cross_entropy, argnums=(2, 3)), in_dims=(0, 0, 0, 0)
)


def batch_fit_logistic_regression(
    X, Y, n_class, lr=0.1, epoch=1000, device="cpu", batchsize=5
):
    """Perform a batch of weighted linear regression

    Args:
        X (Tensor): (batch_size, memory, input_dim)
        y (Tensor): (batch_size, memory, output_dim)

    Returns:
        Tensor: (n_parameter_set, input_dim + 1, n_class)
    """
    b, m, n = X.shape

    Y = Y.squeeze(-1) - 1

    weights = torch.randn(b, n, n_class, device=device)
    biases = torch.randn(b, n_class, device=device)

    mask = torch.zeros(b, n_class, dtype=torch.bool)

    row_indices = torch.arange(b).unsqueeze(1).expand_as(Y)
    mask[row_indices, Y.long()] = True

    mask = mask.unsqueeze(1)

    weights = weights * mask
    biases[~mask.squeeze(1)] = 0

    X, Y = X.to(device), Y.to(device).squeeze(-1)

    for e in range(epoch):
        perm = torch.randperm(m, device=device)

        for i in range(0, m, batchsize):
            xx = X[:, perm[i : i + batchsize], :]
            yy = Y[:, perm[i : i + batchsize]]

            grads_w, grads_b = _batch_grad_binary_cross_entropy(
                *(xx, yy, weights, biases)
            )

            grads_w = grads_w * mask
            grads_b[~mask.squeeze(1)] = 0

            with torch.no_grad():
                weights -= lr * grads_w
                biases -= lr * grads_b

    weights = weights * mask
    biases[~mask.squeeze(1)] = 0

    return torch.cat((weights.detach(), biases.unsqueeze(1).detach()), dim=1)


def predict_logicstic_regression(X, parameters):
    """Perform a linear transformation

    Args:
        X (Tensor): (batch_size, input_dim)
        parameters (Tensor): (input_dim + 1, 1)

    Returns:
        Tensor: (batch_size, 1)
    """

    params = parameters.squeeze()
    w = params[:-1]
    b = params[-1]

    y = torch.sigmoid(X @ w + b)

    return torch.argmax(y, dim=-1) + 1


_batch_predict_logistic_regression = torch.vmap(
    predict_logicstic_regression, in_dims=(None, 0)
)


def batch_predict_logistic_regression(X, parameters):
    """Perform a linear transformation with a batch of parameters

    Args:
        X (Tensor): (n_parameter_set, batch_size, input_dim)
        parameters (Tensor): (n_parameter_set, input_dim + 1, 1)

    Returns:
        Tensor: (n_parameter_set, batch_size, 1)
    """
    return _batch_predict_logistic_regression(X, parameters)
