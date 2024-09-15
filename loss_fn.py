import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

## https://github.com/gabriben/metrics-as-losses/blob/main/VLAP/pytorchLosses.py
## https://github.com/ashrefm/multi-label-soft-f1/blob/master/Multi-Label%20Image%20Classification%20in%20TensorFlow%202.0.ipynb


def macro_soft_f1(y, y_hat):
    """
    Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.

    Args:
        y (Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = y.float()
    y_hat = F.sigmoid(y_hat.float())

    tp = torch.sum(y_hat * y, dim=0)
    fp = torch.sum(y_hat * (1 - y), dim=0)
    fn = torch.sum((1 - y_hat) * y, dim=0)

    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)

    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = torch.mean(cost)  # average on all labels

    return macro_cost


def macro_double_soft_f1(y, y_hat):
    """
    Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y (Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = y.float()
    y_hat = F.sigmoid(y_hat.float())

    tp = torch.sum(y_hat * y, dim=0)
    fp = torch.sum(y_hat * (1 - y), dim=0)
    fn = torch.sum((1 - y_hat) * y, dim=0)
    tn = torch.sum((1 - y_hat) * (1 - y), dim=0)

    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)

    cost_class1 = 1 - soft_f1_class1  # minimize (1 - soft-f1_class1) to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # minimize (1 - soft-f1_class0) to increase soft-f1 on class 0

    cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
    macro_cost = torch.mean(cost)  # average on all labels

    return macro_cost


def compute_loss(loss_name: str, y_preds, y_true, weight=None):
    loss_name = loss_name.lower()
    if loss_name == "cross_entropy":
        if weight is not None:
            weight = torch.tensor(weight).float().to(y_preds.device)
        return F.cross_entropy(y_preds, y_true, weight=weight).mean()
    if loss_name == "mse":
        return F.mse_loss(y_preds.reshape(-1), y_true.reshape(-1).float())
    if loss_name == "mae":
        return F.l1_loss(y_preds.reshape(-1), y_true.reshape(-1).float())
    ## convert y_true to 1-hot encoding
    y_true = F.one_hot(y_true, num_classes=y_preds.shape[1]).float()
    if loss_name == "macro_soft_f1":
        return macro_soft_f1(y_hat=y_preds, y=y_true)
    if loss_name == "macro_double_soft_f1":
        return macro_double_soft_f1(y_hat=y_preds, y=y_true)

    return None
