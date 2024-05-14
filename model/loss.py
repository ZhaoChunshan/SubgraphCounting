import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def log_mse_loss(output_log, target_log):
    return F.mse_loss(output_log, target_log)

