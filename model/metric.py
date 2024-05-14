import torch
import numpy as np

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def log_q_error_median(output_log, target_log):
    return np.quantile((output_log - target_log).detach().cpu().numpy(), 0.5)


def log_q_error_95(output_log, target_log):
    return np.quantile((output_log - target_log).detach().cpu().numpy(), 0.95)


def log_q_error_75(output_log, target_log):
    return np.quantile((output_log - target_log).detach().cpu().numpy(), 0.75)


def log_q_error_25(output_log, target_log):
    return np.quantile((output_log - target_log).detach().cpu().numpy(), 0.25)


def log_q_error_5(output_log, target_log):
    return np.quantile((output_log - target_log).detach().cpu().numpy(), 0.05)


def log_q_error_max(output_log, target_log):
    return np.max((output_log - target_log).detach().cpu().numpy())


def log_q_error_min(output_log, target_log):
    return np.min((output_log - target_log).detach().cpu().numpy())


def log_q_error_range(output_log, target_log):
    return log_q_error_95(output_log, target_log) - log_q_error_5(output_log, target_log)
