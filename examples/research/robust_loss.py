import torch

def compute(prediction, target, delta=1.0):
    error = prediction - target
    return (delta * delta * (torch.sqrt(1.0 + (error / delta).square()) - 1.0)).mean()
