import torch
from torch.optim import Optimizer

class NormalizedSGD(Optimizer):
    def __init__(self, parameters, lr, weight_decay, epsilon=0.0001):
        super().__init__(parameters, dict(lr=lr, weight_decay=weight_decay, epsilon=epsilon))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('closure is unsupported')
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None:
                    gradient = parameter.grad
                    scale = gradient.square().mean().sqrt() + group['epsilon']
                    parameter.mul_(1.0 - group['lr'] * group['weight_decay'])
                    parameter.add_(gradient / scale, alpha=-group['lr'])


def build(parameters, lr, weight_decay, epsilon=0.0001):
    return NormalizedSGD(parameters, lr, weight_decay, epsilon)
