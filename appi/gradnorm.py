"""Two-task GradNorm with detached model gradients and positive normalized weights."""
import torch
from torch import nn


class GradNorm(nn.Module):
    def __init__(self, alpha=1.5, floor=1e-3):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))
        self.register_buffer('initial_losses', torch.ones(2))
        self.register_buffer('initialized', torch.tensor(False))
        self.alpha, self.floor = alpha, floor

    def weights(self):
        return 2 * (self.logits.softmax(0) + self.floor) / (1 + 2 * self.floor)

    def objective(self, losses, raw_norms):
        losses, raw_norms = losses.detach(), raw_norms.detach()
        if not self.initialized:
            self.initial_losses.copy_(losses.clamp_min(1e-8))
            self.initialized.fill_(True)
        relative = losses.clamp_min(1e-8) / self.initial_losses
        rates = relative / relative.mean()
        weighted_norms = self.weights() * raw_norms
        target = (weighted_norms.mean() * rates.pow(self.alpha)).detach()
        return (weighted_norms - target).abs().sum()


def gradient_snapshot(parameters):
    return [None if p.grad is None else p.grad.detach().clone() for p in parameters]


def combine_gradients(parameters, task_gradients, weights):
    """No second-order graph and no optimizer update between the two tasks."""
    for i, p in enumerate(parameters):
        active = [w * g[i] for w, g in zip(weights, task_gradients) if g[i] is not None]
        p.grad = sum(active) if active else None


def gradient_diagnostics(parameters, shared, gradients):
    indices = [i for i, p in enumerate(parameters) if any(p is q for q in shared)]
    vectors = [torch.cat([(g[i] if g[i] is not None else torch.zeros_like(parameters[i])).flatten()
                          for i in indices]) for g in gradients]
    norms = torch.stack([v.norm() for v in vectors])
    cosine = (vectors[0] @ vectors[1]) / (norms.prod().clamp_min(1e-20))
    return norms, float(cosine)
