import torch
from torch.optim import Optimizer

class FOCUS(Optimizer):
    """
    FOCUS: First-Order Concentrated Update Scheme

    This optimizer implements an approach to stochastic gradient descent that aims to
    concentrate parameter updates and reduce the impact of gradient stochasticity.

    Key features:
    - Uses sign-based updates
    - Incorporates a running average of parameters
    - Aims to reduce the impact of gradient stochasticity
    """

    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), gamma=0.1, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        
        defaults = dict(lr=lr, betas=betas, gamma=gamma, weight_decay=weight_decay)
        super(FOCUS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            gamma = group['gamma']
            lr = group['lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['pbar'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1

                exp_avg, pbar = state['exp_avg'], state['pbar']

                # Update running averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                pbar.mul_(beta2).add_(p, alpha=1 - beta2)

                # Compute bias-corrected pbar
                pbar_hat = pbar / (1 - beta2 ** state['step'])

                # Compute update
                update = torch.sign(exp_avg) + gamma * torch.sign(p - pbar_hat)

                # Update parameters
                p.add_(update, alpha=-lr)

                # Apply weight decay
                if weight_decay != 0:
                    p.add_(pbar_hat, alpha=-lr * weight_decay)
