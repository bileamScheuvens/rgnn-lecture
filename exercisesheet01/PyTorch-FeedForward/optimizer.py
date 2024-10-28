import torch
from torch.optim.optimizer import Optimizer

class Adam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, debug=False):

        for group in self.param_groups:

            b1, b2 = group['betas']
            lr  = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is not None:

                    g = p.grad

                    state = self.state[p]

                    # Lazy state initialization
                    if len(state) == 0:
                        state['step']  = 0
                        state['m']     = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['v']     = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['step'] += 1

                    state['m'] = b1 * state['m'] + (1-b1)*g
                    state['v'] = b2 * state['v'] + (1-b2)*g**2

                    m_hat = state['m'] / (1-b1**state['step'])
                    v_hat = state['v'] / (1-b2**state['step'])

                    p -= lr * (m_hat / (torch.sqrt(v_hat) + eps))


