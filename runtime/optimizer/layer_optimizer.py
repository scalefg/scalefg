import torch.optim


class LayerOptimizer(torch.optim.Optimizer):
    """
    Wrapper class that modify gradients before optimizer.step()

    Arguments:
        - optim_name: the name of optimizer, required to create the corresponding base_optimizer (torch.optim.{optim_name})
        - optimizer_args: the keyword arguments passed to base_optimizer.
    """
    def __init__(self, optimizer):
        self.optim = optimizer

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.optim)

    @property
    def param_groups(self):
        return self.optim.param_groups

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self):
        self.optim.load_state_dict()

    def zero_grad(self):
        self.optim.zero_grad()

    def apply_layer_wsize(self, layer_wsize):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Here grads are summed when not zero_grad
                p.grad.mul_(layer_wsize)

    def step(self, layer_wsize, *args, **kwargs):
        self.apply_layer_wsize(layer_wsize)
        return self.optim.step(*args, **kwargs)