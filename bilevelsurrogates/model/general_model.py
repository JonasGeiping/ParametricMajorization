"""
    Class to instantiate an energy minimization model
"""

import torch
import bilevelsurrogates
from copy import deepcopy


class Energy(torch.nn.Module):
    """
        Basic 'problem' / 'energy' / 'objective' / 'value' / 'term' / 'model' class
    """
    def __init__(self, x_dims, setup):
        super().__init__()
        self.setup = deepcopy(setup)
        self.x_dims = x_dims
        self.validate_setup()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        self.setup["data"] = dict(device=device, dtype=dtype)
        for key, data in self.setup.items():
            if isinstance(data, torch.Tensor):
                self.setup[key] = data.to(device=device, dtype=dtype)

    def prepare_training(self, x, y):
        pass

    def initialize_dual(self, x):
        raise NotImplementedError()

    def initialize_latent(self, x=0, y=0):
        return [torch.tensor(0.0)]

    def initialize_primal(self, x):
        x_init = x.clone()
        self.prox_primal_step(x_init)
        return x_init

    def forward(self, x, y):
        raise NotImplementedError()

    def dual(self, p, y, argument=0):
        raise NotImplementedError()

    def primal_from_dual(self, dual_vector, y):
        raise NotImplementedError()

    def minimize(self, y, x0, return_dual=False, lr=0.01):
        # Prepare inference
        self.freeze_parameters()
        aux_x = x0.clone().detach()
        self.prox_primal_step(aux_x)

        with torch.enable_grad():
            aux_x.requires_grad = True
            aux_optimizer = torch.optim.Adam([{'params': aux_x, 'lr' : lr}])
            # Iterate
            for iteration in range(self.setup['inference']['max_iterations']):
                aux_optimizer.zero_grad()
                loss = self.forward(aux_x, y)
                loss.backward()
                aux_optimizer.step()

            # Finalize
            aux_optimizer.zero_grad()
            aux_x.requires_grad = False
            self.unfreeze_parameters()

        if return_dual:
            return aux_x, 0
        else:
            return aux_x

    def prox_primal_step(self, x, w1=None, w2=None, tau=1.0):
        pass

    def prox_dual_step(self, dual_vector, sigma=1.0):
        pass

    def correction(self):
        return 0

    def initialize(self):
        x_init = torch.zeros_like(self.x_dims, **self.setup['data'])
        return x_init

    def clone_parameters(self):
        theta_tmp = []
        for param in self.parameters():
            theta_tmp.append(param.data.clone().detach())
        return theta_tmp

    def set_parameters(self, theta_tmp):
        for idx, param in enumerate(self.parameters()):
            param.data = theta_tmp[idx].clone().detach()

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = True

    def validate_setup(self):
        pass
