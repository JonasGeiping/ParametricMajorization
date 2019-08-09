"""
Custom pytorch optimizers
"""
import torch
import torch.optim
import numpy as np
import math

from torch.optim.optimizer import required


class FISTA(torch.optim.Optimizer):
    """
    Implement the FISTA algorithm, or FISTA-MOD
    borrows heavily from pytorch ADAM implementation!
    """

    def __init__(self, params, projection=None, lr=1e-4,
                 fista_mod=(1.0, 1.0, 4.0)):
        """
        This requires that projetion is a function handle to be applied to
        the parameter
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if projection is None:
            self.projection = None
        else:
            self.projection = projection

        defaults = dict(lr=lr, fista_mod=fista_mod, projection=projection)
        super(FISTA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FISTA, self).__setstate__(state)

    def step(self, closure=None):
        """
        Single optimization step
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data

                state = self.state[param]

                # Initialization of State:
                if len(state) == 0:
                    state['x+'] = param.clone().detach()
                    state['x-'] = param.clone().detach()
                    state['tk'] = param.new_ones(1, requires_grad=False)

                # Gradient step
                state['x+'] = param.data - grad * group['lr']
                if group['projection'] is not None:
                    state['x+'] = group['projection'](state['x+'])

                # Overrelaxation factor
                p_factor, q_factor, r_factor = group['fista_mod']
                tk = (p_factor
                      + torch.sqrt(q_factor + r_factor * state['tk']**2)) / 2
                ak = (state['tk'] - 1) / tk
                state['tk'] = tk

                # The actual parameter corresponds to 'yk'
                param.data = state['x+'] * (1 + ak) - state['x-'] * ak
                state['x-'].data = state['x+'].clone()

        return loss


class FISTALineSearch(torch.optim.Optimizer):
    """
    Implement the FISTA algorithm, or FISTA-MOD
    borrows heavily from pytorch ADAM implementation!
    with added linesearch
    """

    def __init__(self, params, projection=None, lr=10, eta=0.8,
                 max_searches=25, fista_mod=(1.0, 1.0, 4.0), tk=1.0):
        """
        This requires that projection is a function handle to be applied to the
        parameter
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if projection is None:
            self.projection = lambda x: x
        else:
            self.projection = projection

        defaults = dict(lr=lr, eta=eta, max_searches=max_searches,
                        fista_mod=fista_mod, projection=projection, tk=tk)
        super(FISTALineSearch, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FISTALineSearch, self).__setstate__(state)

    def step(self, closure=None):
        """
        Single optimization step
        but backtracking is done for every group, many groups will thus lead to
        a long computation times.
        """

        if closure is None:
            raise ValueError('A closure is necessary, containing the keyword '
                             ' "requires_grad", which determines whether '
                             'loss.backward is called')
        else:
            loss = closure(requires_grad=True)

        for group in self.param_groups:

            # Phase 0: Update overrelaxation factor
            p_factor, q_factor, r_factor = group['fista_mod']
            tk = (p_factor + np.sqrt(q_factor + r_factor * group['tk']**2)) / 2
            ak = (group['tk'] - 1) / tk
            group['tk'] = tk

            # Phase I Linesearch for largest possible lr
            loss_yk = closure(requires_grad=False)

            for searches in range(group['max_searches']):
                linearization = group['params'][0].new_zeros(1)
                distance = group['params'][0].new_zeros(1)
                for param in group['params']:
                    if param.grad is None:
                        continue
                    grad = param.grad.data
                    state = self.state[param]

                    # Initialization of State:
                    if len(state) == 0:
                        state['x-'] = param.clone().detach()

                    # Gradient step
                    state['yk'] = param.data.clone().detach()
                    param.data -= grad * group['lr']
                    if self.projection is not None:
                        param.data = self.projection(param.data)
                    linearization += torch.sum(grad * (param.data - state['yk']))
                    distance += torch.sum((param.data - state['yk'])**2) / 2

                loss_xk = closure(requires_grad=False)
                D_h_xk_yk = loss_xk - loss_yk - linearization

                # Check lineseach condition:
                if D_h_xk_yk * group['lr'] > distance:
                    # Reduce lr:
                    group['lr'] *= group['eta']
                    # Undo gradient step
                    # If we had no projection this would be easier done via
                    # adding the gradient back on
                    for param in group['params']:
                        if param.grad is None:
                            continue
                        param.data = self.state[param]['yk'].clone()

                else:
                    # Break loop and continue to next phase
                    break

            # Phase II - Overrelaxation Step
            for param in group['params']:
                if param.grad is None:
                    continue
                # Get param state
                state = self.state[param]

                # The value of param is currently x^{k+1}, due to the step
                param_xp = param.data.clone()

                # The actual parameter corresponds to 'yk'
                param.data = param.data * (1 + ak) - state['x-'] * ak
                state['x-'].data = param_xp

        return loss
