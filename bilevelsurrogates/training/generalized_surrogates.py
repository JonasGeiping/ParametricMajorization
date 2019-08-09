"""
    Problem-agnostic training
"""
import torch
import numpy as np
import torch.nn.functional as F
import types

import bilevelsurrogates
from .default_configurations import default_setup

"""
    Interface Function
"""


def DiscriminativeLearning(energy, loss, training_samples, training_setup=default_setup('DiscriminativeLearning'),
                           algorithm='joint-primal'):
    """
    Input: An energy-based model so that the inference is given by argmin_x E(x,y,theta) for some data y
    This class trains the energy-based model by minimizing D_E(x^*, x(theta))
    """
    if algorithm == 'joint-dual':
        return _DiscriminativeLearningJointDual(energy, loss, training_samples, training_setup)
    else:
        raise ValueError()

"""
    Parent Class defined for different implementations, but not exposed
"""


class _DiscriminativeLearningBase(bilevelsurrogates.Optimization):
    """
    Input: An energy-based model so that the inference is given by argmin_x E(x,y,theta) for some data y
    """
    def __init__(self, energy, loss, training_samples, training_setup, additional_term=None, x_ref=None):
        super().__init__(energy, training_setup['min_iterations'], training_setup['max_iterations'],
                         training_setup['validation'], training_setup['verbose'], training_setup['tolerance'])
        self.loss = loss
        self.energy = energy
        self.energy.unfreeze_parameters()
        self.samples = training_samples
        self.additional_term = additional_term
        if x_ref is None:
            self.x_ref = [self.samples.x]
        else:
            if isinstance(x_ref, list):
                self.x_ref = x_ref
            else:
                self.x_ref = [x_ref]

        self.setup = training_setup

    def initialize(self):
        raise NotImplementedError()

    def set_algorithm(self, parameters, auxiliary_variables):
        """
            Returns a torch.optim object that trains the given parameters
        """
        if self.setup['algorithm'] == 'Adam':
            optimizer = torch.optim.Adam([{'params': parameters, 'lr' : self.setup['lr']},
                                          {'params': auxiliary_variables, 'lr': self.setup['lr']}])
        elif self.setup['algorithm'] == 'AMSgrad':
            optimizer = torch.optim.Adam([{'params': parameters, 'lr' : self.setup['lr'], 'amsgrad':True},
                                          {'params': auxiliary_variables, 'lr': self.setup['lr'], 'amsgrad':True}])
        elif self.setup['algorithm'] == 'Adabound':
            optimizer = bilevelsurrogates.optim.AdaBound([{'params': parameters, 'lr' : self.setup['lr']},
                                                         {'params': auxiliary_variables, 'lr': self.setup['lr']}])
        elif self.setup['algorithm'] == 'GD':
            optimizer = torch.optim.SGD([{'params': parameters, 'lr' : self.setup['lr'], 'momentum' :0.95},
                                         {'params': auxiliary_variables,
                                          'lr': self.setup['lr'] * auxiliary_variables[0].numel() / np.sqrt(8),
                                          'momentum' :0.95}])
        elif self.setup['algorithm'] == 'FISTA':
            optimizer = bilevelsurrogates.optim.FISTA(
                [{'params': parameters, 'lr' : self.setup['lr'], 'projection' : None},
                 {'params': auxiliary_variables, 'lr': self.setup['lr'] * auxiliary_variables[0].numel() / np.sqrt(8),
                  'projection' :None}])
        else:
            raise ValueError('Invalid setup algorithm choice')

        return optimizer

    def step(self):
        raise NotImplementedError()

    def inertial_energy(self):
        loss = 0
        for idx, reference in enumerate(self.x_ref):
            if idx > 0 and self.setup['inertia'] > 0:
                loss = loss + self.setup['inertia'] * (self.energy(self.x_ref[0], self.samples.y)
                                                       - self.energy(reference, self.samples.y))
        return loss

    def record_loss(self, loss):
        self.stats['loss'].append((loss * self.setup['L'] + self.setup['loss_offset']).item())

    def check_error(self):
        if self.stats['loss'][-1] > 100:
            if self.verbose:
                print(f'loss value {self.stats["loss"][-1]} too large, terminating ...')
            return torch.tensor(0.0)
        elif not np.isfinite(self.stats['loss'][-1]):
            if self.verbose:
                print(f'loss value {self.stats["loss"][-1]} is invalid, terminating ...')
            return torch.tensor(0.0)
        elif self.iteration > 1:
            return np.abs(self.stats['loss'][-2] - self.stats['loss'][-1]) / np.abs(self.stats['loss'][-2])
        else:
            return torch.tensor(1.0)

    def print_status(self, iteration, progress):
        if (iteration % self.setup['callback'] == 0) and self.verbose:
            metric_est, metric_n, metric_f = self.loss.metric_est(self.stats['loss'][-1])
            print(f'status: iteration {iteration:>{len(str(self.max_iterations))}},'
                  f' loss: {self.stats["loss"][-1]:.5f} ({metric_n} est: {metric_est:{metric_f}}),'
                  f' tol: {100*progress.item():.3f}%')

    def compute_statistics(self):
        x_parameters = self.energy.minimize(self.samples.y)
        metric, metric_n, _ = self.loss.metric(self.samples.x, x_parameters)
        self.stats[metric_n].append(metric)
        return metric, x_parameters

    def finalize(self):
        pass


class _DiscriminativeLearningJointDual(_DiscriminativeLearningBase):
    """
        Learning via D_E(x^*,x(theta)) = E(x^*) - E(x(theta))
        with joint optimization of the latent x and the parameters theta
    """
    def __init__(self, energy, loss, training_samples, training_setup, additional_term=None, x_ref=None):
        super().__init__(energy, loss, training_samples, training_setup, additional_term, x_ref)
        self.gradient_offset = 0

    def initialize(self):
        with torch.no_grad():
            self.p = self.energy.initialize_dual(self.x_ref[0])
            self.energy.prox_dual_step(self.p)
        self.w = self.energy.initialize_latent(self.x_ref[0], self.samples.y)
        # The second latent variable is fused into p

        self.p.requires_grad = True
        self.optimizer = self.set_algorithm(self.energy.parameters(), [self.p, *self.w])

    def step(self):
        # Gradient Step
        with torch.enable_grad():
            def closure():
                self.optimizer.zero_grad()

                # ### -- Main Loss
                loss = self.energy(self.x_ref[0], self.samples.y, self.w)
                loss = loss + self.energy.dual(self.p, self.samples.y, argument=self.gradient_offset)
                # ### ------------
                if self.gradient_offset is not 0 and self.setup['correction']:
                    loss = loss - 0.5 * (self.gradient_offset**2).mean() * self.energy.smoothness()
                if self.setup['inertia'] > 0:
                    loss = loss + self.inertial_energy()
                loss.backward()
                return loss
            loss = self.optimizer.step(closure)
        # Projection Step
        self.energy.prox_dual_step(self.p)

        # Record loss
        self.record_loss(loss)

    def finalize(self):
        self.x = self.energy.primal_from_dual(self.p, self.samples.y)
        self.energy.prox_primal_step(self.x)
