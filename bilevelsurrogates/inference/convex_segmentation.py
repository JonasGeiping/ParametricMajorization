"""
  Primal-Dual Algorithms for convex segmentation
"""
import numpy as np
import torch
import time

import bilevelsurrogates

import torch.nn.functional as F


class ConvexSegmentation(bilevelsurrogates.Optimization):
    """
        Minimize models of the form -<n(theta, y), x > + h(x) + ||Dx||_1 with a primal dual algorithm
    """
    def __init__(self, energy, data, relaxation='loose', legendre='entropy', entropy=1.0,
                 max_iterations=2500, min_iterations=100, tolerance=1e-4, p0=0, x0=0):
        super().__init__(energy, min_iterations, max_iterations, validation=-1, verbose=True, tolerance=tolerance)

        self.energy = energy
        self.data = torch.as_tensor(data, **self.energy.setup['data'])

        self.relaxation = relaxation
        self.legendre = legendre
        self.entropy = entropy
        if self.legendre == 'euclidean' and self.entropy > 0:
            raise NotImplementedError()

        self.p0 = p0
        self.x0 = x0

    def initialize(self):
        K = self.energy.grad.normest()
        self.tau = 1 / K
        self.sigma = 1 / K
        self.theta = 1

        [self.B, self.c, self.m, self.n] = self.data.shape
        self.C = self.energy.grad.out_channels

        self.potential = self.energy.operator(self.data)
        self.num_labels = self.potential.shape[1]

        if self.x0 == 0:
            self.x = F.softmax(self.potential, dim=1)
            self.x_ = torch.clone(self.x)
        else:
            self.x = torch.as_tensor(self.x0, **self.energy.setup['data'])
        if self.p0 == 0:
            self.p = self.data.new_zeros(self.B, self.C, self.m, self.n, **self.energy.setup['data'])
        else:
            self.p = torch.as_tensor(self.p0, **self.energy.setup['data'])

        if self.legendre == 'euclidean':
            self.s = torch.zeros(self.B, 1, self.m, self.n, **self.energy.setup['data'])

        if self.relaxation == 'tight_lagrangian':
            self.constraints = []
            for i2 in range(self.num_labels):
                for i1 in range(i2):
                    self.constraints.append([i1, i2])
            self.slack = self.p.new_zeros(self.B, 2 * len(self.constraints), self.m, self.m)
            self.slack_ = self.p.new_zeros(self.B, 2 * len(self.constraints), self.m, self.n)
            self.pi1i2 = self.p.new_zeros(self.B, 2 * len(self.constraints), self.m, self.n)
            # The norm of the new pairwise operator is sqrt(C), hence
            K = torch.max(torch.tensor(np.sqrt(self.num_labels), **self.energy.setup['data']), K)
            self.tau = 1 / K
            self.sigma = 1 / K

    def step(self):
        # Update dual variables
        self.p += self.sigma * self.energy.grad(self.x_)

        if self.relaxation == 'tight_dykstra' and self.iteration > 100:
            self.energy.prox_tight_dual_step(self.p)
        elif self.relaxation == 'tight_lagrangian':
            self.pi1i2 += self.sigma * self.slack_
            self.energy.prox_dual_step(self.pi1i2)
            for idx, [i1, i2] in enumerate(self.constraints):
                # Dual update
                update_dual = self.sigma * self.slack_[:, 2 * idx:2 * idx + 2, :, :]
                self.p[:, 2 * i2:2 * i2 + 2, :, :] += update_dual
                self.p[:, 2 * i1:2 * i1 + 2, :, :] -= update_dual
            # Overrelaxation I
            self.slack_ = -self.theta * self.slack
            for idx, [i1, i2] in enumerate(self.constraints):
                # Primal Update
                update_primal = (self.pi1i2[:, 2 * idx:2 * idx + 2, :, :]
                                 + self.p[:, 2 * i2:2 * i2 + 2, :, :]
                                 - self.p[:, 2 * i1:2 * i1 + 2, :, :])
                self.slack[:, 2 * idx:2 * idx + 2, :, :] -= self.tau * update_primal
            # Overrelaxation II
            self.slack_ += (1 + self.theta) * self.slack
        else:
            self.energy.prox_dual_step(self.p)

        if self.legendre == 'euclidean':
            self.s += self.sigma * (torch.sum(self.x_, 1, keepdim=True) - 1.0)

        # Overrelax I
        self.x_ = -self.theta * self.x
        if self.legendre == 'euclidean':
            # Update primal variable
            self.x -= self.tau * (-self.potential + self.s + self.energy.grad(self.p, 't'))
            self.x.data.clamp_(0, 1)
        elif self.legendre == 'entropy':
            # Update primal variable with entropy Bregman prox
            # grad_term = -self.tau * (-self.potential + self.energy.grad(self.p, 't')) / (1 + self.tau)
            # self.x = self.x**(1 / (1 + self.tau)) * torch.exp(grad_term)
            grad_term = -self.tau * (-self.potential + self.energy.grad(self.p, 't')) / (1 + self.entropy * self.tau)
            self.x = self.x**(1 / (1 + self.entropy * self.tau)) * torch.exp(grad_term)
            self.x /= torch.sum(self.x, dim=1, keepdim=True)

        # Overrelax II
        self.x_ += (1 + self.theta) * self.x

    def check_error(self):
        """
            Return 'binarity' of solution as a stopping criterion
        """
        return F.mse_loss((self.x > 0.5).to(self.energy.setup['data']['dtype']), self.x)

    def finalize(self):
        pass
