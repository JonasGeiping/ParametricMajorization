"""
Implementations of convex denoising strategies
"""
import numpy as np
import torch
import time

import bilevelsurrogates

import torch.nn.functional as F


class SparseAnalysis(bilevelsurrogates.ConvexOptimization):
    """
        Primal-Dual Implementation of the sparse analysis problem
        min_x  1/2*||x-f||^2 + alpha*||Dx||_H_epsilon
    """
    def __init__(self, data, operator, alpha=1.0, gamma=0.1, epsilon=0.0, norm='aniso',
                 max_iterations=2500, min_iterations=100, tolerance=1e-4, clip=True, p0=0, x0=0):
        super().__init__(operator, data, alpha, min_iterations, max_iterations, tolerance)
        self.gamma = gamma
        self.epsilon = epsilon
        self.norm = norm
        self.clip = clip

        self.p0 = p0
        self.x0 = x0

    def initialize(self):

        K = self.operator.normest()
        self.tau = 1 / K
        self.sigma = 1 / K

        if self.x0 is 0:
            self.x = torch.clone(self.data)
            self.x_ = torch.clone(self.data)
        else:
            self.x = torch.as_tensor(self.x0, device=self.device, dtype=self.dtype)
            self.x_ = torch.clone(self.x)
        if self.p0 is 0:
            self.p = self.data.new_zeros(self.B, self.C, self.m, self.n, device=self.device, dtype=self.dtype)
        else:
            self.p = torch.as_tensor(self.p0, device=self.device, dtype=self.dtype)

    def step(self):
        # Update overrelaxation size
        self.theta = 1 / torch.sqrt(1 + 2 * self.gamma * self.tau)

        # Update dual variables
        self.p += self.operator(self.x_) * self.sigma
        self.p = self.norm_project(self.p)

        # Overrelaxation I:
        self.x_ = -self.x * self.theta
        # Update primal variable
        self.x -= self.operator(self.p, 't') * self.tau
        self.x = (self.x + self.data * self.tau) / (1 + self.tau)
        if self.clip:
            self.x.data.clamp_(0, 1)
        #
        # Overrelaxation II:
        self.x_ += self.x * (1 + self.theta)

        # Update step sizes
        self.tau *= self.theta
        self.sigma /= self.theta

    def check_error(self):
        return torch.norm(self.x - self.x_) / self.theta

    def finalize(self):
        if self.clip:
            self.x.data.clamp_(0, 1)

    def norm_project(self, p):
        """
        efficiently project onto p in ||.||_infty < alpha
        """
        if self.alpha > 0:
            # 1) Huber correction
            p *= self.alpha / (self.alpha + self.sigma * self.epsilon)

            # 2) local norm
            if self.norm == 'aniso':
                pnorm = p.abs()
            elif self.norm == 'iso2':
                pnorm = torch.sqrt(p[:, 0:self.C:2, :, :]**2 + p[:, 1:self.C:2, :, :]**2)  # iso
                pnorm = torch.stack((pnorm, pnorm), dim=2).view(self.B, self.C, self.m, self.n)
            elif self.norm == 'iso6':
                pnorm = torch.sqrt(p[:, 0:self.C:6, :, :]**2 + p[:, 1:self.C:6, :, :]**2
                                   + p[:, 2:self.C:6, :, :]**2 + p[:, 3:self.C:6, :, :]**2
                                   + p[:, 4:self.C:6, :, :]**2 + p[:, 5:self.C:6, :, :]**2)  # iso6
                pnorm = torch.stack([pnorm] * 6, dim=2).view(self.B, self.C, self.m, self.n)
            elif self.norm == 'infiso':
                pnorm = p[:, 0:self.C:2, :, :].abs() + p[:, 1:self.C:2, :, :].abs()  # inf aniso
                pnorm = torch.stack((pnorm, pnorm), dim=2).view(self.B, self.C, self.m, self.n)
            else:
                raise ValueError()

            # 3) Projection
            return self.alpha * p / torch.max(self.alpha, pnorm)
        else:  # protect against NaNs in singular case alpha=0
            return p.zero_()


class BregmanSparseAnalysis(SparseAnalysis):
    """
    replaces non-smooth projections by smooth operations in a Bregman-primal dual framework
    """
    def __init__(self, data, operator, alpha=1.0, gamma=0.0, epsilon=0.0, norm='aniso',
                 max_iterations=2500, min_iterations=100, tolerance=1e-4, clip=False, requires_grad=True):
        super().__init__(data, operator, alpha, gamma, epsilon, norm, max_iterations, min_iterations, tolerance, clip)
        self.requires_grad = requires_grad

        if self.norm is not 'aniso':
            raise ValueError('Only anisotropic norms implemented in diff. setting.')

    def initialize(self):
        # with torch.autograd.set_grad_enabled(self.requires_grad):
        with torch.no_grad():
            K = self.operator.detach().normest()
        self.tau = 1 / K
        self.sigma = 1 / K
        self.theta = 1

        self.x = torch.clone(self.data)
        self.x_ = torch.clone(self.data)
        self.p = self.data.new_zeros(self.B, self.C, self.m, self.n, device=self.device, dtype=self.dtype)

    def step(self):
        with torch.autograd.set_grad_enabled(self.requires_grad):

            if self.gamma > 0:
                self.theta = 1 / torch.sqrt(1 + 2 * self.gamma * self.tau)

            # Update dual variables
            if self.alpha > 0:
                if self.epsilon > 0:
                    huber = self.epsilon / self.alpha * self.p  # This is (only) a gradient step as in Chambolle'16
                else:
                    huber = 0.0
                self.p = self.shannon_prox(self.p, -self.operator(self.x_) + huber, self.sigma, self.alpha)
            else:
                self.p = self.p.zero_()

            # Overrelaxation I:
            self.x_ = -self.x * self.theta
            # Update primal variable
            if self.clip:   # Shannon primal
                grad = (self.x - self.data + self.operator(self.p, 't')) * self.tau
                self.x = self.x * torch.exp(-grad)
            else:   # Euclidean primal
                self.x = self.x - self.operator(self.p, 't') * self.tau
                self.x = (self.x + self.data * self.tau) / (1 + self.tau)

            # Overrelaxation II:
            self.x_ = self.x_ + self.x * (1 + self.theta)

            # Update step sizes
            if self.gamma > 0:
                self.tau = self.tau * self.theta
                self.sigma = self.sigma * self.theta

    def check_error(self):
        return torch.norm(self.x.detach() - self.x_.detach()) / self.theta

    def shannon_prox(self, p, linear_term, step_size, alpha):
        """
        as discussed in Ochs et al. 2016, using h(x) = log(alpha+x)*(alpha+x)+log(alpha-x)*(alpha-x)
        """
        exp_term = torch.exp(- linear_term * step_size)
        constraint_term = (alpha - p) / (alpha + p)

        return alpha * (exp_term - constraint_term) / (exp_term + constraint_term)

    def finalize(self):
        pass
