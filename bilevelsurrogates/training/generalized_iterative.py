"""
  Generalized Iterative Methods
"""
import torch
import numpy as np
import torch.nn.functional as F
import types
import warnings

import bilevelsurrogates


class IterativeLearning(bilevelsurrogates.Optimization):
    """
        Iterative Discrimination via sucessive parametric linearization and overapproximation
    """
    def __init__(self, discriminative_subroutine, iterative_setup):
        super().__init__(discriminative_subroutine.energy, 1, iterative_setup['linearizer_iterations'],
                         -1, iterative_setup['verbose'], iterative_setup['tolerance'])
        self.subroutine = discriminative_subroutine
        self.setup = iterative_setup

        # Rebind samples, energy and higher-level loss
        self.energy = self.subroutine.energy
        self.samples = self.subroutine.samples
        self.loss = self.subroutine.loss

        self.stats['stat'].append(0)

    def initialize(self):
        self.exit = False
        self.discard = False
        # (x, p) is the current primal-dual solution to the lower-level problem
        self.x = 0
        self.p = 0
        # (x_e, p_e) is the current primal-dual extrapolation point
        self.x_e = self.samples.x.clone()
        self.p_e = 0
        # (x_i) is (another) primal refernence point, usually x_e of the previous iteration
        if self.subroutine.setup['inertia'] > 0:
            self.x_i = self.samples.x.clone()

        if self.setup['stochastic']:
            self.samples.reset()
        if self.setup['restart'] == 2:
            self.theta_init = self.energy.clone_parameters()

    def step(self):

        if self.setup['backtracking'] or (self.setup['stochastic'] and self.subroutine.setup['inertia'] > 0):
            theta_temp = self.energy.clone_parameters()

        # Prepare next iteration of discriminator w.r.t to loss gradient
        self.subroutine.x_ref = [self.x_e]
        loss, _, _ = self.loss(self.samples.x, self.x_e)

        self.subroutine.additional_term = self.additional_term
        self.set_constants(loss)

        if self.subroutine.setup['inertia'] > 0:
            self.subroutine.x_ref.append(self.x_i)

        # Re-run discriminator
        if self.setup['restart'] == 1 or (self.iteration == 0 and self.setup['restart'] == 0):
            self.subroutine.run()
        elif self.setup['restart'] == 0 and self.iteration > 0:
            self.subroutine.rerun()
        elif self.setup['restart'] == 2:
            self.energy.set_parameters(self.theta_init)
            self.subroutine.run()
        else:
            raise ValueError()
        self.stats['loss_sub'].append(self.subroutine.stats['loss'][-1])

        # Check higher-level loss
        if self.setup['check_loss'] or self.setup['backtracking'] or self.setup['extrapolation_choice'] == 'theta-k':
            self.x, self.p = self.energy.minimize(self.samples.y, x0=self.x, p0=self.p, return_dual=True)
            loss, _, _ = self.loss(self.samples.x, self.x)
            loss = loss.item()
        else:
            # As a substitute, use the subroutine loss
            loss = self.stats['loss_sub'][-1]

        # Either do backtracking, or save the best result
        if self.setup['backtracking']:
            if self.iteration > 0:
                if loss > self.stats['loss'][-1]:
                    self.backtracks += 1
                    self.energy.set_parameters(theta_temp)
                    if not np.isfinite(loss):
                        self.x, self.p = 0, 0
                    self.subroutine.setup['lr'] /= 1.5
                    self.subroutine.setup['max_iterations'] += 1_000
                    self.discard = True
                    if self.setup['backtracking'] <= self.backtracks:
                        self.exit = True
                        if self.verbose:
                            print(f'Backtracking unsuccessful for {self.setup["backtracking"]} times '
                                  f', terminating ...')
                    elif self.verbose:
                        print(f'Loss value {loss:.5f} higher than previous iteration ({self.stats["loss"][-1]:.5f}), '
                              f'reducing step size and increasing iterations.')
                    if self.subroutine.setup['lr'] < 1e-8:
                        self.exit = True
                        if self.verbose:
                            print(f'Backtracking reduced step size to {self.subroutine.setup["lr"]}, terminating ...')
                    return
                else:
                    self.discard = False
                    self.backtracks = 0
            else:
                self.backtracks = 0
        else:
            if self.iteration == 0:
                self.opt_theta = self.energy.clone_parameters()
            elif self.setup['output'] == 'best-loss':
                if loss < min(self.stats['loss']):
                    self.opt_theta = self.energy.clone_parameters()
            elif self.setup['output'] == 'best-metric':
                metric, metric_n, _ = self.loss.metric(self.samples.x, self.x)
                if metric > max(self.stats[metric_n]):
                    self.opt_theta = self.energy.clone_parameters()
            else:
                raise NotImplementedError()

        self.stats['loss'].append(loss)

        if self.subroutine.setup['inertia'] > 0:
            self.x_i = self.x_e.clone().detach()

        # Stastistics
        self.compute_statistics()

        # Stochastic Setup
        if self.setup['stochastic']:
            self.samples.step()
            self.x, self.p = self.samples.x, 0
            # grad = self.loss.gradient(self.samples.x, self.x_e)
            # self.x_e, self.p_e = 0, 0
            if self.subroutine.setup['inertia'] > 0:  # ughh ...
                theta_new = self.energy.clone_parameters()
                self.energy.set_parameters(theta_temp)
                self.x_i, _ = self.extrapolation()
                self.energy.set_parameters(theta_new)

        # Run the extrapolation step
        self.x_e, self.p_e = self.extrapolation()

    def additional_term(self, x):
        grad = self.loss.gradient(self.samples.x, self.x_e)
        linearization = (grad * x).mean() / self.subroutine.setup['L']
        if self.subroutine.setup['correction']:
            correction = 0.5 * ((grad / self.subroutine.setup['L'])**2).mean() * self.energy.smoothness()
        else:
            correction = 0.0

        return linearization - correction

    def set_constants(self, loss_val):
        if not torch.isfinite(loss_val):
            warnings.warn('Loss constant calculation failed. Resetting loss constant.')
            loss_val = 0

        grad = self.loss.gradient(self.samples.x, self.x_e)
        self.subroutine.setup['loss_offset'] = loss_val - (grad * self.x_e).mean()
        self.subroutine.gradient_offset = grad / self.subroutine.setup['L']

    def extrapolation(self):
        # Compute extrapolation step
        if self.setup['extrapolation_choice'] == 'theta-k':
            if self.setup['stochastic']:
                x_val, p_val = self.energy.minimize(self.samples.y, x0=self.samples.x, p0=0, return_dual=True)
            else:
                x_val, p_val = self.x, self.p
        elif self.setup['extrapolation_choice'] == 'trivial':
            x_val, p_val = self.samples.x, 0
        else:
            raise ValueError()

        return x_val, p_val

    def check_error(self):
        if self.exit:
            tol = torch.tensor(0.0)
        elif len(self.stats['loss']) > 1:
            tol = np.abs(self.stats['loss'][-2] - self.stats['loss'][-1]) / self.stats['loss'][-2]
        else:
            tol = torch.tensor(1.0)
        self.stats['tolerance'].append(tol)
        return tol

    def print_status(self, iteration, progress):
        if self.verbose:
            msg = f'-- Iteration {iteration} done, tol: {100*progress.item():.3f}%,'
            metric_n, metric_f = self.loss.metric()
            loss_n, loss_f = self.loss()
            if self.setup['check_loss']:
                msg += f' {loss_n}: {self.stats["loss"][-1]:{loss_f}},'
                msg += f' {metric_n}: {self.stats[metric_n][-1]:{metric_f}}'
            print(msg)

    def compute_statistics(self):
        """
            Stats are mostly only computed if x(theta^k) is actually computed
        """
        if self.setup['check_loss'] and not self.discard:
            metric, metric_n, _ = self.loss.metric(self.samples.x, self.x_e)
            self.stats['aug_' + metric_n].append(metric)
            metric, metric_n, _ = self.loss.metric(self.samples.x, self.x)
            self.stats[metric_n].append(metric)
            self.stats['norm'].append(self.energy.operator.weight.norm().item())  # this is kind of iffy

    def finalize(self):
        if not self.setup['backtracking']:
            self.energy.set_parameters(self.opt_theta)
