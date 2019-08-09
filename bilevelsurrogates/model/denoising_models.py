"""
    Denoising Models
"""
import torch
import bilevelsurrogates

from .general_model import Energy

# ## Define defaults:
default_setup = dict()
default_setup['data'] = dict(dtype=torch.float, device=torch.device('cpu'))
# Sizes:
default_setup['x_dims'] = [200, 1, 64, 64]
# Hyperparameters:
default_setup['alpha'] = 0.01
default_setup['epsilon'] = 0.0
default_setup['bias'] = False
default_setup['norm'] = 'aniso'
default_setup['clip'] = True
default_setup['inference'] = dict(tolerance=1e-4, max_iterations=250)


class AnalysisSparsity(Energy):
    """
        The Analysis Sparsity Learning model E(x,y,theta) = 0.5||x-y||^2 + alpha ||D(theta)x||_epsilon
    """
    def __init__(self, operator, setup=default_setup):
        super().__init__(setup['x_dims'], setup)
        self.operator = operator

        self.setup['alpha'] = torch.as_tensor(self.setup['alpha'], **self.setup['data'])
        self.setup['epsilon'] = torch.as_tensor(self.setup['epsilon'], **self.setup['data'])

    def initialize_dual(self, x):
        return self.operator(x)

    def forward(self, x, y, w=0):
        return self.E_1(x, y) + self.E_2(x, y)

    def dual(self, p, y, w=None, argument=0):
        return self.E_1_dual(p, y, argument) + self.E_2_dual(p, y, argument=0)

    def E_1(self, x, y):
        return 0.5 * ((x - y)**2).mean()

    def E_2(self, x, y):
        return self.setup['alpha'] * self.huber(self.operator(x)).sum() / y.numel()

    def E_1_dual(self, p, y, argument=None):
        if p is 0:
            Dtp = 0
        else:
            Dtp = self.operator(p, 't')
        residual = argument - Dtp
        if self.setup['clip']:
            cutoff = residual + y
            val = torch.where(cutoff > 1, residual - 0.5 * (1 - y)**2,
                              torch.where(cutoff < 0, -0.5 * y**2,
                                          0.5 * (residual**2) + residual * y)).mean()
        else:
            val = 0.5 * (residual**2).mean() + (residual * y).mean()

        if self.operator.bias is not None:
            val = val - (self.operator.bias[None, :, None, None] * p).mean()
        return val

    def E_2_dual(self, p, y, argument=0):
        if self.setup['alpha'] > 0 and self.setup['epsilon'] > 0:
            val = 0.5 / self.setup['alpha'] * (p**2 * self.setup['epsilon']).sum()
            return val / y.numel()
        else:
            return 0

    def smoothness(self):
        """
            This function returns 1/L(theta) where L(theta) is the L-smoothness of E
        """
        if self.setup['epsilon'] > 0:
            op_norm = self.operator.weight.norm()  # Estimate, as normest is not really diff'able
            return 1 / (1 + op_norm**2 / self.setup['epsilon'])
        else:
            return 0

    def primal_from_dual(self, dual_vector, y):
        return y - self.operator(dual_vector, 't')

    def minimize(self, y, x0=0, p0=0, return_dual=False, alpha=None):
        if alpha is None:
            alpha = self.setup['alpha']
        optimizer = bilevelsurrogates.inference.SparseAnalysis(y, self.operator, alpha=alpha,
                                                               epsilon=self.setup['epsilon'], norm=self.setup['norm'],
                                                               clip=self.setup['clip'], x0=x0, p0=p0,
                                                               max_iterations=self.setup['inference']['max_iterations'],
                                                               tolerance=self.setup['inference']['tolerance'])
        optimizer.verbose = False
        optimizer.run()
        if return_dual:
            return optimizer.x, optimizer.p
        else:
            return optimizer.x

    def differentiable_minimize(self, y, iterations=None, alpha=None):
        if iterations is None:
            iterations = self.setup['inference']['max_iterations']
        if alpha is None:
            alpha = self.setup['alpha']
        optimizer = bilevelsurrogates.inference.BregmanSparseAnalysis(
            y, self.operator, alpha=alpha,
            epsilon=self.setup['epsilon'], norm=self.setup['norm'],
            clip=self.setup['clip'], x0=x0, p0=p0,
            max_iterations=iterations,
            min_iterations=iterations,
            tolerance=self.setup['inference']['tolerance'],
            requires_grad=True)
        optimizer.verbose = False
        optimizer.run()
        return optimizer.x

    def prox_primal_step(self, x, w1=None, w2=None, tau=1.0):
        if self.setup['clip']:
            x.data.clamp_(0, 1)
        else:
            pass

    def huber(self, x):
        """
        Huber function in standard definition as infconv between |x| and epsilon/2|x|**2
        """
        C = x.shape[1]
        if self.setup['norm'] == 'aniso':
            local_norm = x.abs()
        elif self.setup['norm'] == 'iso2':
            local_norm = torch.sqrt(x[:, 0:C:2, :, :]**2 + x[:, 1:C:2, :, :]**2 + 1e-10)        # iso2
        elif self.setup['norm'] == 'iso6':
            local_norm = torch.sqrt(x[:, 0:C:6, :, :]**2 + x[:, 1:C:6, :, :]**2
                                    + x[:, 2:C:6, :, :]**2 + x[:, 3:C:6, :, :]**2
                                    + x[:, 4:C:6, :, :]**2 + x[:, 5:C:6, :, :]**2 + 1e-10)      # iso6
        elif self.setup['norm'] == 'infiso':
            local_norm = x[:, 0:C:2, :, :].abs() + x[:, 1:C:2, :, :].abs()              # inf aniso
        else:
            raise ValueError()

        if self.setup['epsilon'] > 0:
            return torch.where(local_norm > self.setup['epsilon'],
                               local_norm - self.setup['epsilon'] / 2, 0.5 * local_norm**2 / self.setup['epsilon'])
        else:
            return local_norm

    def prox_dual_step(self, p, sigma=1.0):
        """
        efficiently project onto p in ||.||_infty < alpha
        """
        if self.setup['alpha'] > 0:
            # 1) Huber correction
            p *= self.setup['alpha'] / (self.setup['alpha'] + sigma * self.setup['epsilon'])

            # 2) local norm
            C = p.shape[1]

            if self.setup['norm'] == 'aniso':
                pnorm = p.abs()
            elif self.setup['norm'] == 'iso2':
                pnorm = torch.sqrt(p[:, 0:C:2, :, :]**2 + p[:, 1:C:2, :, :]**2)  # iso
                pnorm = torch.stack((pnorm, pnorm), dim=2).view_as(p)
            elif self.setup['norm'] == 'iso6':
                pnorm = torch.sqrt(p[:, 0:C:6, :, :]**2 + p[:, 1:C:6, :, :]**2
                                   + p[:, 2:C:6, :, :]**2 + p[:, 3:C:6, :, :]**2
                                   + p[:, 4:C:6, :, :]**2 + p[:, 5:C:6, :, :]**2)  # iso6
                pnorm = torch.stack([pnorm] * 6, dim=2).view_as(p)
            elif self.setup['norm'] == 'infiso':
                pnorm = p[:, 0:C:2, :, :].abs() + p[:, 1:C:2, :, :].abs()  # inf aniso
                pnorm = torch.stack((pnorm, pnorm), dim=2).view_as(p)
            else:
                raise ValueError()

            # 3) Projection
            p *= self.setup['alpha'] / torch.max(self.setup['alpha'], pnorm)
        else:  # protect against NaNs in singular case alpha=0
            p.zero_()


class TotalVariation(AnalysisSparsity):
    """
        Sanity check class. Only the scalar parameter alpha in front of a total variation term is learned.
    """
    def __init__(self, operator, setup):
        gradient = bilevelsurrogates.Gradient(setup['x_dims'][1]).to(**setup['data'])
        super().__init__(gradient, setup)
