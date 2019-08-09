"""
    Segmentation Models
"""
import torch
import torch.nn.functional as F
import bilevelsurrogates

from .general_model import Energy


class Segmentation(Energy):
    """
        The model is E(x, y, theta)  = -< D_1(theta, y), x > + h(x) + alpha*||D_2(theta)x||_epsilon
    """
    def __init__(self, data_operator, setup, grad_operator=None):
        super().__init__(setup['x_dims'], setup)
        self.operator = data_operator
        if grad_operator is None:
            self.grad = bilevelsurrogates.Gradient(setup['x_dims'][1], scalable=False).to(**setup['data'])
        else:
            self.grad = grad_operator

        self.setup['alpha'] = torch.as_tensor(self.setup['alpha'], **self.setup['data'])
        self.setup['epsilon'] = torch.as_tensor(self.setup['epsilon'], **self.setup['data'])

    def prepare_training(self, x_star_layers, data):
        if self.setup['only_valid']:
            self.valid_pixels = x_star_layers.sum(dim=1)
        else:
            self.valid_pixels = torch.ones_like(x_star_layers.sum(dim=1))
        self.normalization = self.valid_pixels.float().sum()

    def forward(self, x, y, w=None):
        return self.E_1(x, y, w) + self.E_2(x, y, w)

    def dual(self, p, y, argument=0):
        return self.E_1_dual(p, y, argument) + self.E_2_dual(p, y, argument=0)

    def E_1(self, x, y, w=None):
        if self.setup['only_valid']:
            return ((-(self.operator(y) * x) + self.entropy(x)).sum(1) * self.valid_pixels).sum() / self.normalization
        else:
            return (-(self.operator(y) * x).sum() + self.entropy(x).sum()) / self.normalization

    def entropy(self, x):
        return torch.where(x > 0, x * torch.log(x), torch.zeros_like(x))

    def E_2(self, x, y, w=None):
        if self.setup['alpha'] > 0:
            return self.setup['alpha'] * self.huber(self.grad(x)).sum() / self.normalization
        else:
            return 0

    def E_1_dual(self, p, y, argument=0):
        if p is 0:
            Dtp = 0
        else:
            Dtp = self.grad(p, 't')
        z = argument - Dtp

        if self.setup['only_valid']:
            val = (torch.logsumexp(self.operator(y) + z, dim=1) * self.valid_pixels).sum() / self.normalization
        else:
            val = torch.logsumexp(self.operator(y) + z, dim=1).sum() / self.normalization

        if self.grad.bias is not None:
            val = val - (self.grad.bias[None, :, None, None] * p) / self.normalization
        return val

    def E_2_dual(self, p, y, argument=0):
        if self.setup['alpha'] > 0 and self.setup['epsilon'] > 0:
            val = 0.5 / self.setup['alpha'] * (p**2 * self.setup['epsilon']).sum() / self.normalization
        else:
            val = 0
        return val

    def initialize_primal(self, x):
        return torch.ones_like(x) * 1 / self.setup['x_dims'][1]

    def smoothness(self):
        """
            This function returns 1/L(theta) where L(theta) is the L-smoothness of E
        """
        if self.setup['epsilon'] > 0:
            grad_norm = self.grad.weight.norm(dim=[2, 3]).sum()
            return self.setup['epsilon'] / grad_norm**2
        else:
            return 0

    def primal_from_dual(self, dual_vector, y):
        return F.softmax(self.operator(y) - self.grad(dual_vector, 't'), dim=1)

    def minimize(self, y, x0=0, p0=0, return_dual=False):
        if self.setup['alpha'] > 0:
            optimizer = bilevelsurrogates.inference.ConvexSegmentation(
                self, y, max_iterations=self.setup['inference']['max_iterations'],
                tolerance=self.setup['inference']['tolerance'])
            optimizer.verbose = False
            optimizer.run()
            if return_dual:
                return optimizer.x, optimizer.p
            else:
                return optimizer.x
        else:
            return F.softmax(self.operator(y), dim=1)

    def test_minimize(self, y, x0=0, p0=0, return_dual=False):
        if self.setup['alpha'] > 0:
            optimizer = bilevelsurrogates.inference.ConvexSegmentation(
                self, y, relaxation='loose', legendre='euclidean', entropy=0,
                max_iterations=self.setup['inference']['max_iterations'],
                tolerance=self.setup['inference']['tolerance'])
            optimizer.verbose = False
            optimizer.run()
            if return_dual:
                return optimizer.x.argmax(dim=1), optimizer.p
            else:
                return optimizer.x.argmax(dim=1)
        else:
            return self.maximal_potential(y)

    def maximal_potential(self, y):
        return torch.argmax(self.operator(y), dim=1)

    def initialize_dual(self, x):
        return self.grad(x)

    def prox_primal_step(self, primal_vector, w1=None, w2=None, tau=1.0):
        # return self.simplex_projection_(primal_vector)
        return F.softmax(primal_vector, dim=1)

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

    def prox_tight_dual_step(self, p, sigma=1.0):
        """
        project p onto |p_i - p_j | < alpha \forall i<j
        which is the dual to the largest local convex functional relaxation
        as discussed in Chambolle, Cremers, Pock
        "A convex approach to minimal partitions"
        """
        alpha = self.setup['alpha']
        B, _, M, N = p.shape

        if alpha <= 0:
            return p.zero_()

        def check_constraints(p):
            for i2 in range(self.operator.out_channels):
                for i1 in range(i2):
                    p_diff = (p[:, 2 * i2:2 * i2 + 2, :, :] - p[:, 2 * i1:2 * i1 + 2, :, :])
                    pnorm = torch.sqrt(p_diff[:, 0, :, :]**2 + p_diff[:, 1, :, :]**2)
                    if pnorm.max() > alpha:
                        return False
            return True

        # Compute a list of index pairs violating the constraints
        constraints = []
        for i2 in range(self.operator.out_channels):
            for i1 in range(i2):
                p_diff = p[:, 2 * i2:2 * i2 + 2, :, :] - p[:, 2 * i1:2 * i1 + 2, :, :]
                pnorm = torch.sqrt(p_diff[:, 0, :, :]**2 + p_diff[:, 1, :, :]**2)
                if pnorm.max() > alpha:
                    constraints.append([i1, i2])
        # pytorch 1.0.* : F.pdist(p=2) could also be used

        # Run Dykstra's projection algorithm, i.e. an augmented Lagrangian:
        # a scalar alpha trivially satisfies the surface tension triangle
        # inquality s_ij^2 \leq s_ik^2 + s_kj^2
        # hence we can get away with projecting only violated constraints

        slack = p.new_zeros(B, 2 * len(constraints), M, N)

        # Actually running only 2*C subiterations seems ok
        for i in range(2 * self.operator.out_channels):
            for idx, i1i2 in enumerate(constraints):
                i1, i2 = i1i2  # most important step...
            # (a)
            p_diff = (p[:, 2 * i2:2 * i2 + 2, :, :] - p[:, 2 * i1:2 * i1 + 2, :, :]
                      + slack[:, 2 * idx:2 * idx + 2, :, :])
            pnorm = torch.sqrt(p_diff[:, 0, :, :]**2 + p_diff[:, 1, :, :]**2).unsqueeze(1)
            # (b)
            # This is (|p| - alpha)^+p/|p|
            #       = (|p| - alpha)^(+)*p/max(|p|,alpha)
            #       = (|p| - alpha)^(+)*(p/(|p| - alpha)^(+)*+alpha)
            pi1i2 = F.relu(pnorm - alpha)
            pi1i2 = pi1i2 * p_diff / (pi1i2 + alpha)
            # if alpha > 0:
            # 	pi1i2 = alpha*p_diff/torch.max(alpha,pnorm)
            # else: # protect against NaNs in singular case alpha=0
            # 	pi1i2 = p_diff.zero_()

            # (c)
            update = 0.5 * (pi1i2 - slack[:, 2 * idx:2 * idx + 2, :, :])
            p[:, 2 * i1:2 * i1 + 2, :, :] += update
            p[:, 2 * i2:2 * i2 + 2, :, :] -= update
            # (d)
            slack[:, 2 * idx:2 * idx + 2, :, :] = pi1i2.clone()

            # (e) Check if all constraints are fulfilled
            if check_constraints(p):
                break

    def simplex_projection_(self, v):
        """
        efficiently project onto u in Delta
        follows the usual Duchy et al sorting strategy
        adapted from python code at
        https://gist.github.com/mblondel/6f3b7aaad90606b98f71
        """
        channels = v.shape[1]
        u, _ = v.sort(dim=1, descending=True)
        cssv = u.cumsum(dim=1) - 1.0
        ind = torch.arange(1, channels + 1, dtype=v.dtype, device=v.device)
        cond = u - cssv / ind[None, :, None, None] > 0
        rho = torch.sum(cond, dim=1, keepdim=True)
        theta = torch.gather(cssv, 1, F.relu(rho - 1)) / rho.to(v.dtype)
        v.data = torch.max(v - theta, torch.zeros_like(v))
        return v

    def label_to_layer(self, x_labels, num_labels):
        [B, M, N] = x_labels.shape
        label_layer = torch.zeros(B, num_labels, M, N, dtype=torch.uint8, device=x_labels.device)
        for channel in range(num_labels):
            label_layer[:, channel, :, :] = (x_labels == channel)
        return label_layer
