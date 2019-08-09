"""
    Norm estimation via Power method
"""

import torch
import warnings
import time


def normest(operator, in_channels=1, x_dims=[64, 64], tol=1e-7, max_iterations=5000, verbose=False):
    """
        normest implementation, following the matlab baseline at
        MATLAB/R2018b/toolbox/matlab/matfun/normest.m
    """
    with torch.no_grad():
        start_time = time.time()
        x = torch.randn(1, in_channels, *x_dims, device=operator.weight.device, dtype=operator.weight.dtype)
        e = x.norm()
        if e == 0:
            return e
        x = x / e
        e0 = 0

        for cnt in range(max_iterations):
            e0 = e.clone()
            Ax = operator(x)
            x = operator(Ax, 't')
            xnorm = x.norm()
            e = xnorm / Ax.norm()
            x = x / xnorm
            if torch.abs(e - e0) < tol * e:
                if verbose:
                    m, s = m, s = divmod(time.time() - start_time, 60)
                    print(f'Tolerance {tol} reached after {cnt+1} iterations in {m:.0f} minutes and {s:.2f} seconds')
                return e
        # Return anyway if not converged
        final_tol = torch.abs(e - e0) / e
        warnings.warn(f'Normest did not convergence to tolerance {tol} within {cnt+1} iterations.'
                      f' Final tol was {final_tol}')
        return e
