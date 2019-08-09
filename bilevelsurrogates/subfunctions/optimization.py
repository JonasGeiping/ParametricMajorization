import torch
import time
from collections import defaultdict


class Optimization:
    """
    Basics for an iterative algorithm solving a minimization problem
    """
    def __init__(self, problem, min_iterations=100, max_iterations=2500, validation=-1, verbose=True, tolerance=1e-4):

        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.validation = validation
        self.verbose = verbose
        self.tolerance = tolerance

        self.stats = defaultdict(list)

    def run(self):
        self.initialize()
        self.iterate()

        return self

    def rerun(self):
        self.iterate()

        return self

    def iterate(self):
        start_time = time.time()

        with torch.no_grad():
            for self.iteration in range(self.max_iterations):
                self.step()

                # Test progess
                progress = self.check_error()

                # Compute stats
                self.stats['progress'].append(progress.item())
                if (self.iteration % self.validation == 0) & (self.validation >= 0):
                    self.compute_statistics()

                # Print Status
                self.print_status(self.iteration, progress)

                if self.iteration >= self.min_iterations:
                    if progress <= self.tolerance:
                        break

        # Finale
        m, s = divmod(time.time() - start_time, 60)
        if self.verbose:
            print(f'status: finished  {self.iteration+1:>{len(str(self.max_iterations))}}  iterations'
                  f' in {m:.0f} minutes and {s:.2f} seconds. tol: {100*self.check_error().item():.3f}%')
        self.stats['iterations'].append(self.iteration + 1)
        self.finalize()

        return self

    def initialize(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def check_error(self):
        raise NotImplementedError()
        return progress

    def print_status(self, iteration, progress):
        if (iteration % 1000 == 0) and self.verbose:
            print(f'status: iteration {iteration:>{len(str(self.max_iterations))}}, tol: {100*progress.item():.3f}%')

    def compute_statistics(self):
        pass

    def finalize(self):
        raise NotImplementedError()


class ConvexOptimization(Optimization):
    """
    Basics for an iterative algorithm solving a minimization problem
    """
    def __init__(self, operator, data, alpha, min_iterations=100, max_iterations=2500, tolerance=1e-4, validation=-1,
                 verbose=True):

        super().__init__(data, min_iterations, max_iterations, validation, verbose)
        self.device = data.data.device
        self.dtype = data.data.dtype

        self.tolerance = tolerance
        [self.B, self.c, self.m, self.n] = data.shape
        self.C = operator.out_channels
        self.operator = operator
        self.data = torch.as_tensor(data, device=self.device, dtype=self.dtype)
        self.alpha = torch.as_tensor(alpha, device=self.device, dtype=self.dtype)

    def norm_project(self, p, alpha='self'):
        """
        efficiently project onto p in ||.||_infty < alpha
        """
        if alpha == 'self':
            alpha = self.alpha
        else:
            alpha = torch.as_tensor(alpha, device=self.device, dtype=self.dtype)

        if alpha > 0:
            return alpha * p / torch.max(alpha, p.abs())
        else:  # protect against NaNs in singular case alpha=0
            return p.zero_()
