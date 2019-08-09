"""
    Problem-agnostic training
"""
import torch
import numpy as np
import torch.nn.functional as F
import types

import bilevelsurrogates


def default_setup(method='DiscriminativeLearning', algorithm='joint-dual'):
    """
    """
    if method == 'DiscriminativeLearning':
        setup = dict(lr=5e-3, max_iterations=8_000, min_iterations=1, loss_offset=0,
                     tolerance=1e-10, algorithm='Adam', correction=False, callback=1000,
                     validation=-1, verbose=True, L=1, inertia=0)
        if algorithm == 'joint-dual':
            setup['lr'] = 0.1
    elif method == 'IterativeLearning':
        setup = dict(linearizer_iterations=5, tolerance=1e-5, extrapolation_choice='theta-k',
                     backtracking=5, check_loss=True, verbose=True, stochastic=False, restart=True, output='best-loss')
    else:
        raise ValueError()
    return setup
