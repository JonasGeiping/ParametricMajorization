"""
Miscellaneous functions
"""

import torch
import numpy as np
import random

import matplotlib.pyplot as plt


def normalize_image(img):
    ''' Normalize image to [0,1], img is a numpy array'''
    return (img - img.min()) / (img.max() - img.min())


def visualize(dictionary):
    weights = dictionary.return_filters().clone().detach()
    num_filters = weights.shape[0]

    plot_horz = int(np.ceil(np.sqrt(num_filters)))
    fig, axes = plt.subplots(plot_horz, int(np.ceil(num_filters / plot_horz)), figsize=(20, 10))
    fmax = weights.max()
    fmin = weights.min()
    for i, ax in enumerate(axes.flatten()):
        try:
            ax.imshow(((weights[i, :, :, :] - fmin) / (fmax - fmin)
                       ).permute(1, 2, 0).detach().cpu().numpy().squeeze())
            ax.axis('off')
            if dictionary.bias is not None:
                ax.set_title(f'Bias: {dictionary.bias.data[i].item():.2f}')
        except IndexError:
            ax.axis('off')
    fig.subplots_adjust(hspace=0.3)
    fig.canvas.draw()
    print(f'Dictionary norm is {weights.norm().item():.3f}.')
    return fig


def visualize_potentials(operator, class_names=None, mode='rescale'):
    weights = operator.return_filters().clone().detach()
    num_filters = weights.shape[0]

    plot_horz = int(np.ceil(np.sqrt(num_filters)))
    fig, axes = plt.subplots(plot_horz, int(np.ceil(num_filters / plot_horz)), figsize=(20, 10))

    if mode == 'clip':
        weights.clamp_(0, 1)
    elif mode == 'rescale':
        fmax = weights.max()
        fmin = weights.min()
        weights = (weights - fmin) / (fmax - fmin)
    for i, ax in enumerate(axes.flatten()):
        try:
            ax.imshow(weights[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy().squeeze())
            ax.axis('off')
            title = ''
            if class_names is not None:
                title += class_names[i + 1] + ' '
            if operator.bias is not None:
                title += f', bias: {operator.bias.data[i].item():.2f}'
            ax.set_title(title)

        except IndexError:
            ax.axis('off')
    fig.subplots_adjust(hspace=0.3)
    fig.canvas.draw()
    print(f'Dictionary norm is {operator.return_filters().norm().item():.3f}.')


def deterministic(seed=233):
    """
    233 = 144 + 89 is my favorite number
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
