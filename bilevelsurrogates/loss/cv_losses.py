"""
    Specfics Losses
"""
import torch
import numpy as np


import bilevelsurrogates
from .general_losses import Loss, BregmanLoss


class PSNR(Loss):
    """
        A classical MSE target. The minimized criterion is MSE Loss, the actual metric is average PSNR
    """
    def __call__(self, reference=None, argmin=None):
        """
            Return l(x^*,x(theta))
        """
        name = 'MSE'
        format = '.6f'
        if reference is None:
            return name, format
        else:
            value = 0.5 * ((reference - argmin)**2).mean()
            return value, name, format

    def gradient(self, reference, argmin):
        """
            Return the gradient of l(x^*,x(theta)) w.r.t to the second argument
        """
        return argmin - reference

    def metric(self, reference=None, argmin=None):
        """
            The actually sought metric
        """
        name = 'avg PSNR'
        format = '.3f'
        if reference is None:
            return name, format
        else:
            value = bilevelsurrogates.psnr_compute(argmin, reference)
            return value, name, format

    def metric_est(self, majorizer_loss):
        """
            Estimate of metric based on majorizer loss
        """
        name = 'avg PSNR'
        format = '.3f'
        if majorizer_loss > 0:
            value = 10 * np.log10(0.5 / majorizer_loss)  # correct for 1/2 in loss definition vs PSNR definition
        else:
            value = float('NaN')
        return value, name, format


class ClassificationLoss(BregmanLoss):
    """
        The minimized criterion is NLL loss, the actual metric is label accuracy
        todo:https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
    """
    def __init__(self, n_classes=19, valid_pixels=None):
        super().__init__()
        self.n_classes = n_classes
        self.valid_pixels = valid_pixels

    def __call__(self, reference=None, argmin=None):
        """
            Return l(x^*,x(theta))
        """
        name = 'NLLloss'
        format = '1.5f'
        if reference is None:
            return name, format
        else:
            if self.valid_pixels is None:
                self.valid_pixels = torch.ones_like(reference.sum(1))
            normalization = self.valid_pixels.sum()

            # value = torch.where(self.valid_pixels > 0,
            #                    torch.where(argmin > 0,
            #                                -reference * torch.log(argmin),
            #                                torch.zeros_like(argmin)).sum(dim=1),
            #                    torch.zeros_like(self.valid_pixels)).sum() / normalization
            value = ((-reference * torch.where(argmin > 0, argmin.log(),
                                               torch.zeros_like(argmin))).sum(dim=1) * self.valid_pixels).sum()
            value /= normalization

            return value, name, format

    def gradient(self, reference, argmin):
        """
            Return the gradient of l(x^*,x(theta)) w.r.t to the second argument
        """
        if self.valid_pixels is None:
            self.valid_pixels = torch.ones_like(reference.sum(1))
        normalization = self.valid_pixels.sum()

        # grad = torch.where(self.valid_pixels.unsqueeze(1) > 0,
        #                   torch.where(argmin > 0,
        #                               1 - reference / argmin,
        #                               torch.zeros_like(argmin)),
        #                   torch.zeros_like(self.valid_pixels.unsqueeze(1))) / normalization
        grad = torch.where(argmin > 0, 1 - reference / argmin, torch.zeros_like(argmin))

        return grad

    def metric(self, reference=None, argmin=None, labels_given=False):
        """
            The actually sought metric
        """
        name = 'acc '
        format = '.2%'
        if reference is None:
            return name, format
        else:
            if self.valid_pixels is None:
                self.valid_pixels = torch.ones_like(reference.sum(dim=1))
            normalization = self.valid_pixels.sum()
            if labels_given:
                acc_per_pixel = (reference.argmax(dim=1) == argmin).float() * self.valid_pixels
            else:
                acc_per_pixel = (reference.argmax(dim=1) == argmin.argmax(dim=1)).float() * self.valid_pixels
            value = acc_per_pixel.sum() / normalization
            return value.item(), name, format

    def metric_est(self, majorizer_loss):
        """
            Estimate of metric based on majorizer loss, this is kind of a back-of-the-envelope estimate ...
        """
        name = 'acc'
        format = '.2%'
        if majorizer_loss > 0:
            value = np.maximum(np.minimum(1 + majorizer_loss / np.log(1 / self.n_classes), 1), 1 / self.n_classes)
        else:
            value = float('NaN')
        return value, name, format

    def legendre(self, x):
        """
            Simplex entropy, only valid if x is in the simplex, otherwise the function value is infinite
        """
        if valid_pixels is None:
            self.valid_pixels = torch.ones_like(reference.sum(1))
        normalization = self.valid_pixels.sum()
        return torch.where(self.valid_pixels > 0,
                           torch.where(x > 0, x * torch.log(x), torch.zeros_like(x)).sum(dim=1),
                           torch.zeros_like(self.valid_pixels)).sum() / normalization
