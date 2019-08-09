"""
Implement linear operators
"""

import torch
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init

from .normest import normest


class Convolution(torch.nn.Conv2d):
    """
    Instantiate a learnable convolution operator with two extra options.
    - forward can now also be called as a transpose
    - the weights can be filled with gradient weights (this also set groups == inChannels)
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, mean=False):

        padding = int(np.floor(kernel_size / 2))
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

    def forward(self, input, direction='op'):
        if direction == 'op':
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        elif direction == 't':
            return F.conv_transpose2d(input, self.weight, None, self.stride, output_padding=0,
                                      padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            raise ValueError('Invalid Direction')

    def initialize_gradient(self):
        grad_weight = self.weight.new_tensor([[0, 0, 0],
                                             [0, -1, 1],
                                             [0, 0, 0]]).unsqueeze(0).unsqueeze(1)
        grad_weight = torch.cat((torch.transpose(grad_weight, 2, 3), grad_weight), 0)
        grad_weight = torch.cat([grad_weight] * self.in_channels , 0)
        self.weight = Parameter(grad_weight)
        self.groups = self.in_channels
        self.padding = [1, 1]
        self.out_channels = self.in_channels * 2

        return self

    def initialize_symmetric_gradient(self):
        grad_weight = self.weight.new_tensor([[0, 0, 0],
                                             [0, -1, 1],
                                             [0, 0, 0]]).unsqueeze(0).unsqueeze(1)
        grad_weight = torch.cat((torch.transpose(grad_weight, 2, 3), grad_weight), 0)

        sym_weight = self.weight.new_tensor([[0, 0, 0],
                                             [0, -1, 0.5],
                                             [0, 0.5, 0]]).unsqueeze(0).unsqueeze(1)
        sym_weight = torch.cat((torch.transpose(sym_weight, 2, 3), sym_weight), 0)
        final_weight = torch.cat([grad_weight, sym_weight] * self.in_channels , 0)
        self.weight = Parameter(final_weight)
        self.groups = self.in_channels
        self.padding = [1, 1]
        self.out_channels = self.in_channels * 4

        return self

    def initialize_locked_groups(self):
        grad_weight = self.weight[:, 0:1, :, :]
        self.weight = Parameter(grad_weight)
        self.groups = self.in_channels
        self.padding = [1, 1]

        return self

    def initialize_tri_operator(self):
        depth_weight = self.weight.new_zeros(self.in_channels, self.out_channels, 1, 1)
        for i in range(self.out_channels - 1):
            depth_weight[i, i, :, :] = 1
            depth_weight[i, i + 1, :, :] = -1
        depth_weight[-1, -1, :, :] = 1
        self.weight = Parameter(depth_weight)

        return self

    def return_filters(self):
        with torch.no_grad():
            return self.weight

    def normest(self):
        # return self.weight.norm(dim=[2, 3]).sum() / self.groups
        return normest(self, self.in_channels, verbose=False)


class Gradient(Convolution):
    """
        Gradient Operator. Only its scalar weight can be learned.
    """
    def __init__(self, in_channels, scalable=True):
        super().__init__(in_channels, in_channels * 2, 3, bias=False, mean=False)
        self.initialize_gradient()
        grad_weight = self.weight.clone().detach()
        del self.weight
        self.register_buffer('weight', grad_weight)
        if scalable:
            self.alpha = Parameter(torch.tensor(1.0))
        else:
            self.alpha = 1

    def forward(self, input, direction='op'):
        if direction == 'op':
            return self.alpha * F.conv2d(input, self.weight, self.bias, self.stride,
                                         self.padding, self.dilation, self.groups)
        elif direction == 't':
            return self.alpha * F.conv_transpose2d(input, self.weight, None, self.stride, output_padding=0,
                                                   padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            raise ValueError('Invalid Direction')


class DCTConvolution(torch.nn.Conv2d):
    """
    Instantiate a learnable convolution operator - the trainable weights are the coefficients of the
    DCT decomposition of the convolution operator (i.e. this change is bijective for convex training procedures!)
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, mean=False, dilation=1):

        self.mean = mean
        padding = int(np.floor(kernel_size / 2))
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias, dilation=dilation)

        self.initialize_DCT()
        torch.nn.init.orthogonal_(self.weight, gain=1)

    def initialize_DCT(self):
        num_basis_functions = self.kernel_size[0] * self.kernel_size[1]
        dct_basis = self.weight.new_zeros(num_basis_functions, 1, *self.kernel_size)

        b = 0  # enumerate basis functions
        for b1 in range(self.kernel_size[0]):
            for b2 in range(self.kernel_size[1]):
                for i in range(self.kernel_size[0]):
                    for j in range(self.kernel_size[1]):
                        dct_basis[b, 0, i, j] = (np.cos(np.pi / self.kernel_size[0] * (i + 0.5) * b1)
                                                 * np.cos(np.pi / self.kernel_size[1] * (j + 0.5) * b2))
                b += 1

        if not self.mean:
            dct_basis = dct_basis[1:, 0:1, :, :]
            num_weights = (num_basis_functions - 1) * self.in_channels
            self.weight = Parameter(self.weight.new_zeros(self.out_channels, num_weights, 1, 1))
        else:
            num_weights = num_basis_functions * self.in_channels
            self.weight = Parameter(self.weight.new_zeros(self.out_channels, num_weights, 1, 1))

        self.register_buffer('dct_basis', torch.cat([dct_basis] * self.in_channels , 0) / torch.norm(dct_basis))
        init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, input, direction='op'):
        if direction == 'op':
            dct_response = F.conv2d(input, self.dct_basis, None, self.stride, self.padding,
                                    self.dilation, groups=self.in_channels)
            return F.conv2d(dct_response, self.weight, self.bias, self.stride,
                            padding=0, dilation=self.dilation, groups=self.groups)
        elif direction == 't':
            input_weighted = F.conv_transpose2d(input, self.weight, None, self.stride, output_padding=0,
                                                padding=0, dilation=self.dilation, groups=self.groups)
            return F.conv_transpose2d(input_weighted, self.dct_basis, None, self.stride, output_padding=0,
                                      padding=self.padding, dilation=self.dilation, groups=self.in_channels)
        else:
            raise ValueError('Invalid Direction')

    def return_filters(self):
        with torch.no_grad():
            num_basis_functions = self.dct_basis.shape[0] // self.in_channels

            color_weights = self.weight.reshape(self.out_channels, num_basis_functions, self.in_channels, 1, 1)
            return (self.dct_basis[0:num_basis_functions, :, :, :].unsqueeze(0) * color_weights).sum(1)

    def normest(self):
        # return self.weight.norm(dim=[2, 3]).sum() / self.groups
        return normest(self, self.in_channels, verbose=False)
