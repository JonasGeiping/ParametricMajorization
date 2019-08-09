"""
    Implements a simple data structure to work with/without stochastic training seamlessly
"""

import torch
# from collections.abc import Mapping


class Samples:
    """
        Class made to hide loader dynamics for denoising data with Gaussian noise
        Draw a new noise via .redraw_noise()
        Draw a new batch of images via .step()
    """
    def __init__(self, dataset, batch_size, device=torch.device('cpu'), dtype=torch.float, notation='(x,y)',
                 shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.setup = dict(device=device, dtype=dtype)
        self.notation = notation

        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, num_workers=0,
                                                  drop_last=drop_last, shuffle=shuffle)
        if self.notation == '(x,y)':
            self.x, self.y = [d.to(**self.setup) for d in next(iter(self.loader))]
        else:
            self.y, self.x = [d.to(**self.setup) for d in next(iter(self.loader))]
        self.loader_state = iter(self.loader)

    def redraw_noise(self):
        # Add noise
        self.y = self.x + self.dataset.noise_std * torch.randn_like(self.x)
        # Clip to [0,1]
        if self.dataset.clip_to_realistic:
            self.y.clamp_(0, 1)

    def step(self):
        try:
            data = next(self.loader_state)
        except (StopIteration, FileNotFoundError) as e:
            # Reset loader
            self.loader_state = iter(self.loader)
            data = next(self.loader_state)
        if self.notation == '(x,y)':
            self.x, self.y = [d.to(**self.setup) for d in data]
        else:
            self.y, self.x = [d.to(**self.setup) for d in data]

    def reset(self):
        self.loader_state = iter(self.loader)
