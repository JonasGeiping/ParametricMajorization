"""
Error measures
"""
from skimage.measure import compare_ssim
import torch
import numpy as np


def ssim_compute(img_batch, ref_batch, batched=True):
    """
    Compute average SSIM value between two torch tensors along the last two dimensions.
    """
    [B, C, m, n] = img_batch.shape
    img_in = (img_batch.view(-1, m, n).permute(1, 2, 0) * 255).to(torch.uint8)
    img_ref = (ref_batch.view(-1, m, n).permute(1, 2, 0) * 255).to(torch.uint8)
    if batched:
        return compare_ssim(img_in.cpu().numpy(), img_ref.cpu().numpy(), multichannel=True)
    else:
        [B, C, m, n] = img_batch.shape
        ssims = []
        for sample in range(B * C):
            ssims.append(compare_ssim(img_in[sample, :, :].cpu().numpy(),
                                      img_ref[sample, :, :].cpu().numpy(),
                                      multichannel=True))
        return np.mean(ssims)


def psnr_compute(img_batch, ref_batch, batched=False, factor=1.0):
    """
        Standard PSNR
    """
    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor**2 / mse)).item()
        elif not torch.isfinite(mse):
            return float('nan')
        else:
            return float('inf')

    if batched:
        psnr = get_psnr(img_batch, ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = np.mean(psnrs)

    return psnr
