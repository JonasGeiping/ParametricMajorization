import torch
import numpy as np
import time
import skimage.restoration as skir


def training_error(energy, loss, samples):
    timer = time.time()
    output = energy.minimize(samples.y)
    val, val_n, val_f = loss(samples.x, output)
    metric, metric_n, metric_f = loss.metric(samples.x, output)
    m, s = divmod(time.time() - timer, 60)
    print(f'TRAINING | {val_n}: {val:{val_f}} | {metric_n} for these samples: {metric:{metric_f}}, '
          f' Time: {m:02.0f}:{s:02.0f} mins')
    return output, metric


def testing_error(energy, loss, testset, batch_size=17):
    timer = time.time()
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=1, drop_last=False)
    if len(testset) % batch_size != 0:
        raise ValueError('Invalid batch size, choose a divisor of dataset length.')
    loss_vals = []
    metric_vals = []
    for data in testloader:
        test_ref, test_in = data
        img_input = test_in.to(**energy.setup['data'])
        test_ref = test_ref.to(**energy.setup['data'])
        test_out = energy.minimize(img_input)
        val, val_n, val_f = loss(test_ref, test_out)
        metric, metric_n, metric_f = loss.metric(test_ref, test_out)
        loss_vals.append(val.item())
        metric_vals.append(metric)
    metr = np.mean(metric_vals)
    m, s = divmod(time.time() - timer, 60)
    print(f'TESTING | {val_n}: {np.mean(loss_vals):{val_f}} | {metric_n} for the given testset: {metr:{metric_f}}'
          f', Time: {m:02.0f}:{s:02.0f} mins---')
    return test_out, metr


def test_twice_error(energy, loss, testset, batch_size=8, noise=25 / 255):
    timer = time.time()
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=1, drop_last=False)
    loss_vals = []
    metric_vals = []
    for data in testloader:
        test_ref, test_in = data
        img_input = test_in.to(**energy.setup['data'])
        test_ref = test_ref.to(**energy.setup['data'])
        test_out = energy.minimize(img_input)
        noise_remaining = skir.estimate_sigma(test_out.permute(2, 3, 0, 1).cpu().numpy(),
                                              multichannel=True, average_sigmas=True)
        alpha_corrected = energy.setup['alpha'] * noise_remaining / noise
        test_out = energy.minimize(test_out, alpha=alpha_corrected)
        val, val_n, val_f = loss(test_ref, test_out)
        metric, metric_n, metric_f = loss.metric(test_ref, test_out)
        loss_vals.append(val.item())
        metric_vals.append(metric)
    metr = np.mean(metric_vals)
    m, s = divmod(time.time() - timer, 60)
    print(f'TESTING | {val_n}: {np.mean(loss_vals):{val_f}} | {metric_n} for the given testset: {metr:{metric_f}}'
          f', Time: {m:02.0f}:{s:02.0f} mins---')
    return test_out, metr
