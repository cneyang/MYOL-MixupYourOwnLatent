import torch
import numpy as np

def mixup_data(x1, x2, y=None, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    x2 = x2[index]

    mixed_x = lam * x1 + (1 - lam) * x2
    if y is not None:
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam, index
    return mixed_x, x1, x2, lam
