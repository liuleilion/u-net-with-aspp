import torch


def dice_loss(pred, gt, eps=1e-6):
    prob = torch.sigmoid(pred)
    dims = (0, 2, 3)
    intersection = (prob * gt).sum(dims)
    union = prob.sum(dims) + gt.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()
