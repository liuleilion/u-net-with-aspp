import torch
import torch.nn.functional as F


def compute_weighted_loss(pred, gt):
        prob, predicted_class = torch.max(gt, dim=1, keepdim=True)

        seg_gt = predicted_class.squeeze(1)# [B,H,W]

        ce_loss = F.cross_entropy(pred, seg_gt, reduction='none')

        pred_label = torch.argmax(pred, dim=1).detach()
        wrong_mask = (pred_label != seg_gt)

        weight = torch.ones_like(seg_gt, dtype=torch.float32)
        weight[wrong_mask] = 5.0

        loss = (ce_loss * weight).mean()
        return loss




