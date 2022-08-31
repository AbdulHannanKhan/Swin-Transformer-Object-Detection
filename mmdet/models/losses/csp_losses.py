import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..builder import LOSSES
from .utils import weight_reduce_loss


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


@LOSSES.register_module()
class RegLoss(nn.Module):
    def __init__(self,
                 reduction='none',
                 loss_weight=0.1,
                 reg_param_count=2):
        """RegLoss.

        Args:
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(RegLoss, self).__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_param_count = reg_param_count
        self.l1 = nn.L1Loss(reduction=self.reduction)

    def forward(self,
                h_pred,
                h_label,
                **kwargs):
        """Forward function.

        Args:
            h_pred (torch.Tensor[n, 1, H//4, W//4]): The prediction.
            h_label (torch.Tensor[n, 2, H//4, W//4]): The learning label of the prediction.
        Returns:
            torch.Tensor: The calculated loss
        """
        l1_loss = h_label[:, -1, :, :]*self.l1(h_pred[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10),
                                                    h_label[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10))

        for i in range(1, self.reg_param_count):
            l1_loss = l1_loss + h_label[:, -1, :, :] * self.l1(h_pred[:, i, :, :] / (h_label[:, i, :, :] + 1e-10),
                                          h_label[:, i, :, :] / (h_label[:, i, :, :] + 1e-10))

        # pos_points = h_label[:,1,:,:].reshape(-1).nonzero()
        # if pos_points.shape[0] != 0:
        #     print(h_pred[:, 0,:,:].reshape(-1)[pos_points])
        #     print(h_label[:,0,:,:].reshape(-1)[pos_points])
        reg_loss = torch.sum(l1_loss) / max(1.0, torch.sum(h_label[:, -1, :, :])*self.reg_param_count)
        return self.loss_weight * reg_loss


@LOSSES.register_module()
class OffsetLoss(nn.Module):
    def __init__(self,
                 reduction='none',
                 loss_weight=0.1):
        """OffsetLoss.

        Args:
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(OffsetLoss, self).__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.smoothl1 = nn.SmoothL1Loss(reduction=self.reduction)

    def forward(self,
                offset_pred,
                offset_label,
                **kwargs):
        """Forward function.

        Args:
            offset_pred (torch.Tensor[n, 2, H//4, W//4]): The prediction.
            offset_label (torch.Tensor[n, 3, H//4, W//4]): The learning label of the prediction.
        Returns:
            torch.Tensor: The calculated loss
        """
        l1_loss = offset_label[:, 2, :, :].unsqueeze(dim=1)*self.smoothl1(offset_pred, offset_label[:, :2, :, :])
        off_loss = torch.sum(l1_loss) / max(1.0, torch.sum(offset_label[:, 2, :, :]))

        return self.loss_weight * off_loss


@LOSSES.register_module()
class CenterLoss(nn.Module):
    def __init__(self,
                 alpha=2.0,
                 beta=4.0,
                 reduction='none',
                 loss_weight=0.01):
        """CenterLoss.

        Args:
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CenterLoss, self).__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss(reduction=self.reduction)

    def forward(self,
                pos_pred,
                pos_label,
                **kwargs):
        """Forward function.

        Args:
            pos_pred (torch.Tensor[n, 1, H//4, W//4]): The prediction.
            pos_label (torch.Tensor[n, 3, H//4, W//4]): The learning label of the prediction.
        Returns:
            torch.Tensor: The calculated loss
        """
        classes = pos_pred.shape[1]
        cls_loss = None
        pos_pred_sgm = pos_pred.sigmoid()
        assigned_box = 0

        for i in range(classes):
            ignore_ind = 0
            mask_ind = 1 + 2*i
            center_ind = 2 + 2*i
            
            log_loss = self.bce(pos_pred[:, i, :, :], pos_label[:, center_ind, :, :])

            positives = pos_label[:, center_ind, :, :]
            negatives = pos_label[:, ignore_ind, :, :] - pos_label[:, center_ind, :, :]

            fore_weight = positives * (1.0 - pos_pred_sgm[:, i, :, :]) ** 2
            back_weight = negatives * ((1.0 - pos_label[:, mask_ind, :, :]) ** self.beta) * (pos_pred_sgm[:, i, :, :] ** self.alpha)
            focal_weight = fore_weight + back_weight

            assigned_box += torch.sum(pos_label[:, center_ind, :, :])

            if cls_loss is not None:
                cls_loss = cls_loss + torch.sum(focal_weight * log_loss)
            else:
                cls_loss = torch.sum(focal_weight * log_loss)
        cls_loss = cls_loss / assigned_box
        return self.loss_weight * cls_loss
