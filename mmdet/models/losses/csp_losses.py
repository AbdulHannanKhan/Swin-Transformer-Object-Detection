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

        # sel = h_label[:, 0]
        # msk = h_label[:, -1] == 1
        # sel = sel[msk]
        # mn = torch.min(sel)

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
        # if self.loss_weight * reg_loss > 1.0:
        #     print(f"\nmin: {mn}\n")
        return self.loss_weight * reg_loss


@LOSSES.register_module()
class MiDLoss(nn.Module):
    def __init__(self,
                 reduction='none',
                 loss_weight=1e1):
        """RegLoss.

        Args:
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            loss_weight (float, optional): Weight of the loss. Defaults to 10.0.
        """
        super(MiDLoss, self).__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

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

        if h_pred.shape[2:] != h_label.shape[2:]:
            print("h_pred size: ", h_pred.shape)
            print("h_label size: ", h_label.shape)

        # check if h_pred and h_label are the same size
        assert h_pred.shape[2:] == h_label.shape[2:], "h_pred and h_label should be the same size"

        pos_ind = h_label[:, -1, :, :] == 1
        probs = h_pred[:, 0, :, :]
        probs = probs[pos_ind]
        labels = h_label[:, 0, :, :]
        labels = labels[pos_ind]

        mask = torch.logical_and(labels > 2e-1, labels < 2)
        probs = probs[mask]
        labels = labels[mask]
        probs = probs.clamp(min=1e-10)

        mid_loss = torch.abs(torch.log(probs) - torch.log(labels))

        return self.loss_weight * mid_loss


@LOSSES.register_module()
class TVRoDLoss(nn.Module):
    def __init__(self,
                 reduction='none', tv_weight=2e-6, rod_weight=0.1):
        """RegLoss.

        Args:
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            loss_weight (float, optional): Weight of the loss. Defaults to 10.0.
        """
        super(TVRoDLoss, self).__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.tv_weight = tv_weight
        self.rod_weight = rod_weight

    def forward(self,
                h_pred,
                h_label,
                **kwargs):
        """Forward function.

        Args:
            h_pred (torch.Tensor[n, 1, H//4, W//4]): The prediction.
            h_label (torch.Tensor[n, 2, H//4, W//4]): n, 0 => ttc, n, 1 => weight mask
        Returns:
            torch.Tensor: The calculated loss
        """

        if h_pred.shape[2:] != h_label.shape[2:]:
            print("h_pred size: ", h_pred.shape)
            print("h_label size: ", h_label.shape)

        # check if h_pred and h_label are the same size
        assert h_pred.shape[2:] == h_label.shape[2:], "h_pred and h_label should be the same size"

        rod_loss = h_label[:, 1, :, :] * torch.abs(h_pred[:, 0, :, :] - h_label[:, 0, :, :])
        rod_loss = torch.sum(rod_loss) / max(1.0, torch.sum(h_label[:, 1, :, :]))

        tv_loss = (torch.sum(torch.abs(h_pred[:, 0, :, :-1] - h_pred[:, 0, :, 1:])) +
                                                            torch.sum(torch.abs(h_pred[:, 0, :-1, :] - h_pred[:, 0, 1:, :])))
        tv_loss = tv_loss.mean()
        return self.tv_weight * tv_loss, self.rod_weight * rod_loss


@LOSSES.register_module()
class L1RoDLoss(nn.Module):
    def __init__(self,
                 reduction='none',
                 loss_weight=0.25):
        """RegLoss.

        Args:
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            loss_weight (float, optional): Weight of the loss. Defaults to 10.0.
        """
        super(L1RoDLoss, self).__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

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

        if h_pred.shape[2:] != h_label.shape[2:]:
            print("h_pred size: ", h_pred.shape)
            print("h_label size: ", h_label.shape)

        # check if h_pred and h_label are the same size
        assert h_pred.shape[2:] == h_label.shape[2:], "h_pred and h_label should be the same size"

        pos_ind = h_label[:, -1, :, :] == 1
        probs = h_pred[:, 0, :, :]
        probs = probs[pos_ind]
        labels = h_label[:, 0, :, :]
        labels = labels[pos_ind]

        mask = torch.logical_and(labels > 2e-1, labels < 2)
        probs = probs[mask]
        labels = labels[mask]
        # probs = probs.clamp(min=1e-10)

        rod_loss = torch.abs(probs - labels)
        return self.loss_weight * rod_loss


@LOSSES.register_module()
class TTCLoss(nn.Module):
    def __init__(self,
                 reduction='none',
                 loss_weight=1e-2):
        """RegLoss.

        Args:
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(TTCLoss, self).__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
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
        t = 1/12
        pos_ind = h_label[:, -1, :, :] == 1
        probs = h_pred[:, 0, :, :]
        probs = probs[pos_ind]
        labels = h_label[:, 0, :, :]
        pos_probs = 1 - t/probs
        pos_labs = 1 - t/labels[pos_ind]

        l1_loss = self.l1(torch.log(pos_probs), torch.log(pos_labs))

        reg_loss = torch.sum(l1_loss) / max(1.0, torch.sum(h_label[:, -1, :, :]))
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
class MultiClassCenterLoss(nn.Module):
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
        super(MultiClassCenterLoss, self).__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(reduction=self.reduction)

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
        pos_pred_sgm = pos_pred
        # assigned_box = 0
        log_loss = self.ce(pos_pred, pos_label)

        for i in range(classes):
            ignore_ind = 0
            mask_ind = 1 + 2 * i
            center_ind = 2 + 2 * i

            # log_loss = self.bce(pos_pred[:, i, :, :], pos_label[:, center_ind, :, :])

            positives = pos_label[:, center_ind, :, :]
            negatives = pos_label[:, ignore_ind, :, :] - pos_label[:, center_ind, :, :]

            fore_weight = positives * (1.0 - pos_pred_sgm[:, i, :, :]) ** 2
            back_weight = negatives * ((1.0 - pos_label[:, mask_ind, :, :]) ** self.beta) * (
                        pos_pred_sgm[:, i, :, :] ** self.alpha)
            focal_weight = fore_weight + back_weight

            assigned_box = torch.sum(pos_label[:, center_ind, :, :])

            if cls_loss is not None:
                cls_loss = cls_loss + torch.sum(focal_weight * log_loss) / max(1.0, assigned_box)
            else:
                cls_loss = torch.sum(focal_weight * log_loss) / max(1.0, assigned_box)
        cls_loss = cls_loss / classes
        return self.loss_weight * cls_loss


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
        # assigned_box = 0

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

            assigned_box = torch.sum(pos_label[:, center_ind, :, :])

            if cls_loss is not None:
                cls_loss = cls_loss + torch.sum(focal_weight * log_loss)/max(1.0, assigned_box)
            else:
                cls_loss = torch.sum(focal_weight * log_loss)/max(1.0, assigned_box)
        cls_loss = cls_loss/classes
        return self.loss_weight * cls_loss

@LOSSES.register_module()
class QTTCLoss(nn.Module):
    def __init__(self,
                 loss_weight=0.1):
        """CenterLoss.

        Args:
            loss_weight (float, optional): Weight of the loss. Defaults to 0.1.
        """
        super(QTTCLoss, self).__init__()
        self.loss_weight = loss_weight
        self.bce = nn.BCELoss(reduction='none')

    def forward(self,
                ttc_pred,
                ttc_label,
                **kwargs):
        """Forward function.

        Args:
            ttc_pred (torch.Tensor[n, bins, H//4, W//4]): The prediction.
            ttc_label (torch.Tensor[n, 1 + bins, H//4, W//4]): The learning label of the prediction.
        Returns:
            torch.Tensor: The calculated loss
        """
        bins = ttc_pred.shape[1]
        mask = ttc_label[:, 0, :, :] == 1
        ttc_label = ttc_label[:, 1:, :, :]
        mask = mask.reshape(-1)
        ttc_loss = 0

        for i in range(bins):

            log_loss = self.bce(ttc_pred[:, i, :, :], ttc_label[:, i, :, :])
            log_loss = log_loss.reshape(-1)[mask].mean()

            ttc_loss = ttc_loss + log_loss
        ttc_loss = ttc_loss / bins
        return self.loss_weight * ttc_loss