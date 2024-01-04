import torch
import torch.nn as nn
from inspect import signature
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from ..builder import HEADS, build_loss
from .csp_ttc_head import CSPTTCHead
from mmdet.core import multi_apply, multiclass_nms, bbox2result
from mmcv.runner import force_fp32

INF = 1e8


class Scale(nn.Module):

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


@HEADS.register_module()
class CSPMiDHead(CSPTTCHead):
    def __init__(self,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 loss_offset=dict(
                     type='OffsetLoss',
                     loss_weight=0.1),
                 loss_cls=dict(
                     type='CenterLoss',
                     beta=4.0,
                     alpha=2.0,
                     loss_weight=0.05),
                 loss_bbox=dict(type='RegLoss', loss_weight=0.01),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 num_classes=1,
                 predict_width=True,
                 bn_for_ttc=False,
                 drop_ttc=0.0,
                 wh_ratio=0.41,
                 loss_ttc=dict(type='MiDLoss', loss_weight=10),
                 **kwargs):

        super(CSPMiDHead, self).__init__(
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            wh_ratio=wh_ratio,
            predict_width=predict_width,
            regress_ranges=regress_ranges,
            loss_offset=loss_offset,
            bn_for_ttc=bn_for_ttc,
            drop_ttc=drop_ttc,
            **kwargs)

        self.loss_ttc = build_loss(loss_ttc)

    def _init_layers(self):
        """Initialize layers of the head."""
        super(CSPMiDHead, self)._init_layers()

        self.ttc_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.ttc_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        if self.bn_for_ttc:
            self.ttc_convs.append(nn.BatchNorm2d(self.feat_channels))
        if self.drop_ttc > 0:
            self.ttc_convs.append(nn.Dropout2d(self.drop_ttc))

        self.csp_ttc = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.ttc_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        super(CSPMiDHead, self).init_weights()
        normal_init(self.csp_ttc, std=0.01)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      classification_maps=None,
                      scale_maps=None,
                      offset_maps=None,
                      proposal_cfg=None,
                      ttc_maps=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, classification_maps=classification_maps, scale_maps=scale_maps,
                           offset_maps=offset_maps, gt_bboxes_ignore=gt_bboxes_ignore, ttc_maps=ttc_maps)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward(self, feats, *args):
        return multi_apply(self.forward_single, feats, self.reg_scales, self.offset_scales, self.ttc_scales)

    def forward_single(self, x, reg_scale, offset_scale, ttc_scale):
        cls_feat = x
        reg_feat = x
        offset_feat = x
        ttc_feat = x

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)

        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        for ttc_conv in self.ttc_convs:
            ttc_feat = ttc_conv(ttc_feat)

        for offset_conv in self.offset_convs:
            offset_feat = offset_conv(offset_feat)

        cls_score = self.csp_cls(cls_feat)
        bbox_pred = reg_scale(self.csp_reg(reg_feat).float())
        offset_pred = offset_scale(self.csp_offset(offset_feat).float())
        ttc_pred = ttc_scale(self.csp_ttc(ttc_feat)).float()
        return cls_score, bbox_pred, offset_pred, ttc_pred