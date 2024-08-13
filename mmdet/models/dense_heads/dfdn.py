import torch
import torch.nn as nn
from inspect import signature
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from mmdet.core import multi_apply, multiclass_nms, bbox2result
from mmcv.runner import force_fp32
from .csp_head import CSPHead, Scale
from ..utils import MixerBlock, window_reverse, window_partition

INF = 1e8

@HEADS.register_module()
class DFDN(CSPHead):
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
                 num_classes=5,
                 patch_dim=4,
                 wh_ratio=0.41,
                 windowed_input=True,
                 predict_width=True,
                 **kwargs):

        self.patch_dim = patch_dim
        self.wh_ratio = wh_ratio
        self.width = None
        self.height = None
        self.predict_width = predict_width
        self.windowed_input = windowed_input

        super(DFDN, self).__init__(
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            **kwargs)

        self.regress_ranges = regress_ranges
        self.loss_offset = build_loss(loss_offset)

    def _init_layers(self):
        self.mlp_with_feat_reduced = nn.Sequential(
            MixerBlock(self.patch_dim ** 2, self.in_channels),
            nn.Linear(self.in_channels, self.feat_channels)
        )

        self.pos_mlp = nn.Sequential(
            MixerBlock(self.patch_dim ** 2, self.feat_channels),
            nn.Linear(self.feat_channels, self.num_classes),
        )

        if self.predict_width:
            self.reg_mlp = nn.Sequential(
                MixerBlock(self.patch_dim ** 2, self.feat_channels),
                nn.Linear(self.feat_channels, 2)  # Predict width and height
            )
        else:
            self.reg_mlp = nn.Sequential(
                MixerBlock(self.patch_dim ** 2, self.feat_channels),
                nn.Linear(self.feat_channels, 1)  # Predict only height
            )

        self.off_mlp = nn.Sequential(
            MixerBlock(self.patch_dim ** 2, self.feat_channels),
            nn.Linear(self.feat_channels, 2)
        )

    def init_weights(self):
        self.reg_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.offset_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward_single(self, x, reg_scale, offset_scale):
        if not self.windowed_input:
            B, _, self.height, self.width = x.shape
            windows = window_partition(x, self.patch_dim, channel_last=False)
        else:
            windows = x
        feat = self.mlp_with_feat_reduced(windows)

        x_cls = self.pos_mlp(feat)
        x_reg = self.reg_mlp(feat)
        x_off = self.off_mlp(feat)

        h = int(self.height)
        w = int(self.width)

        x_cls = window_reverse(x_cls, self.patch_dim, w, h)
        x_reg = window_reverse(x_reg, self.patch_dim, w, h)
        x_off = window_reverse(x_off, self.patch_dim, w, h)

        return x_cls, reg_scale(x_reg).float(), offset_scale(x_off).float()
