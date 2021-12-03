import torch
import torch.nn as nn
from inspect import signature
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from ..builder import HEADS, build_loss
from .csp_head import CSPHead
from mmdet.core import multi_apply, multiclass_nms, bbox2result
from mmcv.runner import force_fp32
from ..backbones.swin_transformer import SwinTransformer

INF = 1e8


class Scale(nn.Module):

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


@HEADS.register_module()
class CSPTransHead(CSPHead):
    def __init__(self,
                 *args,
                 t_heads=[2],
                 t_depths=[1],
                 t_patch_size=3,
                 **kwargs):
        super(CSPTransHead, self).__init__(
            *args,
            **kwargs)
        self.t_heads = t_heads
        self.t_depths = t_depths
        self.t_patch_size = t_patch_size

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.offset_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.offset_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

            # self.csp_cls = nn.Conv2d( self.feat_channels, self.cls_out_channels, 3, padding=1)
            # self.csp_reg = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
            # self.csp_offset = nn.Conv2d(self.feat_channels, 2, 3, padding=1)
            self.csp_cls = SwinTransformer(patch_size=self.t_patch_size, in_chans=self.feat_channels, embed_dim=1,
                                           depths=self.t_depths, num_heads=self.t_heads, out_indices=(0,))
            self.csp_reg = SwinTransformer(patch_size=self.t_patch_size, in_chans=self.feat_channels, embed_dim=1,
                                           depths=self.t_depths, num_heads=self.t_heads, out_indices=(0,))

            self.csp_offset = SwinTransformer(patch_size=self.t_patch_size, in_chans=self.feat_channels, embed_dim=2,
                                           depths=self.t_depths, num_heads=self.t_heads, out_indices=(0,))
            self.reg_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
            self.offset_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.offset_convs:
            normal_init(m.conv, std=0.01)
        # bias_cls = bias_init_with_prob(0.01)
        # normal_init(self.csp_cls, std=0.01, bias=bias_cls)
        # normal_init(self.csp_reg, std=0.01)
        # normal_init(self.csp_offset, std=0.01)

    def forward_single(self, x, reg_scale, offset_scale):
        cls_feat = x
        reg_feat = x
        offset_feat = x

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)

        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        for offset_conv in self.offset_convs:
            offset_feat = offset_conv(offset_feat)

        cls_score = self.csp_cls(cls_feat)
        bbox_pred = reg_scale(self.csp_reg(reg_feat)[0].float())
        offset_pred = offset_scale(self.csp_offset(offset_feat)[0].float())
        return cls_score, bbox_pred, offset_pred
