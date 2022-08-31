import torch
import torch.nn as nn
from inspect import signature
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
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
class CSPHead(AnchorFreeHead):
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
                 wh_ratio=0.41,
                 **kwargs):
        super(CSPHead, self).__init__(
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            **kwargs)

        self.wh_ratio = wh_ratio
        self.predict_width = predict_width
        self.regress_ranges = regress_ranges
        self.loss_offset = build_loss(loss_offset)

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

            self.csp_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
            self.csp_reg = nn.Conv2d(self.feat_channels, 2, 3, padding=1)
            self.csp_offset = nn.Conv2d(self.feat_channels, 2, 3, padding=1)

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
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.csp_cls, std=0.01, bias=bias_cls)
        normal_init(self.csp_reg, std=0.01)
        normal_init(self.csp_offset, std=0.01)

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
                           offset_maps=offset_maps, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward(self, feats, *args):
        return multi_apply(self.forward_single, feats, self.reg_scales, self.offset_scales)

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
        bbox_pred = reg_scale(self.csp_reg(reg_feat).float())
        offset_pred = offset_scale(self.csp_offset(offset_feat).float())
        return cls_score, bbox_pred, offset_pred

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        pass

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'offset_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             offset_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             classification_maps=None,
             scale_maps=None,
             offset_maps=None,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(offset_preds)
        cls_maps = self.concat_batch_gts(classification_maps)
        bbox_gts = self.concat_batch_gts(scale_maps)
        offset_gts = self.concat_batch_gts(offset_maps)

        loss_cls = []
        for cls_score, cls_gt in zip(cls_scores, cls_maps):
            loss_cls.append(self.loss_cls(cls_score, cls_gt))

        loss_cls = loss_cls[0]

        loss_bbox = []
        for bbox_pred, bbox_gt in zip(bbox_preds, bbox_gts):
            loss_bbox.append(self.loss_bbox(bbox_pred, bbox_gt))

        loss_bbox = loss_bbox[0]

        loss_offset = []
        for offset_pred, offset_map in zip(offset_preds, offset_gts):
            loss_offset.append(self.loss_offset(offset_pred, offset_map))

        loss_offset = loss_offset[0]

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_offset=loss_offset)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'offset_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   offset_preds,
                   img_metas,
                   cfg,
                   rescale=None,
                   with_nms=True):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            offset_pred_list = [
                offset_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                offset_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale, with_nms)
            result_list.append(det_bboxes)
        return result_list

    def cspdet2bbox(self, points, scales, offsets, stride=1, max_shape=None):
        """Decode height and offset prediction to bounding box.
            Args:
                points (Tensor): Shape (n, 2), [x, y].
                heights (Tensor): height of the bounding box
                offsets (Tensor): offset of the bounding box center.
                stride: stride of coordinates
                wh_ratio: ratio of width and height, equal to width/height
                max_shape (tuple): Shape of the image.
            Returns:
                Tensor: Decoded bboxes.
            """
        x = points[:, 0] + (offsets[:, 1]) * stride
        y = points[:, 1] + (offsets[:, 0]) * stride

        heights = scales[..., 0] * stride
        if self.predict_width:
            widths = scales[..., 1] * stride
        else:
            widths = heights * self.wh_ratio
        x1 = x - widths / 2
        y1 = y - heights * 0.5
        x2 = x + widths / 2
        y2 = y + heights * 0.5

        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1] - 1)
            y1 = y1.clamp(min=0, max=max_shape[0] - 1)
            x2 = x2.clamp(min=0, max=max_shape[1] - 1)
            y2 = y2.clamp(min=0, max=max_shape[0] - 1)
        return torch.stack([x1, y1, x2, y2], -1)

    def _get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          offset_preds,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False,
                          with_nms=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, offset_pred, points, stride in zip(
                cls_scores, bbox_preds, offset_preds, mlvl_points, self.strides):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # self.show_debug_info(cls_score, bbox_pred, offset_pred, stride)
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 2 if self.predict_width else 1).exp()
            offset_pred = offset_pred.permute(1, 2, 0).reshape(-1, 2)

            nms_pre = self.test_cfg.get('nms_pre', -1)
            if 0 < nms_pre < scores.shape[0]:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                offset_pred = offset_pred[topk_inds, :]
            bboxes = self.cspdet2bbox(points, bbox_pred, offset_pred, stride=stride, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        det_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_labels = torch.cat(mlvl_scores)
        if with_nms:
            padding = det_labels.new_zeros(det_labels.shape[0], 1)
            det_labels = torch.cat([det_labels, padding], dim=1)
            cfg = self.test_cfg
            det_bboxes, det_labels = multiclass_nms(
                det_bboxes,
                det_labels,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img)

        return det_bboxes, det_labels

    def _get_points_single(self, featmap_size, stride, dtype, device, flatten=None):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def concat_batch_gts(self, scale_maps):
        return [scale_maps]

    def aug_test_bboxes(self, feats, img_metas, rescale=False):

        # check with_nms argument
        gb_sig = signature(self.get_bboxes)
        gb_args = [p.name for p in gb_sig.parameters.values()]
        if hasattr(self, '_get_bboxes'):
            gbs_sig = signature(self._get_bboxes)
        else:
            gbs_sig = signature(self._get_bboxes_single)
        gbs_args = [p.name for p in gbs_sig.parameters.values()]
        assert ('with_nms' in gb_args) and ('with_nms' in gbs_args), \
            f'{self.__class__.__name__}' \
            ' does not support test-time augmentation'

        aug_bboxes = []
        aug_scores = []
        aug_factors = []
        for x, img_meta in zip(feats, img_metas):

            outs = self.forward(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            bbox_outputs = self.get_bboxes(*bbox_inputs)[0]
            aug_bboxes.append(bbox_outputs[0])
            aug_scores.append(bbox_outputs[1])

            if len(bbox_outputs) >= 3:
                aug_factors.append(bbox_outputs[2])

        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_factors = torch.cat(aug_factors, dim=0) if aug_factors else None
        padding = merged_scores.new_zeros(merged_scores.shape[0], 1)
        merged_scores = torch.cat([merged_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes,
            merged_scores,
            self.test_cfg.score_thr,
            self.test_cfg.nms,
            self.test_cfg.max_per_img,
            score_factors=merged_factors)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels, self.num_classes + 1)
        return bbox_results
