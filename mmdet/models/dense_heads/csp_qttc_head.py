import torch
import torch.nn as nn
from inspect import signature
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from ..builder import HEADS, build_loss
from .csp_ttc_head import CSPTTCHead
from mmdet.core import multi_apply, multiclass_nms, bbox2result
from mmcv.runner import force_fp32

INF = 1e8


@HEADS.register_module()
class CSPQTTCHead(CSPTTCHead):
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
                 ttc_bins=8,
                 bin_bias=0.5,
                 bin_weights=None,
                 loss_ttc=dict(type='QTTCLoss', loss_weight=0.1),
                 **kwargs):

        super(CSPQTTCHead, self).__init__(
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_ttc=loss_ttc,
            wh_ratio=wh_ratio,
            predict_width=predict_width,
            regress_ranges=regress_ranges,
            loss_offset=loss_offset,
            **kwargs)

        self.ttc_bins = ttc_bins
        self.bin_bias = bin_bias
        if bin_weights is None:
            self.bin_weights = torch.ones(ttc_bins, dtype=torch.float32) * 0.1
        else:
            assert len(bin_weights) == ttc_bins
            self.bin_weights = bin_weights

    def _init_layers(self):
        """Initialize layers of the head."""
        super(CSPQTTCHead, self)._init_layers()

        self.csp_ttc = nn.Conv2d(self.feat_channels, self.ttc_bins, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        super(CSPTTCHead, self).init_weights()
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
        return multi_apply(self.forward_single, feats, self.reg_scales, self.offset_scales)

    def forward_single(self, x, reg_scale, offset_scale):
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
        ttc_pred = self.csp_ttc(ttc_feat).float()
        return cls_score, bbox_pred, offset_pred, ttc_pred

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        pass

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'offset_preds', 'ttc_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             offset_preds,
             ttc_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             classification_maps=None,
             scale_maps=None,
             offset_maps=None,
             gt_bboxes_ignore=None,
             ttc_maps=None):
        assert len(cls_scores) == len(bbox_preds) == len(offset_preds)
        cls_maps = self.concat_batch_gts(classification_maps)
        bbox_gts = self.concat_batch_gts(scale_maps)
        ttc_maps = self.concat_batch_gts(ttc_maps)
        offset_gts = self.concat_batch_gts(offset_maps)

        loss_cls = []
        loss_tv = []
        for cls_score, cls_gt in zip(cls_scores, cls_maps):
            loss_cls.append(self.loss_cls(cls_score, cls_gt))

        loss_cls = loss_cls[0]

        loss_bbox = []
        for bbox_pred, bbox_gt in zip(bbox_preds, bbox_gts):
            loss_bbox.append(self.loss_bbox(bbox_pred, bbox_gt))

        loss_bbox = loss_bbox[0]
        loss_ttc = []
        mid = 0
        for ttc_pred, ttc_gt in zip(ttc_preds, ttc_maps):
            ttc = self.loss_ttc(ttc_pred, ttc_gt)
            mid = self.mid_loss(ttc_pred, ttc_gt)
            if isinstance(ttc, tuple):
                tv = ttc[0]
                loss_tv.append(tv)
                ttc = ttc[1]
            loss_ttc.append(ttc)

        loss_ttc = loss_ttc[0]

        loss_offset = []
        for offset_pred, offset_map in zip(offset_preds, offset_gts):
            loss_offset.append(self.loss_offset(offset_pred, offset_map))

        loss_offset = loss_offset[0]

        if len(loss_tv) > 0:
            loss_tv = loss_tv[0]
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_offset=loss_offset,
                loss_L1ttc=loss_ttc.mean(),
                loss_tv=loss_tv.mean(),
            )

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_offset=loss_offset,
            loss_qttc=loss_ttc,
            MiD=mid * 1e4,
        )

    def mid_loss(self, ttc_preds, ttc_maps):

        # first channel of ttc_maps is the mask
        mask = ttc_maps[:, 0:1]
        ttc_maps = ttc_maps[:, 1:]

        mask = mask.reshape(-1)

        maps = self.bins2ttc(ttc_maps)
        maps = maps.reshape(-1)
        maps = maps[mask == 1]

        preds = self.bins2ttc(ttc_preds)
        preds = preds.reshape(-1)
        preds = preds[mask == 1]

        loss = torch.abs(torch.log(preds) - torch.log(maps)).mean()

        return loss

    def bins2ttc(self, bins):

        # (B, Bins, H, W) --> (B, 1, H, W)

        bins = bins.permute(0, 2, 3, 1)
        ttc = torch.sum(bins * self.bin_weights, dim=-1, keepdim=True) + self.bin_bias
        ttc = ttc.permute(0, 3, 1, 2)

        return ttc

    def _get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          offset_preds,
                          ttc_preds,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False,
                          with_nms=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points) == len(ttc_preds)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_ttcs = []
        for cls_score, bbox_pred, offset_pred, ttc_pred, points, stride in zip(
                cls_scores, bbox_preds, offset_preds, ttc_preds, mlvl_points, self.strides):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # self.show_debug_info(cls_score, bbox_pred, offset_pred, stride)
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 2 if self.predict_width else 1).exp()
            ttc_pred = ttc_pred[None, :, :, :]
            ttc_pred = self.bins2ttc(ttc_pred)[0]
            ttc_pred = ttc_pred.permute(1, 2, 0).reshape(-1, 1)
            offset_pred = offset_pred.permute(1, 2, 0).reshape(-1, 2)

            nms_pre = self.test_cfg.get('nms_pre', -1)
            if 0 < nms_pre < scores.shape[0]:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                ttc_pred = ttc_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                offset_pred = offset_pred[topk_inds, :]
            bboxes = self.cspdet2bbox(points, bbox_pred, offset_pred, stride=stride, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_ttcs.append(ttc_pred)
        det_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_labels = torch.cat(mlvl_scores)
        det_ttcs = torch.cat(mlvl_ttcs)
        if with_nms:
            padding = det_labels.new_zeros(det_labels.shape[0], 1)
            det_labels = torch.cat([det_labels, padding], dim=1)
            # det_ttcs = torch.cat([det_ttcs, padding], dim=1)
            cfg = self.test_cfg
            _det_bboxes, _det_labels, keep = multiclass_nms(
                det_bboxes,
                det_labels,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                return_inds=True)
            # if keep.max() >= det_ttcs.shape[0]-1:
            #    keep[keep >= det_ttcs.shape[0]-1] = 0  # TODO: fix the hack
            # det_ttcs = det_ttcs[keep]
            # det_ttcs = det_ttcs.new_zeros(_det_labels.shape[0], 1)

        return _det_bboxes, _det_labels

    def concat_batch_gts(self, scale_maps):
        return [scale_maps]
