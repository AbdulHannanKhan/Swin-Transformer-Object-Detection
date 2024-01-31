from mmdet.core import bbox2result
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from ...utils.logger import log_image_with_boxes
import numpy as np
import torch
from ..losses import MiDLoss


@DETECTORS.register_module()
class CSP(SingleStageDetector):
    r"""Implementation of `DETR: End-to-End Object Detection with
    Transformers <https://arxiv.org/pdf/2005.12872>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 val_img_log_prob=-1,
                 with_ttc=False):
        super(CSP, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)
        self.bbox_head.test_cfg = test_cfg.csp_head
        if train_cfg is not None:
            self.bbox_head.train_cfg = train_cfg.csp_head
        self.val_img_log_prob = val_img_log_prob
        self.with_ttc = with_ttc
        if hasattr(self.neck, 'backlinks'):
            self.neck.backlinks.append(self)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      classification_maps,
                      scale_maps,
                      offset_maps,
                      gt_bboxes_ignore=None,
                      ttc_maps=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels,
                                              gt_bboxes_ignore, classification_maps=classification_maps,
                                              scale_maps=scale_maps, offset_maps=offset_maps, ttc_maps=ttc_maps)
        return losses

    def exp_test(self, img, cx, cy):

        feat = self.extract_feat(img)[0]
        qttc = self.bbox_head.exp_ttc_bin(feat, cx, cy)

        return qttc


    def simple_test(self, img, img_metas, rescale=False, ttc_maps=None, error_func="mid", gt_labels=None, gt_bboxes=None, check_range=(0.5, 1.3), ttc_out=False, **kwargs):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        batch_size = len(img_metas)
        # assert batch_size == 1, 'Currently only batch_size 1 for inference ' \
        #     f'mode is supported. Found batch_size {batch_size}.'
        x = self.extract_feat(img)
        # print("x shape", img[0].shape)
        outs = self.bbox_head(x, img_metas)
        predicted_MiD = len(outs) == 4
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, self.test_cfg, rescale=rescale)

        if ttc_out:
            return [i[0] for i in outs[3][0]]

        ef = error_func

        def mid_error(pred, gt):
            pred = pred.clamp(min=1e-10)
            return torch.abs(torch.log(gt) - torch.log(pred)) * 1e4

        def ttc_error(pred, gt, _check_range=(0, 10)):
            pred = 0.1/(1 - pred)
            gt = 0.1/(1 - gt)

            if gt < _check_range[0] or gt > _check_range[1]:
                return None

            # pred = pred.clamp(min=_check_range[0], max=_check_range[1])
            # gt = gt.clamp(min=_check_range[0], max=_check_range[1])

            return torch.abs(pred - gt)/gt

        def bin_error(pred, gt, _check_range=(0, 1)):
            if gt < 0.5 or gt > 1.3:
                return None
            pred = 0.1 / (1 - pred)
            gt = 0.1 / (1 - gt)

            if _check_range[0] <= gt:
                pred = (pred < _check_range[-1]) + 0
                gt = (gt < _check_range[-1]) + 0
                return torch.abs(pred - gt)
            else:
                return None

        gt_tti = ttc_maps
        if gt_tti is not None or predicted_MiD:
            tti_pred = outs[3]
            det_ttc = []
            tti_pred = tti_pred[0]
            if tti_pred.shape[1] > 1:
                tti_pred = self.bbox_head.bins2ttc(tti_pred.sigmoid())
            tti_pred = tti_pred[0][0]
            if error_func != "acc":
                tti_pred = tti_pred.clamp(min=1e-10)
            else:
                tti_pred = tti_pred.sigmoid()

        mid_array = []

        if gt_tti is not None:

            # mid = MiDLoss(loss_weight=1e4)
            # mid_array = mid(tti_pred[0], gt_tti[0].permute(0, 2, 3, 1))
            gt_tti = gt_tti[0].permute(0, 2, 3, 1)
            if gt_tti.shape[1] > 2:
                gt_tti = self.bbox_head.bins2ttc(gt_tti[:, 1:, :, :])
            gt_tti = gt_tti[0][0]


            det_ttc = []
            gt_bboxes = gt_bboxes[0][0]

            if gt_labels is not None:
                gt_labels = gt_labels[0][0]

            # create ttc_bins array of 9 empty arrays
            ttc_bins = {}
            for i in range(9):
                ttc_bins[i] = []

            a_seq = lambda _x: 0.2 * (_x + 1)

            for i in range(len(gt_bboxes)):
                gt_bbox = gt_bboxes[i]

                # calculate the center of the gt bbox
                gt_center = (gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2
                gt_center = torch.tensor(gt_center).to(tti_pred.device)

                # rescale the gt center to the tti_pred size
                gt_center = gt_center / self.bbox_head.strides[0]
                gt_center = gt_center.long()

                # calculate the gt_tti value at the gt center
                gt_tti_value = gt_tti[gt_center[1], gt_center[0]]

                # calculate the tti_pred value at the gt center
                pred_tti_value = tti_pred[gt_center[1], gt_center[0]]

                eval = None

                if ef == "mid":
                    if check_range[0] <= gt_tti_value <= check_range[1]:
                        error = mid_error(pred_tti_value, gt_tti_value)
                        if error is not None:
                            eval = error.item()
                elif ef != "acc":
                    for _b in range(9):
                        error = ttc_error(pred_tti_value, gt_tti_value, _check_range=(a_seq(_b), a_seq(_b + 1)))
                        if error is not None:
                            ttc_bins[_b].append(error.item())
                    error = ttc_error(pred_tti_value, gt_tti_value, check_range)
                    if error is not None:
                        mid_array.append(error.item())
                else:
                    pred_tti_value = pred_tti_value.item() > 0.5
                    gt_tti_value = gt_tti_value.item() > 0.5
                    eval = pred_tti_value.item() == gt_tti_value.item()
                if eval is not None:
                    if gt_labels is None:
                        mid_array.append(eval)
                    else:
                        cls = gt_labels[i].item()
                        mid_array.append(cls)
                        mid_array.append(eval)

            # get det_bbox from bbox_list
            det_bboxes = bbox_list[0][0]

            # mid_array = ttc_bins

            for i in range(len(det_bboxes)):
                bbox = det_bboxes[i]

                # calculate the center of the det bbox
                det_center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                det_center = torch.tensor(det_center).to(tti_pred.device)

                # rescale the det center to the tti_pred size
                det_center = det_center / self.bbox_head.strides[0]
                det_center = det_center.long()

                # calculate the tti_pred value at the det center
                pred_tti_value = tti_pred[det_center[1], det_center[0]]

                det_ttc.append(pred_tti_value.item())

            det_ttc = torch.tensor(det_ttc).to(tti_pred.device)
            # reshape to (n, 1)
            det_ttc = det_ttc.view(-1, 1)

            bbox_results = [
                bbox2result(torch.cat([det_bboxes, det_ttc], 1), det_labels, self.bbox_head.num_classes + 1,
                            box_dim=6)
                for det_bboxes, det_labels in bbox_list
            ]

        elif predicted_MiD:
            det_bboxes = bbox_list[0][0]

            # mid_array = ttc_bins

            for i in range(len(det_bboxes)):
                bbox = det_bboxes[i]

                # calculate the center of the det bbox
                det_center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                det_center = torch.tensor(det_center).to(tti_pred.device)

                # rescale the det center to the tti_pred size
                det_center = det_center / self.bbox_head.strides[0]
                det_center = det_center.long()

                # calculate the tti_pred value at the det center
                pred_tti_value = tti_pred[det_center[1], det_center[0]]

                det_ttc.append(pred_tti_value.item())

            det_ttc = torch.tensor(det_ttc).to(tti_pred.device)
            # reshape to (n, 1)
            det_ttc = det_ttc.view(-1, 1)

            bbox_results = [
                bbox2result(torch.cat([det_bboxes, det_ttc], 1), det_labels, self.bbox_head.num_classes + 1,
                            box_dim=6)
                for det_bboxes, det_labels in bbox_list
            ]

        else:
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes + 1)
                for det_bboxes, det_labels in bbox_list
            ]

        if gt_tti is not None:
            return bbox_results, mid_array

        return bbox_results
