width=False

model = dict(
    type='CSP',
    val_img_log_prob=0.02,
    pretrained="/home/hkhan/Convolutional-MLPs/output/train/20230127-124135-convmlp_hr_classification-224/model_best.pth.tar",
    backbone=dict(type='DetConvMLPHR'),
    neck=dict(
        type='HRFPN',
        in_channels=[64, 128, 256, 512],
        out_channels=768,
        num_outs=1,
    ),
    bbox_head=dict(
        type='CSPHead',
        num_classes=1,
        in_channels=768,
        stacked_convs=1,
        feat_channels=256,
        strides=[4],
        predict_width=width,
        loss_cls=dict(
            type='CenterLoss',
            loss_weight=0.01),
        loss_bbox=dict(type='RegLoss', loss_weight=0.05, reg_param_count=(2 if width else 1)),
        loss_offset=dict(
            type='OffsetLoss', loss_weight=0.1,
        ),
    ),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
               ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
            pos_fraction=0.25,
            neg_pos_ub=3,
            add_gt_as_proposals=True),
            pos_weight=2,
            debug=True
        ),
        csp_head=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.1,  # 0.2, #0.05,
            nms=dict(type='nms', iou_thr=0.5),
            max_per_img=100,
        ),
    ),
    test_cfg=dict(
        csp_head=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.1, #0.2, #0.05,
            nms=dict(type='nms', iou_thr=0.5),
            max_per_img=100,
        ),
    ),
)
