width=True

model = dict(
    val_img_log_prob=0.01,
    type='CSP',
    pretrained="/home/hkhan/Convolutional-MLPs/output/train/20230716-195427-convmlp_hr_classification-224/model_best.pth.tar",
    backbone=dict(type="DetConvMLPHR"),
    neck=dict(
        type='BP3',
        in_channels=[64, 128, 256, 512],
        out_channels=32,
        mixer_count=1,
        linear_reduction=False,
        feat_channels=[4, 16, 128, 1024]
    ),
    bbox_head=dict(
        type='DFDN',
        num_classes=10,
        in_channels=32,
        stacked_convs=1,
        patch_dim=8,
        feat_channels=32,
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