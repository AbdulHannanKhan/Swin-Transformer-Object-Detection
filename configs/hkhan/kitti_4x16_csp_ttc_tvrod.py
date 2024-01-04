_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

width=True
num_classes=2

model = dict(
    type='CSP',
    with_ttc=True,
    val_img_log_prob=0.001,
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
           stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))
        ),
        #frozen_stages=-1,
        norm_eval=False,
    ),
    neck=dict(
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=768,
        num_outs=1,
    ),
    bbox_head=dict(
        type='CSPTTCHead',
        num_classes=num_classes,
        in_channels=768,
        stacked_convs=1,
        feat_channels=256,
        strides=[4],
        drop_ttc=0.2,
        bn_for_ttc=False,
        loss_cls=dict(
            type='CenterLoss',
            loss_weight=0.01),
        loss_bbox=dict(type='RegLoss', loss_weight=0.05),
        loss_offset=dict(
            type='OffsetLoss', loss_weight=0.1,
        ),
        loss_ttc=dict(type='TVRoDLoss', tv_weight=2e-6, rod_weight=0.1),
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
            min_bbox_size=15,
            score_thr=0.01, #0.2, #0.05,
            nms=dict(type='nms', iou_thr=0.7),
            max_per_img=100,
        ),
    ),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
img_scale = (384, 1248)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_ttc=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomBrightness'),
    # dict(type='RemoveSmallBoxes', min_box_size=8, min_gt_box_size=8),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.4, 1.5)),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='RemoveSmallBoxes', min_box_size=8, min_gt_box_size=8),
    dict(type='RandomPave', size=img_scale),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='CSPMaps', radius=2, with_width=width, stride=4, regress_range=(-1, 1e8), image_shape=img_scale, bbox_ttc=True, with_ttc=True, num_classes=num_classes),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'classification_maps',
                               'scale_maps', 'offset_maps', 'ttc_maps']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_ttc=True, with_mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Pad', size_divisor=32),
            dict(type='Resize', keep_ratio=True),
            dict(type='RemoveSmallBoxes', min_box_size=8, min_gt_box_size=8),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='CSPMaps', radius=2, with_width=width, stride=4, regress_range=(-1, 1e8), bbox_ttc=True, with_ttc=True, num_classes=num_classes),
            dict(type='ImageToTensor', keys=['img', 'ttc_maps']),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'classification_maps', 'scale_maps', 'offset_maps', 'ttc_maps']),
       ],
    ),
]
classes=['Car', 'Pedestrian']
data_root = '/ds-av/public_datasets/kitti_tracking/raw/training/image_02/'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type="CocoTTCDataset",
        classes=classes,
        ann_file="/netscratch/hkhan/kitti/train_ign.json",
        img_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type="CocoTTCDataset",
        classes=classes,
        ann_file="/netscratch/hkhan/kitti/val.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
    test=dict(
        type="CocoTTCDataset",
        classes=classes,
        ann_file="/netscratch/hkhan/kitti/val_mod.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
)

optimizer = dict(_delete_=True, type='Adam', lr=0.0002)
lr_config = dict(step=[80], policy='step', warmup='constant', warmup_iters=50, warmup_ratio=1.0/3,)
runner = dict(type='MeanTeacherRunner', max_epochs=120)
optimizer_config=dict(mean_teacher=dict(alpha=0.999))
# do not use mmdet version fp16
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=32, norm_type=2))
fp16 = None # dict(loss_scale=16.)

evaluation=dict(classwise=True, metric='bbox', with_ttc=True)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                entity="hannankhan",
                project="KittiTTC",
                name="csp_tvrod_car_ped_4x16_r2_70_d2_nus",
                config=dict(
                    work_dirs="${work_dir}",
                    total_step="${runner.max_epochs}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)


load_from="./work_dirs/nu_obj_4x8_csp_ttc_sel_front/epoch_25.pth"
# resume_from="./work_dirs/kitti_2x12_csp_ttc/epoch_60.pth"
