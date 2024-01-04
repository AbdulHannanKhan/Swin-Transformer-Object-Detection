_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

width=True

model = dict(
    type='CSP',
    val_img_log_prob=0.01,
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
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
                fuse_till=3,
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))
        ),
        #frozen_stages=-1,
        norm_eval=False,
    ),
    neck=dict(
        type='BP3',
        in_channels=[32, 64, 128, 256],
        out_channels=32,
        mixer_count=1,
        linear_reduction=False,
        feat_channels=[4, 16, 128]
    ),
    bbox_head=dict(
        type='DFDN',
        num_classes=5,
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

img_norm_cfg = dict(
#    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
img_scale = (1632, 1216)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomBrightness'),
    # dict(type='RemoveSmallBoxes', min_box_size=1, min_gt_box_size=8),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.4, 1.5)),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='RemoveSmallBoxes', min_box_size=8, min_gt_box_size=8),
    dict(type='RandomPave', size=img_scale),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='CSPMaps', radius=8, with_width=width, stride=4, regress_range=(-1, 1e8), image_shape=img_scale, num_classes=5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'classification_maps',
                               'scale_maps', 'offset_maps']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]
data_root = '/netscratch/hkhan/tju/dhd_traffic'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type="CocoDataset",
        classes=['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Van'],
        ann_file=data_root+'/annotations/dhd_traffic_train.json',
        img_prefix=data_root+'/images/train/',
        pipeline=train_pipeline,
    ),
    val=dict(
        type="CocoDataset",
        classes=['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Van'],
        ann_file=data_root+'/annotations/dhd_traffic_val.json',
        img_prefix=data_root+'/images/val/',
        pipeline=test_pipeline,
    ),
    test=dict(
        type="CocoDataset",
        classes=['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Van'],
        ann_file=data_root+'/annotations/dhd_traffic_val.json',
        img_prefix=data_root+'/images/val/',
        pipeline=test_pipeline,
    ),
)

optimizer = dict(_delete_=True, type='Adam', lr=0.0002)
lr_config = dict(step=[80], policy='step', warmup='constant', warmup_iters=250, warmup_ratio=1.0/3,)
runner = dict(type='MeanTeacherRunner', max_epochs=120)
optimizer_config=dict(mean_teacher=dict(alpha=0.999))
# do not use mmdet version fp16
fp16 = None
# optimizer_config = dict(
#    type="DistOptimizerHook",
#    update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )
evaluation = dict(type="DistEvalHook", interval=1, classwise=True)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="DHD_Traffic_Obj",
                name="auto",
                entity="hannankhan",
                config=dict(
                    work_dirs="${work_dir}",
                    total_step="${runner.max_epochs}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)


resume_from="./work_dirs/tju_traffic_4x16_auto/epoch_6.pth"
