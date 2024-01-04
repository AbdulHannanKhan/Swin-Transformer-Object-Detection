_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='FCOS',
    pretrained='open-mmlab://detectron/resnet50_caffe',
    backbone=dict(
       type='ResNet',
       depth=50,
       num_stages=4,
       out_indices=(0, 1, 2, 3),
       frozen_stages=1,
       norm_cfg=dict(type='BN', requires_grad=False),
       norm_eval=True,
       style='caffe'),
    neck=dict(
       type='FPN',
       in_channels=[256, 512, 1024, 2048],
       out_channels=256,
       start_level=1,
       add_extra_convs=True,
       extra_convs_on_inputs=False,  # use P5
       num_outs=5,
       relu_before_extra_convs=True),
    bbox_head=dict(
       type='FCOSHead',
       num_classes=6,
       in_channels=256,
       stacked_convs=4,
       feat_channels=256,
       strides=[8, 16, 32, 64, 128],
       loss_cls=dict(
           type='FocalLoss',
           use_sigmoid=True,
           gamma=2.0,
           alpha=0.25,
           loss_weight=1.0),
       loss_bbox=dict(type='IoULoss', loss_weight=1.0),
       loss_centerness=dict(
           type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    ),
    train_cfg=dict(
       assigner=dict(
           type='MaxIoUAssigner',
           pos_iou_thr=0.5,
           neg_iou_thr=0.4,
           min_pos_iou=0,
           ignore_iof_thr=-1),
       allowed_border=-1,
       pos_weight=-1,
       debug=False),
    test_cfg=dict(
       nms_pre=1000,
       min_bbox_size=0,
       score_thr=0.05,
       nms=dict(type='nms', iou_threshold=0.5),
       max_per_img=100),
)

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1280, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data_root = '/netscratch/hkhan/shift_dataset/'
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=2,
    train=dict(
        type="ShiftDataset",
        data_root=data_root,
        ann_file=data_root + "discrete/images/train/front/det_2d.json",
        img_prefix=data_root + "discrete/images/train/front/img.zip",
        backend_type="zip",
        pipeline=train_pipeline,
    ),
    val=dict(
        type="ShiftDataset",
        data_root=data_root,
        ann_file=data_root + "discrete/images/val/front/det_2d.json",
        img_prefix=data_root + "discrete/images/val/front/img.zip",
        backend_type="zip",
        pipeline=test_pipeline,
    ),
    test=dict(
        type="ShiftDataset",
        data_root=data_root,
        ann_file=data_root + "discrete/images/val/front/det_2d.json",
        img_prefix=data_root + "discrete/images/val/front/img.zip",
        backend_type="zip",
        pipeline=test_pipeline,
    ),
)

optimizer = dict(
    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2), mean_teacher=dict(alpha=0.999))
lr_config = dict(step=[80], policy='step', warmup='constant', warmup_iters=250, warmup_ratio=1.0/3,)
runner = dict(type='MeanTeacherRunner', max_epochs=120)
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
evaluation = dict(type="DistEvalHook", interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="ShiftDet",
                entity="hannankhan",
                name="fcos10+",
                config=dict(
                    work_dirs="${work_dir}",
                    total_step="${runner.max_epochs}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)


# resume_from="./work_dirs/shift_4x16/epoch_10.pth"
