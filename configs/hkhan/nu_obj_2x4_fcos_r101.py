_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

width=True

model = dict(
    type='FCOS',
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(
        type='ResNet',
        depth=101,
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
       num_classes=10,
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
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
img_scale = (1600, 900)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomBrightness'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.4, 1.5)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
        classes=['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian',
                 'traffic_cone', 'barrier'],
        ann_file="/netscratch/hkhan/nu/nu_infos_train_mono3d.coco.json",
        img_prefix="/netscratch/hkhan/nuscenes/raw/",
        pipeline=train_pipeline,
    ),
    val=dict(
        type="CocoDataset",
        classes=['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian',
                 'traffic_cone', 'barrier'],
        ann_file="/netscratch/hkhan/nu/nu_infos_val_mono3d.coco.json",
        img_prefix="/netscratch/hkhan/nuscenes/raw/",
        pipeline=test_pipeline,
    ),
    test=dict(
        type="CocoDataset",
        classes=['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian',
                 'traffic_cone', 'barrier'],
        ann_file="/netscratch/hkhan/nu/nu_infos_val_mono3d.coco.json",
        img_prefix="/netscratch/hkhan/nuscenes/raw/",
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
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=32, norm_type=2))
fp16 = None # dict(loss_scale=16.)
evaluation=dict(classwise=True, metric='bbox')
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="NU_Obj",
                name="fcos_r101",
                config=dict(
                    work_dirs="${work_dir}",
                    total_step="${runner.max_epochs}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)


resume_from="./work_dirs/nu_obj_2x4_fcos_r101/epoch_2.pth"
