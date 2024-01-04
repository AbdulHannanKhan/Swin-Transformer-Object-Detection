_base_ = [
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py',
    '../../_base_/models/retinanet_r50_fpn.py'
]

model = dict(
    bbox_head=dict(
        num_classes=10,
    ),
    train_cfg=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.5),
        max_per_img=100,
    ),
)


# use caffe img_norm
# img_norm_cfg = dict(
#     mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromZip', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1600, 928), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromZip', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 928),
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

data_root = '/netscratch/hkhan/nuimages-v1.0-all-samples.zip/nuimages-v1.0-all-samples'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type="CocoDataset",
        classes=['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian',
                 'traffic_cone', 'barrier'],
        ann_file="/netscratch/hkhan/nu/nuimages_v1.0-train.json",
        img_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type="CocoDataset",
        classes=['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian',
                 'traffic_cone', 'barrier'],
        ann_file="/netscratch/hkhan/nu/nuimages_v1.0-val.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
    test=dict(
        type="CocoDataset",
        classes=['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian',
                 'traffic_cone', 'barrier'],
        ann_file="/netscratch/hkhan/nu/nuimages_v1.0-val.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
)

#optimizer = dict(
#    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
#optimizer_config = dict(
#    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2), mean_teacher=dict(alpha=0.999))
#lr_config = dict(step=[80], policy='step', warmup='constant', warmup_iters=250, warmup_ratio=1.0/3,)
runner = dict(type='MeanTeacherRunner', max_epochs=24)
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
evaluation = dict(type="DistEvalHook", interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="nu_img_2D",
                entity="hannankhan",
                name="retina_r50",
                config=dict(
                    work_dirs="${work_dir}",
                    total_step="${runner.max_epochs}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)
# resume_from="./work_dirs/nuim_4x8_crcnn_r50c/latest.pth"
