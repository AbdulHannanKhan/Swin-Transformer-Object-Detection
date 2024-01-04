_base_ = [
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py',
    '../../_base_/models/retinanet_r50_fpn.py'
]

model = dict(
    bbox_head=dict(
        num_classes=5,
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

img_norm_cfg = dict(
#    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
img_scale = (1632, 1216)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', img_scale=img_scale),
    dict(type='RandomFlip', flip_ratio=0.5),
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
    samples_per_gpu=16,
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

# optimizer = dict(_delete_=True, type='Adam', lr=0.0002)
# lr_config = dict(step=[80], policy='step', warmup='constant', warmup_iters=250, warmup_ratio=1.0/3,)
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
evaluation = dict(type="DistEvalHook", interval=1, classwise=True)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="DHD_Traffic_Obj",
                name="retina_4x16_2x",
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


# resume_from="./work_dirs/tju_traffic_4x16_auto/epoch_6.pth"
