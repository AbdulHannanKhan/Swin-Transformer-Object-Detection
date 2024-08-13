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

# img_norm_cfg = dict(
#     mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(1280, 720), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1280, 720),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
classes=['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']
data_root = '/netscratch/hkhan/BDD100K/bdd100k/'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type="CocoDataset",
        classes=classes,
        ann_file=data_root + "labels/det_20/det_train_coco.json",
        img_prefix=data_root+"images/100k/train/",
        pipeline=train_pipeline,
    ),
    val=dict(
        type="CocoDataset",
        classes=classes,
        ann_file=data_root + "labels/det_20/det_val_coco.json",
        img_prefix=data_root+"images/100k/val/",
        pipeline=test_pipeline,
    ),
    test=dict(
        type="CocoDataset",
        classes=classes,
        ann_file=data_root + "labels/det_20/det_val_coco.json",
        img_prefix=data_root+"images/100k/val/",
        pipeline=test_pipeline,
    ),
)

# optimizer = dict(_delete_=True, type='Adam', lr=0.0002)
# lr_config = dict(step=[80], policy='step', warmup='constant', warmup_iters=500, warmup_ratio=0.1,)
runner = dict(type='MeanTeacherRunner', max_epochs=24)
optimizer_config=dict(mean_teacher=dict(alpha=0.999))
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
                entity="hannankhan",
                project="BDD100KDet",
                name="retina_4x16",
                config=dict(
                    work_dirs="${work_dir}",
                    total_step="${runner.max_epochs}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)
