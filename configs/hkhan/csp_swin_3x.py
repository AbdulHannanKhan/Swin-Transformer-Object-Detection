_base_ = [
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='CSP',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,
    ),
    neck=dict(
        type='HRFPN',
        in_channels=[96, 192, 384, 768],
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
        loss_cls=dict(
            type='CenterLoss',
            loss_weight=0.01),
        loss_bbox=dict(type='RegLoss', loss_weight=0.05),
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
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
img_scale = (2048, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion', brightness_delta=180, contrast_range=(0.5, 1.5), 
        saturation_range=(0.5, 1.5), hue_delta=18),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.4, 1.5)),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=img_scale),
    dict(type='CSPMaps', radius=8, stride=4, regress_range=(-1, 1e8), image_shape=img_scale),
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

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        classes=['pedestrain',],
        ann_file="./data/cp/train_trunk.json",
        img_prefix="/netscratch/hkhan/cp/leftImg8bit_trainvaltest/",
        # img_prefix="/home/akhan/projects/Pedestron/datasets/CityPersons/leftImg8bit_trainvaltest/",
        pipeline=train_pipeline,
    ),
    val=dict(
        classes=['pedestrain',],
        ann_file="./data/cp/val.json",
        #img_prefix="/home/akhan/projects/Pedestron/datasets/CityPersons/leftImg8bit_trainvaltest/leftImg8bit/val/",
        img_prefix="/netscratch/hkhan/cp/leftImg8bit_trainvaltest/leftImg8bit/val/",
        pipeline=test_pipeline,
    ),
    test=dict(
        classes=['pedestrain',],
        ann_file="./data/cp/val.json",
        #img_prefix="/home/akhan/projects/Pedestron/datasets/CityPersons/leftImg8bit_trainvaltest/leftImg8bit/val/",
        img_prefix="/netscratch/hkhan/cp/leftImg8bit_trainvaltest/leftImg8bit/val/",
        pipeline=test_pipeline,
    ),
)

optimizer = dict(_delete_=True, type='Adam', lr=0.0002)
lr_config = dict(step=[85, 105], policy='step', warmup='constant', warmup_iters=250, warmup_ratio=1.0/3, gamma=(0.1)**0.5)
runner = dict(type='MeanTeacherRunner', max_epochs=120)
optimizer_config=dict(mean_teacher=dict(alpha=0.999))
fp16 = None
evaluation = dict(type="DistEvalHook", interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="SwinCSP",
                name="Adv",
                config=dict(
                    work_dirs="${work_dir}",
                    total_step="${runner.max_epochs}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)
