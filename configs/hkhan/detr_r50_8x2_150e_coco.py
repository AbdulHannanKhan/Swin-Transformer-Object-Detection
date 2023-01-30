_base_ = [
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]
model = dict(
    type='DETR',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    bbox_head=dict(
        type='TransformerHead',
        num_classes=1,
        in_channels=2048,
        num_fcs=2,
        transformer=dict(
            type='Transformer',
            embed_dims=256,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            feedforward_channels=2048,
            dropout=0.1,
            act_cfg=dict(type='ReLU', inplace=True),
            norm_cfg=dict(type='LN'),
            num_fcs=2,
            pre_norm=False,
            return_intermediate_dec=True),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=1.),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
img_scale = (1280, 640)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion', brightness_delta=180, contrast_range=(0.5, 1.5), 
        saturation_range=(0.5, 1.5), hue_delta=18),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.4, 1.5)),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=img_scale),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        classes=['pedestrain',],
        ann_file="./data/cp/train.json",
        img_prefix="/netscratch/hkhan/cp/leftImg8bit_trainvaltest/",
        # img_prefix="/home/akhan/projects/Pedestron/datasets/CityPersons/leftImg8bit_trainvaltest/",
        pipeline=train_pipeline,
    ),
    val=dict(
        classes=['pedestrain',],
        ann_file="./data/cp/val.json",
        img_prefix="/netscratch/hkhan/cp/leftImg8bit_trainvaltest/leftImg8bit/val/",
        # img_prefix="/home/akhan/projects/Pedestron/datasets/CityPersons/leftImg8bit_trainvaltest/leftImg8bit/val/",
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
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2), mean_teacher=dict(alpha=0.999))
# learning policy
lr_config = dict(policy='step', step=[200])
runner = dict(type='MeanTeacherRunner', max_epochs=300)

evaluation = dict(type="DistEvalHook", interval=2)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="PedDETR",
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
