img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
img_scale = (1920, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RemoveSmallBoxes', min_box_size=1, min_gt_box_size=4),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.4, 1.5)),
    dict(type='RemoveSmallBoxes', min_box_size=4, min_gt_box_size=8),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='RandomBrightness'),
    dict(type='RandomPave', size=img_scale),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='CSPMaps', radius=8, with_width=False, stride=4, regress_range=(-1, 1e8), image_shape=img_scale),
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
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]
data_root = '/netscratch/hkhan/ECP/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type="CocoDataset",
        classes=['pedestrian',],
        ann_file='./datasets/EuroCity/day_train_all_area.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type="CocoDataset",
        classes=['pedestrian',],
        ann_file='./datasets/EuroCity/day_val_visT.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
    test=dict(
        type="CocoDataset",
        classes=['pedestrian',],
        ann_file='./datasets/EuroCity/day_val_visT.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
)