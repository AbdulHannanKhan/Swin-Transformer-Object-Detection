img_norm_cfg = dict(
#    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
# img_scale = (1344, 800)
img_scale = (1632, 1216)
# img_scale = (640, 400)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomBrightness'),
    # dict(type='RemoveSmallBoxes', min_box_size=4, min_gt_box_size=4),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.4, 1.5)),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='RemoveSmallBoxes', min_box_size=8, min_gt_box_size=8),
    dict(type='RandomPave', size=img_scale),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='CSPMaps', radius=8, with_width=True, stride=4, regress_range=(-1, 1e8), image_shape=img_scale, num_classes=5),
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
