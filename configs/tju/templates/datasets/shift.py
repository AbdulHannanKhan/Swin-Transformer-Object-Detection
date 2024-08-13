img_norm_cfg = dict(
#    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (1280, 800)
# img_scale = (640, 400)
train_pipeline = [
    dict(type='LoadImageFromZip', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomBrightness'),
    # dict(type='RemoveSmallBoxes', min_box_size=8, min_gt_box_size=8),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.4, 1.5)),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='RemoveSmallBoxes', min_box_size=8, min_gt_box_size=8),
    dict(type='RandomPave', size=img_scale),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='CSPMaps', radius=8, with_width=True, stride=4, regress_range=(-1, 1e8), image_shape=img_scale, num_classes=6),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'classification_maps',
                               'scale_maps', 'offset_maps',]),
]

test_pipeline = [
    dict(type='LoadImageFromZip', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True, with_mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            # dict(type='Pad', size_divisor=32),
            dict(type='Resize', keep_ratio=True),
            dict(type='Pad', size_divisor=32),
            # dict(type='RemoveSmallBoxes', min_box_size=8, min_gt_box_size=8),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
       ],
    ),
]
classes=["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]
data_root = '/netscratch/hkhan/shift_dataset/'
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=2,
    train=dict(
        type="CocoDataset",
        classes=classes,
        ann_file=data_root + "discrete/images/train/front/det_3d_coco.json",
        img_prefix="",
        pipeline=train_pipeline,
    ),
    val=dict(
        type="CocoDataset",
        classes=classes,
        ann_file=data_root + "discrete/images/val/front/det_3d_coco.json",
        img_prefix="",
        pipeline=test_pipeline,
    ),
    test=dict(
        type="CocoDataset",
        classes=classes,
        ann_file=data_root + "discrete/images/val/front/det_3d_coco.json",
        img_prefix="",
        pipeline=test_pipeline,
    ),
)
