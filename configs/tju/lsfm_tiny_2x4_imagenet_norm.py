_base_ = [
    './templates/models/lsfm_tiny.py',
    './templates/datasets/tju_traffic.py',
    './templates/optimizers/optimizer_1x.py',
]

evaluation = dict(type="DistEvalHook", interval=1, classwise=True)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="DHD_Traffic_Obj",
                name="lsfm_tiny_2x4_imagenet_norm",
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


resume_from="./work_dirs/lsfm_tiny_2x4_imagenet_norm/epoch_17.pth"
