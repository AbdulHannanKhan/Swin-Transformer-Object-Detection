_base_ = [
    './templates/models/lsfm_tiny.py',
    './templates/datasets/shift.py',
    './templates/optimizers/optimizer_1x.py',
]

model = dict(
    bbox_head=dict(
        num_classes=2,
    )
)

evaluation = dict(type="DistEvalHook", interval=1, classwise=True)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="Shift",
                name="lsfm_4x6",
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