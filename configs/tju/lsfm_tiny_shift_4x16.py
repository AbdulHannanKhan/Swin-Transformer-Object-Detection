_base_ = [
    './templates/models/lsfm_tiny.py',
    './templates/datasets/shift.py',
    './templates/optimizers/optimizer_1x.py',
]

data = dict(
    samples_per_gpu=12,
)

model = dict(
    bbox_head = dict(
        num_classes=6,
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
                project="ShiftDet",
                name="lsfm_tiny_4x12_all_res_65",
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


resume_from="./work_dirs/lsfm_tiny_shift_4x16/epoch_65.pth"
