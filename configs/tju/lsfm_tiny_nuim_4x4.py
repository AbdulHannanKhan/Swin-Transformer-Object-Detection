_base_ = [
    './templates/models/lsfm_tiny_nuim.py',
    './templates/datasets/nuimage.py',
    './templates/optimizers/optimizer_1x.py',
]

data = dict(
    samples_per_gpu=4,
)

evaluation = dict(type="DistEvalHook", interval=1, classwise=True)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="nu_img_2D",
                name="lsfm_tiny_4x4_r38",
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


resume_from="./work_dirs/lsfm_tiny_nuim_4x4/epoch_38.pth"
