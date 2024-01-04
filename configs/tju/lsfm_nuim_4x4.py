_base_ = [
    './templates/models/lsfm_base_nuim.py',
    './templates/datasets/nuimage.py',
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
                project="nu_img_2D",
                name="lsfm_4x12",
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


resume_from="./work_dirs/lsfm_nuim_4x4/epoch_73.pth"
