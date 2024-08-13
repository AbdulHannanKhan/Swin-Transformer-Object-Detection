_base_ = [
    "./base_schedule.py", "./base_convhr_f2dnet.py", "./base_cp.py"
]

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                entity="mlpthesis",
                project="MLPOD",
                name="h2x4_3focus_neck_1_head_32c_mixup_ecp_cp",
                config=dict(
                    work_dirs="${work_dir}",
                    total_step="${runner.max_epochs}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)


load_from='/netscratch/hkhan/work_dirs/mlpod/ecp/NH_HR_mixup/epoch_68.pth'

