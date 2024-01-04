_base = [
 '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

optimizer = dict(_delete_=True, type='Adam', lr=0.0002)
lr_config = dict(step=[80], policy='step', warmup='constant', warmup_iters=250, warmup_ratio=1.0/3,)
runner = dict(type='MeanTeacherRunner', max_epochs=120)
optimizer_config=dict(mean_teacher=dict(alpha=0.999))
# do not use mmdet version fp16
fp16 = None

evaluation = dict(type="DistEvalHook", interval=1)