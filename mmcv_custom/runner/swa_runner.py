# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import torch

import mmcv
from mmcv.runner import RUNNERS, EpochBasedRunner
from mmcv.runner.hooks import HOOKS
from torchcontrib.optim import SWA


@RUNNERS.register_module()
class SWARunner(EpochBasedRunner):

    def __init__(self, *args, swa_start=10, swa_freq=1, swa_lr=0.0005, **kwargs):
        super(SWARunner, self).__init__(*args, **kwargs)
        self.current_dl = None
        self.optimizer = SWA(self.optimizer, swa_start=swa_start, swa_freq=swa_freq, swa_lr=swa_lr)

    def train(self, data_loader, **kwargs):
        self.current_dl = data_loader
        super(SWARunner, self).train(data_loader, **kwargs)

    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'SWAOptimizerHook')
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook, priority='ABOVE_NORMAL')

