# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp

import torch

import mmcv
from mmcv.runner import RUNNERS, EpochBasedRunner
from collections import OrderedDict
from mmcv.runner.hooks import HOOKS


@RUNNERS.register_module()
class MeanTeacherRunner(EpochBasedRunner):

    def __init__(self, *args, mean_teacher=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_dict = {}
        self.mean_teacher = mean_teacher

    def load_mean_teacher_checkpoint(self, cfg):
        if cfg.load_from or cfg.resume_from:
            if cfg.load_from:
                checkpoint = torch.load(cfg.load_from + '.stu')
                self.teacher_dict = checkpoint['state_dict']
                for k in self.model.module.state_dict():
                    if not k in self.teacher_dict:
                        self.teacher_dict[k] = self.model.module.state_dict()[k]
            if cfg.resume_from:
                checkpoint = torch.load(cfg.resume_from + '.stu')
                self.teacher_dict = checkpoint['state_dict']
            for k, v in self.teacher_dict.items():
                self.teacher_dict[k] = self.teacher_dict[k].cuda()
            return

        self.teacher_dict = dict()
        for k, v in self.model.module.state_dict().items():
            self.teacher_dict[k] = v

    def save_mean_teacher_checkpoint(self, state_dict, filename):
        checkpoint = {
            'state_dict': self.weights_to_cpu(state_dict)
        }
        torch.save(checkpoint, filename)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        creat_symlink=True):

        super().save_checkpoint(out_dir, filename_tmpl, save_optimizer, meta, creat_symlink)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)

        if self.mean_teacher:
            mean_teacher_path = filepath + ".stu"
            self.save_mean_teacher_checkpoint(self.teacher_dict, mean_teacher_path)

    def weights_to_cpu(self, state_dict):
        """Copy a model state_dict to cpu.
        Args:
            state_dict (OrderedDict): Model weights on GPU.
        Returns:
            OrderedDict: Model weights on GPU.
        """
        state_dict_cpu = OrderedDict()
        for key, val in state_dict.items():
            state_dict_cpu[key] = val.cpu()
        return state_dict_cpu

    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'MeanTeacherOptimizerHook')
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook, priority='ABOVE_NORMAL')

