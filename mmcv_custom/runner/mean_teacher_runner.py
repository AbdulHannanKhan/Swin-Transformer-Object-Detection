# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import torch

import mmcv
from mmcv.runner import RUNNERS, EpochBasedRunner
from collections import OrderedDict
from mmcv.runner.hooks import HOOKS
from mmcv.runner.checkpoint import save_checkpoint


@RUNNERS.register_module()
class MeanTeacherRunner(EpochBasedRunner):

    def __init__(self, *args, mean_teacher=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_dict = {}
        self.mean_teacher = mean_teacher
        self.init_mean_teacher_dict()

    def init_mean_teacher_dict(self):
        self.teacher_dict = dict()
        for k, v in self.model.module.state_dict().items():
            self.teacher_dict[k] = v

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

        self.init_mean_teacher_dict()

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
                        create_symlink=True):

        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

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

