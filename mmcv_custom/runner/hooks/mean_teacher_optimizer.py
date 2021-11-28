from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class MeanTeacherOptimizerHook(Hook):

    def __init__(self, grad_clip=None, mean_teacher=None):
        self.grad_clip = grad_clip
        self.mean_teacher = mean_teacher

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()

        if self.mean_teacher is not None:
            for k, v in runner.model.module.state_dict().items():
                if k.find('num_batches_tracked') == -1:
                    runner.teacher_dict[k] = self.mean_teacher.alpha * runner.teacher_dict[k] + \
                                             (1 - self.mean_teacher.alpha) * v
                else:
                    runner.teacher_dict[k] = 1 * v
