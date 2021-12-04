from mmcv.runner.hooks import HOOKS, Hook, OptimizerHook


@HOOKS.register_module()
class SWAOptimizerHook(OptimizerHook):

    def __init__(self, grad_clip=None):
        super(SWAOptimizerHook, self).__init__(grad_clip=grad_clip)

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
        runner.optimizer.swap_swa_sgd()
        runner.optimizer.bn_update(runner.train_loader, runner.model)
