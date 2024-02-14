import argparse
import os
import warnings
import json
import mmcv
import torch
import numpy as np
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test, single_gpu_ttc_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--with-mid', action='store_true', help='calculate MiD loss')
    parser.add_argument('--bttc', action='store_true', help='use accuracy metric for ttc eval mode')
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--ttc-maps-dir', help='save ttc map to directory')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument('--ttc-range', nargs=2, default=(0.5, 1.3), type=float)
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir or args.ttc_maps_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle', '.json')):
        raise ValueError('The output file must be a pkl file.')

    check_range=(args.ttc_range[0] - 1e-6, args.ttc_range[1] + 1e-6)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = args.samples
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', args.samples)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', args.samples) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    # cfg.model.train_cfg = None
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES


    if args.ttc_maps_dir is not None:
        model = MMDataParallel(model, device_ids=[0])
        single_gpu_ttc_test(model, data_loader, out_dir=args.ttc_maps_dir)
        exit()

    classes = None

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        if args.with_mid:
            outputs, mid, classes = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  show_score_thr=args.show_score_thr, error_func="mid", check_range=check_range, ttc_loss=True)
        else:
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  show_score_thr=args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        if args.with_mid:
            outputs, mid, classes = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect, error_func="mid", check_range=check_range, log=False, ttc_loss=True)
        elif args.bttc:
            outputs, mid, classes = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect, error_func="acc", check_range=check_range, log=False, ttc_loss=True)
        else:
            outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect, log=False)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            outs = []
            for im in outputs:
                ann = []
                if len(im) > 0:
                    # remove extra list encapsulation
                    n = im[0]
                    print(n)
                    # select car class bboxes at index 0
                    n = torch.tensor(n[0]).cpu()
                    anns = n.numpy().tolist()
                outs.append(anns)
            print(f'\nwriting results to {args.out}')
            with open(args.out, 'w+') as f:
                json.dump(outs, f)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()

            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'cat_ids'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            if args.with_mid or args.bttc:
                metric = "acc" if args.bttc else "mid"
                if isinstance(mid, list):
                    # mid = np.array(mid).reshape((-1, 2))
                    cls, mid = mid[::2], mid[1::2]
                    cls = np.array(cls)
                    mid = np.array(mid)
                    bins = mid.shape[1]//2
                    bias = mid[:, bins:]
                    mid = mid[:, :bins]
                    for i in range(len(classes)):
                        mid_c = cls == i
                        bias_c = bias[mid_c]
                        mid_c = mid[mid_c]
                        if len(mid_c) > 0:
                            if len(mid_c[0]) == 1:
                                print(f'{classes[i]} :: TTC {metric}: {np.mean(mid_c)} calculated over {np.array(mid_c).shape} points with bias {np.mean(bias)}.')
                            else:
                                for b in range(len(mid_c[0])):
                                    bin_accs = mid_c[:, b].reshape(-1)
                                    bin_bias = bias_c[:, b].reshape(-1)
                                    print(f'{classes[i]} :: TTC {metric} for bin {b}: {np.mean(bin_accs)} calculated over {np.array(bin_accs).shape} points with bias {np.mean(bin_bias)}.')
                                print(f'{classes[i]} :: TTC {metric} all bins: {np.mean(mid_c)} calculated over {np.array(mid_c).shape} points with bias {np.mean(bias_c)}.')
                        else:
                            print(f'{classes[i]} :: TTC {metric}: 0 calculated over 0 points.')
                    print(f'TTC mid: {np.mean(mid)} calculated over {np.array(mid).shape} points with bias {np.mean(bias)}.')
                else:
                    a_seq = lambda _x: 0.2 * (_x + 1)
                    bucket_means = []
                    for k, v in mid.items():
                        if len(v) == 0:
                            print(
                                f'TTC error ({a_seq(k):1.2f}-{a_seq(k + 1):1.2f}): {0 * 100:2.1f} calculated over {np.array(v).shape} points')
                            continue
                        print(f'TTC error ({a_seq(k):1.2f}-{a_seq(k+1):1.2f}): {np.mean(v)*100:2.1f} calculated over {np.array(v).shape} points')
                        bucket_means.append(np.mean(v))
                    print(f'TTC error (0.2s-2s): {np.mean(bucket_means)*100:2.1f} calculated over {np.array(bucket_means).shape} buckets.')
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
