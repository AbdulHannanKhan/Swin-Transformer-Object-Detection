import os.path as osp
import pickle
import shutil
import tempfile
import time
import os
import cv2

import numpy as np
import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from ..utils.logger import log_image_with_boxes
from mmdet.core import encode_mask_results


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    ttc_loss=False,
                    error_func="mid",
                    check_range=(0.5, 1.3),
                    show_score_thr=0.3):
    model.eval()
    results = []
    ttc_losses = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    delta = 0
    count = 0

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            st = time.time()
            if ttc_loss:
                result, mid = model(return_loss=False, rescale=True, error_func=error_func, check_range=check_range, **data)
            else:
                result = model(return_loss=False, rescale=True, **data)
            delta += time.time() - st
            count += 1
            if ttc_loss:
                ttc_losses.append(mid)
        if i == 0:
            print(data["img"][0].data[0].shape)

        if count % 500 == 0:
            print("\nImg/sec: ", int(count / delta * 1000) / 1000)
        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    print("\nImg/sec: ", int(count / delta * 1000) / 1000)
    if ttc_loss:
        class_names = dataset.CLASSES
        ttc_res = []
        ttc_dict = None
        if ttc_losses is not None:
            for res in ttc_losses:
                # check if res is list
                if isinstance(res, list):
                    ttc_res.extend(res)
                elif isinstance(res, dict):
                    if ttc_dict is None:
                        ttc_dict = res
                    else:
                        for k, v in res.items():
                            ttc_dict[k].extend(v)
                else:
                    ttc_res.append(res)
        if ttc_dict is not None:
            for k, v in ttc_dict.items():
                ttc_dict[k] = np.array(v)
            ttc_res = ttc_dict
        return results, ttc_res, class_names
    return results


def disp2rgb(disp):
    H = disp.shape[0]
    W = disp.shape[1]

    I = disp.flatten()

    map = np.array([[0, 0, 0, 114], [0, 0, 1, 185], [1, 0, 0, 114], [1, 0, 1, 174],
                    [0, 1, 0, 114], [0, 1, 1, 185], [1, 1, 0, 114], [1, 1, 1, 0]])
    bins = map[:-1,3]
    cbins = np.cumsum(bins)
    bins = bins/cbins[-1]
    cbins = cbins[:-1]/cbins[-1]

    ind = np.minimum(np.sum(np.repeat(I[None, :], 6, axis=0) > np.repeat(cbins[:, None],
                                    I.shape[0], axis=1), axis=0), 6)
    bins = np.reciprocal(bins)
    cbins = np.append(np.array([[0]]), cbins[:, None])

    I = np.multiply(I - cbins[ind], bins[ind])
    I = np.minimum(np.maximum(np.multiply(map[ind,0:3], np.repeat(1-I[:,None], 3, axis=1)) \
         + np.multiply(map[ind+1,0:3], np.repeat(I[:,None], 3, axis=1)),0),1)

    I = np.reshape(I, [H, W, 3]).astype(np.float32)

    return I


def single_gpu_ttc_test(model,
                    data_loader,
                    out_dir=None):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    count = 0
    delta = 0
    write = True
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            st = time.time()
            ttc_maps = model(return_loss=False, rescale=True, ttc_out=True, **data)
            delta += time.time() - st
            count += 1

        if count % 500 == 0:
            print("\nImg/sec: ", int(count / delta * 1000) / 1000)

        batch_size = len(ttc_maps)
        if out_dir:

            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            # ttc_maps = [data['ttc_maps'][0].data[0].permute(1, 2, 0)[0]]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            if not write:
                continue

            for i, (ttc_map, img, img_meta) in enumerate(zip(ttc_maps, imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                ttc_map = ttc_map[:h, :w].cpu().numpy()
                img = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]

                # ttc_map[ttc_map < 0.7] = 1.2
                # normalize ttc map to [0, 1]

                ttc_map = np.clip(ttc_map - 0.5, 0, 1.0)

                # ttc_map = np.uint8(disp2rgb(ttc_map)*255)
                # apply opencv jet colormap with center at 0.5
                # ttc_map = cv2.applyColorMap((ttc_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                # ttc_map = cv2.cvtColor(ttc_map, cv2.COLOR_BGR2RGB)

                # ttc_map = mmcv.imresize(ttc_map, (ori_w, ori_h))
                # img = mmcv.imresize(img, (ori_w, ori_h))

                # concat image and ttc map vertically
                # ttc_map = np.concatenate((img, ttc_map), axis=0)

                out_file = osp.join(out_dir, img_meta['ori_filename'])
                out_file = out_file.split('.')[0] + ".npy"
                print('saving to {out_file}')
                with open(out_file, 'wb') as f:
                    np.save(f, ttc_map, allow_pickle=True)
                # mmcv.imwrite(ttc_map, out_file)

        for _ in range(batch_size):
            prog_bar.update()
    print("\nImg/sec: ", int(count / delta * 1000) / 1000)


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, log=True, ttc_loss=False, error_func="mid", check_range=(0.5, 1.3)):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    ttc_losses = []
    dataset = data_loader.dataset
    # class_names = [c["name"] for c in dataset.coco.loadCats(dataset.cat_ids)]
    class_names = dataset.CLASSES
    print(class_names)
    delta = 0
    count = 0
    skip = True
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        st = time.time()
        with torch.no_grad():
            if ttc_loss:
                result, mid_loss = model(return_loss=False, rescale=True, error_func=error_func, check_range=check_range, **data)
                ttc_losses.append(mid_loss)
            else:
                result = model(return_loss=False, rescale=True, **data)
            if not skip:
                delta += time.time() - st
                count += 1
            else:
                skip = False

            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            if i == 0:
                print(data["img"][0].data[0].shape)
            if np.random.rand(1)[0] < 0.01 and log:
                log_image_with_boxes(
                    "proposals",
                    data["img"][0].data[0],
                    result[0],
                    bbox_tag="rpn_pseudo_label",
                    interval=500,
                    class_names=class_names,
                    img_norm_cfg=data["img_metas"][0].data[0][0]["img_norm_cfg"],
                )
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
        if ttc_loss:
            dist.barrier()
            ttc_losses = collect_results_gpu(ttc_losses, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
        if ttc_loss:
            dist.barrier()
            ttc_losses = collect_results_cpu(ttc_losses, len(dataset), ".ttc_losses")
    print("\nImg/sec: ", int(count / delta * 1000) / 1000)
    if ttc_loss:
        ttc_res = []
        ttc_dict = None
        if ttc_losses is not None:
            for res in ttc_losses:
                # check if res is list
                if isinstance(res, list):
                    ttc_res.extend(res)
                elif isinstance(res, dict):
                    if ttc_dict is None:
                        ttc_dict = res
                    else:
                        for k, v in res.items():
                            ttc_dict[k].extend(v)
                else:
                    ttc_res.append(res)
        if ttc_dict is not None:
            for k, v in ttc_dict.items():
                ttc_dict[k] = np.array(v)
            ttc_res = ttc_dict
        return results, ttc_res, class_names
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            os.listdir(tmpdir)
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
