import os.path as osp
import pickle
import shutil
import tempfile
import os
import scipy.io as sio
import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import cv2
import time
import numpy as np

def single_gpu_test(model, data_loader, show=False, out_dir=None):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped
        into the directory to save output results.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
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
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, iterNum = None):
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
    print(tmpdir)
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()

    if iterNum==None:
        output_mat_dir = os.path.join(tmpdir, 'mat')
        #output_png_dir = os.path.join(tmpdir, 'png')
    else:
        output_mat_dir = os.path.join(tmpdir, str(iterNum), 'mat')
        #output_png_dir = os.path.join(tmpdir, str(iterNum), 'png')

    print(output_mat_dir)
    if not os.path.exists(output_mat_dir):
        try:
            os.makedirs(output_mat_dir)
        except FileExistsError:
            pass
    '''
    print(output_png_dir)
    if not os.path.exists(output_png_dir):
        try:
            os.makedirs(output_png_dir)
        except FileExistsError:
            pass
    '''
    depth_output_dir = os.path.join(output_mat_dir, 'depth/mat')
    if not os.path.exists(depth_output_dir):
        os.makedirs(depth_output_dir)
    normal_output_dir = os.path.join(output_mat_dir, 'normal/mat')
    if not os.path.exists(normal_output_dir):
        os.makedirs(normal_output_dir)
    reflectance_output_dir = os.path.join(output_mat_dir, 'reflectance/mat')
    if not os.path.exists(reflectance_output_dir):
        os.makedirs(reflectance_output_dir)
    illumination_output_dir = os.path.join(output_mat_dir, 'illumination/mat')
    if not os.path.exists(illumination_output_dir):
        os.makedirs(illumination_output_dir)



    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    start_time = time.time()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            result = result.squeeze()
            #print(type(result))
            depth_pred = result[0]
            normal_pred = result[1]
            reflectance_pred = result[2]
            illumination_pred = result[3]
            #print(type(out_depth))
            #print(out_depth.shape)
            #depth_pred = out_depth.data.cpu().numpy()
            #depth_pred = depth_pred.squeeze()
            sio.savemat(os.path.join(depth_output_dir, '{}.mat'.format(data['img_metas'][-1].data[-1][-1]['img_id'])), {'result': depth_pred})

            #normal_pred = out_normal.data.cpu().numpy()
            #normal_pred = normal_pred.squeeze()
            sio.savemat(os.path.join(normal_output_dir, '{}.mat'.format(data['img_metas'][-1].data[-1][-1]['img_id'])), {'result': normal_pred})
            #reflectance_pred = out_reflectance.data.cpu().numpy()
            #reflectance_pred = reflectance_pred.squeeze()
            sio.savemat(os.path.join(reflectance_output_dir, '{}.mat'.format(data['img_metas'][-1].data[-1][-1]['img_id'])), {'result': reflectance_pred})

            #illumination_pred = out_illumination.data.cpu().numpy()
            #illumination_pred = illumination_pred.squeeze()
            sio.savemat(os.path.join(illumination_output_dir, '{}.mat'.format(data['img_metas'][-1].data[-1][-1]['img_id'])),{'result': illumination_pred})

            
            
            #sio.savemat(os.path.join(output_mat_dir, '{}.mat'.format(data['img_metas'][-1].data[-1][-1]['img_id'])), {'result': result})
            #png_res= 255*(1-result)
            #cv2.imwrite(os.path.join(output_png_dir, '{}.png'.format(data['img_metas'][-1].data[-1][-1]['img_id'])), png_res)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    tm = time.time() - start_time
    print(tm)
    
    '''
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results
    '''

def multi_gpu_test_city(model, data_loader, tmpdir=None, gpu_collect=False, iterNum=None):
    model.eval()
    print(tmpdir)
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()

    if iterNum is None:
        output_png_dir = os.path.join(tmpdir)
    else:

        output_png_dir = os.path.join(tmpdir)

    
    if not os.path.exists(output_png_dir):
        os.makedirs(output_png_dir, exist_ok=True)

 
                  
    categories = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle']
               
    png_result = ['class_001', 'class_002', 'class_003', 'class_004', 'class_005',
               'class_006', 'class_007', 'class_008', 'class_009', 'class_010',
               'class_011', 'class_012', 'class_013', 'class_014', 'class_015',
               'class_016', 'class_017', 'class_018', 'class_019']
                  

    # categories = ['bottle', 'chair', 'diningtable', 'person']

        
    for png in png_result:
        png_output_dir = os.path.join(output_png_dir, png)
        if not os.path.exists(png_output_dir):
            os.makedirs(png_output_dir)

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    start_time = time.time()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            result = result.squeeze()
            
            for idx, category in enumerate(categories):
                pred = result[idx]
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                
                img_id = data['img_metas'][-1].data[-1][-1]['img_id']

                png_output_dir = os.path.join(output_png_dir, png_result[idx], )


                pred = (pred * 255).astype(np.uint8)
                imsave(os.path.join(png_output_dir, '{}.png'.format(img_id)), pred)


        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    tm = time.time() - start_time
    print(tm)

def multi_gpu_test_sbd(model, data_loader, tmpdir=None, gpu_collect=False, iterNum=None):
    model.eval()
    print(tmpdir)
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()

    if iterNum is None:
        output_mat_dir = os.path.join(tmpdir, 'mat')
        output_png_dir = os.path.join(tmpdir, 'png')
    else:
        output_mat_dir = os.path.join(tmpdir, str(iterNum), 'mat')
        output_png_dir = os.path.join(tmpdir, str(iterNum), 'png')

    if not os.path.exists(output_png_dir):
        os.makedirs(output_png_dir, exist_ok=True)

    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                  'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    png_result = ['class_001', 'class_002', 'class_003', 'class_004', 'class_005',
                  'class_006', 'class_007', 'class_008', 'class_009', 'class_010',
                  'class_011', 'class_012', 'class_013', 'class_014', 'class_015',
                  'class_016', 'class_017', 'class_018', 'class_019', 'class_020']

    for png in png_result:
        png_output_dir = os.path.join(output_png_dir, png, 'png')
        if not os.path.exists(png_output_dir):
            os.makedirs(png_output_dir)

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    start_time = time.time()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            img_meta = data['img_metas'][0].data[0][0]  
            img_shape = img_meta['img_shape']
            ori_shape = img_meta['ori_shape']  # (H, W, C)

            img = data['img'][0]

            if img.dim() == 4:
                img = img.squeeze(0)

            if img.dim() == 3:
                img = img.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            else:
                raise ValueError(f"Unexpected img dimension: {img.dim()}")

            # Pad to minimum size
            padded_img, (bottom, right) = pad_to_min_size(img, min_size=(340, 340))

            # Convert back to tensor
            padded_img_tensor = torch.tensor(padded_img.transpose(2, 0, 1)).unsqueeze(0).to(
                torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            
            #print(padded_img_tensor.shape)
            data['img'] = [padded_img_tensor]  # list 
            
            #print(data)
            


            # Model inference
            result = model(return_loss=False, rescale=True, **data)
            result = result.squeeze()
            

            for idx, category in enumerate(categories):
                pred = result[idx]

                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()

                img_id = data['img_metas'][-1].data[-1][-1]['img_id']
                png_output_dir = os.path.join(output_png_dir, png_result[idx])

                # Scale and remove padding
                pred = (pred * 255).astype(np.uint8)
                pred = remove_padding(pred, bottom, right)

                # Save the image
                imsave(os.path.join(png_output_dir, f'{img_id}.png'), pred, check_contrast=False)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    tm = time.time() - start_time
    print(tm)


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results with CPU."""
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
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
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
    """Collect results with GPU."""
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
