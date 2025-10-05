from __future__ import absolute_import, division, print_function

import os
import time

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

import dataloaders
from metricsgren import quaternion2matrix, compute_pose_error, Result, AverageMeter, from_euler_t
import torch.nn.functional as F
from tqdm import tqdm
from path import Path
import h5py

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def val_pap(encoder, depth_decoder, val_loader, opt):
    """
    Validate the model after a epoch
    """
    MIN_DEPTH = 0.5
    MAX_DEPTH = 100
    # MIN_DEPTH = 1
    # MAX_DEPTH = 200

    average_meter = AverageMeter()
    end = time.time()

    depth_preds = []
    gts = []
    colors = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(val_loader)):
            data_time = time.time() - end
            if opt.is_rgbd:
                input_color = inputs[("rgbsd", 0, 0)].cuda()
            else:
                input_color = inputs[("color", 0, 0)].cuda()
            colors.append(input_color[0, ...].cpu().detach().numpy().astype(np.uint8))

            # compute output
            end = time.time()

            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            output = depth_decoder(encoder(input_color))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], MIN_DEPTH, MAX_DEPTH)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            gpu_time = time.time() - end

            pred_depth = 1 / pred_disp
            depth_pred = torch.tensor(pred_depth)
            # depth_preds.append(depth_pred[0, ...].cpu().detach().numpy())
            depth_preds.append(depth_pred.squeeze().cpu().detach().numpy())
            if ('dgt', 0) in inputs.keys():
                depth_gt = inputs[("dgt", 0)][:, 0, ...]
                gts.append(depth_gt[0, ...].cpu().data.numpy())

                # mask = depth_gt > 0
                mask = np.logical_and(depth_gt > MIN_DEPTH, depth_gt < MAX_DEPTH).bool()
                depth_gt = depth_gt[mask]
                depth_pred = depth_pred[mask]
                depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
                depth_pred = torch.clamp(depth_pred, min=MIN_DEPTH, max=MAX_DEPTH)

                result = Result()
                result.evaluate(depth_pred, depth_gt)
                average_meter.update(result, gpu_time, data_time, depth_pred.size(0))
                end = time.time()

    if ('dgt', 0) in inputs.keys():
        avg = average_meter.average()

        print_string = "DEPTH:  | rmse: {average.rmse:.5f} | absrel: {average.absrel:.5f} | mae: {average.mae:.5f}" + \
                       " | a1: {average.delta1:.5f} | a2: {average.delta2:.5f} | a3: {average.delta3:.5f}" + \
                       " | data_time: {average.data_time:.5f} | gpu_time: {average.gpu_time:.5f}"
        print(print_string.format(average=avg))
    else:
        print('-->NO GT!')

    return [depth_preds, colors, gts]


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        # filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        # print(opt.data_path)

        #dataset = dataloaders.testDataset(opt.data_path, 'val', opt.height, opt.width, opt.frame_ids, 4, has_gt=True)
        dataset = dataloaders.testDataset(opt.data_path, 'val', opt.height, opt.width, opt.frame_ids, 4, has_gt=False)
        # dataset = dataloaders.aDataset(opt.data_path, 'val', opt.height, opt.width, opt.frame_ids, 4)
        #dataset = dataloaders.exrDataset(opt.data_path, 'val', opt.height, opt.width, opt.frame_ids, 4)
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False, is_rgbd=opt.is_rgbd)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        depth_preds, colors, gts = val_pap(encoder, depth_decoder, dataloader, opt)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        gts = []
        colors = []
        depth_preds = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            depth_preds = depth_preds[eigen_to_benchmark_ids]

    # output_num = {1, 2, 3}
    # output_length = len(depth_preds)
    # output_num = range(output_length)
    # depth_preds_length = 0
    print(depth_preds[0].shape)
    preddd = np.concatenate(depth_preds, axis=0)
    print(preddd.shape)

    # sq = [0, 10, 40, 60]
    # for i in tqdm(sq):
    #     dp = depth_preds[i]
    #     dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    #     dp_color = cv2.applyColorMap(dp, cv2.COLORMAP_JET)
    #     # dp_color = cv2.applyColorMap(cv2.convertScaleAbs(dp, alpha=1), cv2.COLORMAP_JET)
    #     cv2.imwrite('{}.png'.format(i), dp_color)

    file = h5py.File('preds.hdf5', "w")
    # dataset = file.create_dataset('preds', shape=(803, 320, 320), dtype='float32',  # seq4: 8401
    h5_dataset = file.create_dataset('preds', shape=preddd.shape, dtype='float32',  # seq4: 8401
                                  compression='gzip', compression_opts=4,
                                  shuffle=False,
                                  fletcher32=True)
    # for i in tqdm(output_num):
    #     # save np
    #     dataset[depth_preds_length:depth_preds_length+depth_preds[i].shape[0], ...] = depth_preds[i]
    #     depth_preds_length += depth_preds[i].shape[0]

        # n_depth = np.expand_dims(depth_preds[i], axis=2)
        # n_depth = (n_depth - np.min(n_depth)) * 255 / (np.max(n_depth) - np.min(n_depth))
        # # n_depth = (np.max(n_depth) - np.min(n_depth)) / (n_depth - np.min(n_depth)) * 255   # 1/d
        # n_depth = np.repeat(n_depth, 3, axis=2).astype(np.uint8)

        # gt = np.expand_dims(gts[i], axis=2)
        # gt = (gt - np.min(gt)) * 255 / (np.max(gt) - np.min(gt))
        # gt = np.repeat(gt, 3, axis=2).astype(np.uint8)
        #
        # color = colors[i].transpose([1, 2, 0]).astype(np.uint8)
    h5_dataset[...] = preddd[...]
    file.close()

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
