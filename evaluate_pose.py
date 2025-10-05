# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import transformation_from_parameters
from utils import readlines
from options import MonodepthOptions
from datasets import KITTIOdomDataset
import networks
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm, trange

import dataloaders
from metrics import quaternion2matrix, from_euler_t, compute_atere, get_gt_pose


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# # from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):
    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def evaluate(opt):
    """Evaluate odometry on the KITTI dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    assert opt.eval_split == "odom_9" or opt.eval_split == "odom_10", \
        "eval_split should be either odom_9 or odom_10"

    sequence_id = int(opt.eval_split.split("_")[1])

    filenames = readlines(
        os.path.join(os.path.dirname(__file__), "splits", "odom",
                     "test_files_{:02d}.txt".format(sequence_id)))

    dataset = dataloaders.testDataset(opt.data_path, 'val', opt.height, opt.height,
                                      [0, 1], 4)
    # dataset = KITTIOdomDataset(opt.data_path, filenames, opt.height, opt.width,
    #                            [0, 1], 4, is_train=False)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    if opt.pose_model_type == "msfusion":
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        pose_model_path = os.path.join(opt.load_weights_folder, "pose.pth")

        encoder_dict = torch.load(encoder_path)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))
        depth_decoder.load_state_dict(torch.load(decoder_path))

        pose_model = networks.create_posenet_multi_layer()
        pose_model.load_state_dict(torch.load(pose_model_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()
        pose_model.cuda()
        pose_model.eval()

    else:
        pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
        pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

        pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
        pose_encoder.load_state_dict(torch.load(pose_encoder_path))

        pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
        pose_decoder.load_state_dict(torch.load(pose_decoder_path))

        pose_encoder.cuda()
        pose_encoder.eval()
        pose_decoder.cuda()
        pose_decoder.eval()

    pred_poses = []

    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids], 1)

            if opt.pose_model_type == "msfusion":
                prev_disp = depth_decoder(encoder(all_color_aug[0]))
                # prev_disp = self.models["depth"](
                #     self.models["encoder"](inputs["color_aug", pose_inputs_num[0], 0]))
                cur_disp = depth_decoder(encoder(all_color_aug[1]))
                # cur_disp = self.models["depth"](
                #     self.models["encoder"](inputs["color_aug", pose_inputs_num[1], 0]))
                disp_inputs = torch.cat([prev_disp[('disp', 0)].repeat(1, 3, 1, 1),
                                         cur_disp[('disp', 0)].repeat(1, 3, 1, 1)], dim=1)
                pose_inputs = [all_color_aug, disp_inputs]  # msfusion
                axisangle, translation = pose_model(pose_inputs)

                output = from_euler_t(axisangle, translation, invert=(opt.frame_ids[0] < 0))
            else:
                features = [pose_encoder(all_color_aug)]
                axisangle, translation = pose_decoder(features)

                output = transformation_from_parameters(axisangle[:, 0], translation[:, 0])

            pred_poses.append(output.cpu().numpy())

    pred_poses = np.concatenate(pred_poses)

    pose_gt = get_gt_pose()  # 获取gtpose

    # gt_poses_path = './position_rotation.csv'
    # with open(gt_poses_path, 'r') as f_position:
    #     f_position_reader = np.loadtxt(f_position, delimiter=',', skiprows=1)
    # pose_gt = []
    # scale = 1  # pose尺度
    # s_num = 0  # 当前图像编号
    # # 初始旋转角归零
    # # init_pose = np.array([[0.0, 1.0, 0.0],
    # #                       [0.0, 0.0, -1.0],
    # #                       [-1.0, 0.0, 0.0]])
    # # 估计结果向真实值对齐：都乘第一帧的rotation
    # # init_pose = R.from_quat(np.array([-0.02238709, -0.9567472, -0.07655796, 0.2797724])).as_matrix()
    # for row in f_position_reader:
    #     if s_num >= 20:  # seq4
    #         x = (row[0] - 1.813364) * scale
    #         y = (row[1] - 9.100907) * scale
    #         z = (row[2] + 2.310724) * scale
    #         # dataxyz = [(row[0] - 1.813364) * scale,
    #         #            (row[1] - 9.100907) * scale,
    #         #            (row[2] + 2.310724) * scale]
    #         dataxyz = [x, y, z]
    #
    #         dataxyz = np.expand_dims(np.array(dataxyz), axis=1)
    #         mat = quaternion2matrix(row[3:7])  # 四元数转矩阵
    #         # mat = np.dot(mat, init_pose)
    #         temp = np.concatenate([mat, dataxyz], axis=1)
    #         pose_gt.append(temp)
    #     s_num += 1

    mean_errors, std_errors = compute_atere(pose_gt, pred_poses)
    error_names = ['ATE', 'RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))


    # gt_poses_path = os.path.join(opt.data_path, "poses", "{:02d}.txt".format(sequence_id))
    # gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    # gt_global_poses = np.concatenate(
    #     (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    # gt_global_poses[:, 3, 3] = 1
    # gt_xyzs = gt_global_poses[:, :3, 3]
    #
    # gt_local_poses = []
    # for i in range(1, len(gt_global_poses)):
    #     gt_local_poses.append(
    #         np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    # ates = []
    # num_frames = gt_xyzs.shape[0]
    # track_length = 5
    # for i in range(0, num_frames - 1):
    #     local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
    #     gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
    #
    #     ates.append(compute_ate(gt_local_xyzs, local_xyzs))
    #
    # print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))

    save_path = os.path.join(opt.load_weights_folder, "poses.npy")
    np.save(save_path, pred_poses)
    print("-> Predictions saved to", save_path)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
