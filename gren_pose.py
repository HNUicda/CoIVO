from __future__ import absolute_import, division, print_function

import os
import torch
import networks
import numpy as np

from torch.utils.data import DataLoader
from layers import transformation_from_parameters
from utils import readlines
from options import MonodepthOptions

import csv
import visdom
import dataloaders
from metricsgren import quaternion2matrix, compute_pose_error, Result, AverageMeter, from_euler_t
from tqdm import trange


# def get_gt_pose():
#     """load gt pose"""
#     gt_poses_path = '../position_rotation.csv'
#     with open(gt_poses_path, 'r') as f_position:
#         f_position_reader = np.loadtxt(f_position, delimiter=',', skiprows=1)
#     pose_gt = []
#     scale = 1  # pose尺度
#     s_num = 0  # 当前图像编号
#     # 初始旋转角归零
#     init_pose = np.array([[0.0, 1.0, 0.0],
#                           [0.0, 0.0, -1.0],
#                           [-1.0, 0.0, 0.0]])
#     # start_rot = np.eye(3)
#     for row in f_position_reader:
#         if s_num >= 50:  # 从编号50开始取gt
#             dataxyz = [(row[0] - 1.767435) * scale,
#                        (row[1] - 9.150367) * scale,
#                        (row[2] + 2.338273) * scale]
#
#             dataxyz = np.expand_dims(np.array(dataxyz), axis=1)
#             mat = quaternion2matrix(row[3:7])  # 四元数转矩阵
#             mat = np.dot(mat, init_pose)
#             temp = np.concatenate([mat, dataxyz], axis=1)
#             pose_gt.append(temp)
#         s_num += 1
#     return pose_gt


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        # cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


def dump_r(source_to_target_transformations):
    rs = []
    cam_to_world = np.eye(4)
    rs.append(cam_to_world[:3, :3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        # cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        rs.append(cam_to_world[:3, :3])
    return rs


# from https://github.com/tinghuiz/SfMLearner
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


def compute_re(gtruth_r, pred_r):
    RE = 0
    gt = gtruth_r
    pred = pred_r
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose @ np.linalg.inv(pred_pose)
        s = np.linalg.norm([R[0, 1] - R[1, 0],
                            R[1, 2] - R[2, 1],
                            R[0, 2] - R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return RE / gtruth_r.shape[0]


def val_pap(models, val_loader, opt):
    """
    Validate the model after a epoch
    """
    pred_poses = []
    frame_ids = [0, 1]  # pose network only takes two frames as input
    for inputs in val_loader:
        for key, ipt in inputs.items():
            inputs[key] = ipt.cuda()

        tat = inputs[("color_aug", frame_ids[0], 0)]
        soc = inputs[("color_aug", frame_ids[1], 0)]
        all_color_aug = torch.cat([tat, soc], 1)
        with torch.no_grad():
            if opt.pose_model_type == 'separate_resnet':
                pose_encoder, pose_decoder = models
                features = [pose_encoder(all_color_aug)]
                axisangle, translation = pose_decoder(features)
                pred_poses.append(
                    transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
            elif opt.pose_model_type == "msfusion":
                depth_encoder, depth_decoder, pose_net = models
                pred_disps1 = depth_decoder(depth_encoder(tat))
                pred_disps2 = depth_decoder(depth_encoder(soc))
                disp_inputs = torch.cat([pred_disps1[('disp', 0)].repeat(1, 3, 1, 1),
                                         pred_disps2[('disp', 0)].repeat(1, 3, 1, 1)], dim=1)
                axisangle, translation = pose_net([all_color_aug, disp_inputs])
                pred_poses.append(from_euler_t(axisangle, translation).cpu().numpy())

    pred_poses = np.concatenate(pred_poses)

    # 计算数值结果
    # seq_length = 5
    # framework = pred_poses.shape[0] // seq_length
    # errors = np.zeros((framework, 2), np.float32)
    # predictions_array = np.zeros((framework, seq_length, 3, 4))
    #
    # pose_gt = get_gt_pose()
    #
    # for j in trange(framework):
    #     # gt的pose取值
    #     pose_seq = []
    #     for i in range(seq_length):
    #         pose = pose_gt[j * seq_length + i]
    #         pose_mat = pose
    #         pose_seq.append(pose_mat[0:3, :])
    #
    #     final_poses_gt = np.stack(pose_seq, axis=0)
    #
    #     # 预测的pose计算
    #     global_pose = np.eye(4)
    #     poses = [global_pose[0:3, :]]
    #
    #     for iter in range(seq_length - 1):
    #         pose_mat = pred_poses[j * seq_length + iter]
    #         # pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
    #
    #         global_pose = global_pose @ np.linalg.inv(pose_mat)
    #         poses.append(global_pose[0:3, :])
    #
    #     final_poses = np.stack(poses, axis=0)
    #
    #     predictions_array[j] = final_poses
    #
    #     ATE, RE = compute_pose_error(final_poses_gt, final_poses)
    #     errors[j] = ATE, RE
    #
    # mean_errors = errors.mean(0)
    # std_errors = errors.std(0)
    # print_string = "POSE:  | ATE: {:.5f} ± {:.5f} | RE: {:.5f} ± {:.5f}"
    # print(print_string.format(mean_errors[0], std_errors[0], mean_errors[1], std_errors[1]))
    # print('')

    # save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
    # save_folder = os.path.join(opt.log_dir, opt.model_name, "models")
    save_path = os.path.join(opt.load_weights_folder, "poses.npy")
    np.save(save_path, pred_poses)


def evaluate(opt):
    """Evaluate odometry on the SCARED dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    dataset = dataloaders.testDataset(opt.data_path, 'val', opt.height, opt.width, opt.frame_ids, 4, has_gt=False)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    models = []
    if opt.pose_model_type == 'separate_resnet':
        pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
        pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

        pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
        pose_encoder.load_state_dict(torch.load(pose_encoder_path))

        pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
        pose_decoder.load_state_dict(torch.load(pose_decoder_path))

        pose_encoder.cuda()
        pose_encoder.eval()
        models.append(pose_encoder)

        pose_decoder.cuda()
        pose_decoder.eval()
        models.append(pose_decoder)

    elif opt.pose_model_type == 'msfusion':
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        pose_net_path = os.path.join(opt.load_weights_folder, "pose.pth")

        encoder_dict = torch.load(encoder_path)
        encoder = networks.ResnetEncoder(opt.num_layers, False, is_rgbd=opt.is_rgbd)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))
        pose_net = networks.create_posenet_multi_layer()

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))
        pose_net.load_state_dict(torch.load(pose_net_path))

        encoder.cuda()
        encoder.eval()
        models.append(encoder)

        depth_decoder.cuda()
        depth_decoder.eval()
        models.append(depth_decoder)

        pose_net.cuda()
        pose_net.eval()
        models.append(pose_net)

    print("-> Computing pose predictions")
    val_pap(models, dataloader, opt)

    # opt.frame_ids = [0, 1]  # pose network only takes two frames as input
    #
    # with torch.no_grad():
    #     for inputs in dataloader:
    #         for key, ipt in inputs.items():
    #             inputs[key] = ipt.cuda()
    #
    #         all_color_aug = torch.cat([inputs[("color", 1, 0)], inputs[("color", 0, 0)]], 1)
    #
    #         features = [pose_encoder(all_color_aug)]
    #         axisangle, translation = pose_decoder(features)
    #
    #         pred_poses.append(
    #             transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
    #
    # pred_poses = np.concatenate(pred_poses)
    #
    # gt_path = os.path.join(os.path.dirname(__file__), "splits", "endovis", "gt_poses_sq2.npz")
    # gt_local_poses = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
    #
    # ates = []
    # res = []
    # num_frames = gt_local_poses.shape[0]
    # track_length = 5
    # for i in range(0, num_frames - 1):
    #     local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
    #     gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
    #     local_rs = np.array(dump_r(pred_poses[i:i + track_length - 1]))
    #     gt_rs = np.array(dump_r(gt_local_poses[i:i + track_length - 1]))
    #
    #     ates.append(compute_ate(gt_local_xyzs, local_xyzs))
    #     res.append(compute_re(local_rs, gt_rs))
    #
    # print("\n   Trajectory error: {:0.4f}, std: {:0.4f}\n".format(np.mean(ates), np.std(ates)))
    # print("\n   Rotation error: {:0.4f}, std: {:0.4f}\n".format(np.mean(res), np.std(res)))


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
