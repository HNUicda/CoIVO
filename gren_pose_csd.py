from __future__ import absolute_import, division, print_function

import os
import torch
import networks
import numpy as np

from torch.utils.data import DataLoader
from options import MonodepthOptions

from tqdm import tqdm
import dataloaders
from metricsgren import from_euler_t
from PDutils import get_csd_pose
from metricsgren import compute_atere

txt1 = 'Camera Position Data.txt'
txt2 = 'Camera Quaternion Rotation Data.txt'
pose_path = 'E:\experiment\ColVO-20240408'

def val_pap(depth_encoder, depth_decoder, pose_net, val_loader, opt):
    """
    Validate the model after a epoch
    """
    pred_poses = []
    # frame_ids = [0, 1]  # pose network only takes two frames as input
    for rgb, depth in tqdm(val_loader):
        # for key, ipt in inputs.items():
        #     inputs[key] = ipt.cuda()
        tgt = rgb[0].cuda()
        ref = rgb[1][1].cuda()
        # depth = depth.cuda()

        # all_color_aug = torch.cat([inputs["color_aug", 0, 0], inputs["color_aug", 1, 0]], 1)
        all_color_aug = torch.cat([tgt, ref], 1)
        with torch.no_grad():
            # if opt.pose_model_type == "separate_resnet":  # 默认separate
            #     features = [pose_encoder(all_color_aug)]
            #     axisangle, translation = pose_decoder(features)
            #     pred_poses.append(
            #         transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
            # elif opt.pose_model_type == "msfusion":
            pred_disp1 = depth_decoder(depth_encoder(tgt))
            pred_disp2 = depth_decoder(depth_encoder(ref))
            disp_inputs = torch.cat([pred_disp1[('disp', 0)].repeat(1, 3, 1, 1),
                                     pred_disp2[('disp', 0)].repeat(1, 3, 1, 1)], dim=1)
            axisangle, translation = pose_net([all_color_aug, disp_inputs])
            pred_poses.append(from_euler_t(axisangle, translation).cpu().numpy())

    pred_poses = np.concatenate(pred_poses)

    #pose_gt = get_csd_pose(txt1, txt2)
    pose_gt=get_csd_pose(pose_path,True)
    pose_gt = pose_gt[2]

    mean_errors, std_errors = compute_atere(pose_gt, pred_poses)
    print_string = "POSE:  | ATE: {:.5f} ± {:.5f} | RE: {:.5f} ± {:.5f}"
    print(print_string.format(mean_errors[0], std_errors[0], mean_errors[1], std_errors[1]))
    print('')

    # save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
    # save_folder = os.path.join(opt.log_dir, opt.model_name, "models")
    # save_path = os.path.join(opt.load_weights_folder, "poses.npy")
    save_path = "poses.npy"
    np.save(save_path, pred_poses)


def evaluate(opt):
    """Evaluate odometry on the SCARED dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)
    dataset = dataloaders.exrDataset_sin(opt.data_path, 'val')
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

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
    depth_decoder.cuda()
    depth_decoder.eval()
    pose_net.cuda()
    pose_net.eval()

    # pred_poses = []

    print("-> Computing pose predictions")

    val_pap(encoder, depth_decoder, pose_net, dataloader, opt)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
