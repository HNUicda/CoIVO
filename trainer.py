# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


from __future__ import absolute_import, division, print_function
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

import csv
import visdom
import dataloaders
from metrics import quaternion2matrix, Result, AverageMeter, from_euler_t, compute_atere, get_gt_pose
import cv2
import image_opencv as imgcv
vis = visdom.Visdom(port=8009, env='mono2_gt',use_incoming_socket=False)
#vis = visdom.Visdom(port=8009, env='mono2_gt')

slim_penalty = lambda var: torch.abs(var).sum()


def brightnes_equator(source, target):
    def image_stats(image):
        # compute the mean and standard deviation of each channel

        l = image[:, 0, :, :]
        a = image[:, 1, :, :]
        b = image[:, 2, :, :]

        (lMean, lStd) = (torch.mean(torch.squeeze(l)), torch.std(torch.squeeze(l)))

        (aMean, aStd) = (torch.mean(torch.squeeze(a)), torch.std(torch.squeeze(a)))

        (bMean, bStd) = (torch.mean(torch.squeeze(b)), torch.std(torch.squeeze(b)))

        # return the color statistics
        return (lMean, lStd, aMean, aStd, bMean, bStd)

    def color_transfer(source, target):
        # convert the images from the RGB to L*ab* color space, being
        # sure to utilizing the floating point data type (note: OpenCV
        # expects floats to be 32-bit, so use that instead of 64-bit)

        # compute color statistics for the source and target images
        (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
        (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

        # subtract the means from the target image
        l = target[:, 0, :, :]
        a = target[:, 1, :, :]
        b = target[:, 2, :, :]

        l = l - lMeanTar
        # print("after l",torch.isnan(l))
        a = a - aMeanTar
        b = b - bMeanTar
        # scale by the standard deviations
        l = (lStdTar / lStdSrc) * l
        a = (aStdTar / aStdSrc) * a
        b = (bStdTar / bStdSrc) * b
        # add in the source mean
        l = l + lMeanSrc
        a = a + aMeanSrc
        b = b + bMeanSrc
        transfer = torch.cat((l.unsqueeze(1), a.unsqueeze(1), b.unsqueeze(1)), 1)
        # print(torch.isnan(transfer))
        return transfer

    # return the color transferred image
    transfered_image = color_transfer(target, source)
    return transfered_image


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            is_rgbd=self.opt.is_rgbd)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            elif self.opt.pose_model_type == "msfusion":
                self.models["pose"] = networks.create_posenet_multi_layer()  # TODO:定义msfusion

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

            if self.opt.pose_model_type == "msfusion":
                self.slim_params = []
                for name, param in self.models["pose"].named_parameters():
                    if param.requires_grad and name.endswith('weight') and 'bn2' in name:
                        if len(self.slim_params) % 2 == 0:
                            self.slim_params.append(param[:len(param) // 2])
                        else:
                            self.slim_params.append(param[len(param) // 2:])

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        # datasets_dict = {"kitti": datasets.KITTIRAWDataset,
        #                  "kitti_odom": datasets.KITTIOdomDataset}
        # self.dataset = datasets_dict[self.opt.dataset]

        self.dataset = dataloaders.aDataset

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        #print(fpath)
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        # num_train_samples = len(train_filenames)
        # self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        #读取数据
        train_dataset = self.dataset(self.opt.data_path, 'train', self.opt.height, self.opt.width,
                                     self.opt.frame_ids, 4)
        # train_dataset = self.dataset(
        #     self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
        #     self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        val_dataset = self.dataset(self.opt.data_path, 'val', self.opt.height, self.opt.width,
                                   self.opt.frame_ids, 4)
        # val_dataset = self.dataset(self.opt.data_path, 'val', self.opt.height, self.opt.height,
        #                            [0, 1], 4)
        # val_dataset = self.dataset(
        #     self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
        #     self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        # self.val_loader = DataLoader(
        #     val_dataset, self.opt.batch_size, True,
        #     num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        # visdom plot
        self.plot_data = {'X': [], 'Y': [], 'legend': ['total_loss', 'gt_loss']}

        # get GT pose
        self.pose_gt = get_gt_pose()

        # create test_csv
        self.fieldnames = ['ate_mean', 'ate_std', 're_mean', 're_std',
                           'rmse', 'absrel', 'mae', 'delta1', 'delta2', 'delta3',
                           'mse', 'lg10', 'data_time', 'gpu_time']
        #self.test_csv = os.path.join(self.log_path, 'test.csv')
        self.test_csv = os.path.join(self.log_path, 'test1.csv')
        with open(self.test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0

        # 测试用代码
        self.test = self.epoch - 1

        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            self.val_pap()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        sum_of_loss = 0
        # sum_of_re_loss = 0
        sum_of_gt_loss = 0
        sum_of_gt_pose_loss = 0

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)
            sum_of_loss += losses["loss"].cpu().detach().numpy()
            # sum_of_re_loss += losses["reprojection_loss"].cpu().detach().numpy()

            # TODO:loss输出修改
            sum_of_gt_loss += losses["gt_loss"].cpu().detach().numpy()
            sum_of_gt_pose_loss += losses["gt_pose_loss"].cpu().detach().numpy()
            # sum_of_gt_loss += losses["gt_loss"]
            # sum_of_gt_pose_loss += losses["gt_pose_loss"]

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                # 输出pred
                n_depth = outputs[('depth', 0, 0)][0].cpu().detach().numpy().transpose([1, 2, 0])
                # n_depth = 1 / n_depth
                n_depth = (n_depth - np.min(n_depth)) * 255 / (np.max(n_depth) - np.min(n_depth))
                # 跑出来的n_depth
                n_depth = np.repeat(n_depth, 3, axis=2).astype(np.uint8)
                ##test将灰色图变成伪彩色图
                filename = str(self.step + 1)
                imgcv.image_grey_save(n_depth, filename)

                gt = inputs[('dgt', 0)][0].cpu().data.numpy().transpose([1, 2, 0])
                gt = (gt - np.min(gt)) * 255 / (np.max(gt) - np.min(gt))
                gt = np.repeat(gt, 3, axis=2).astype(np.uint8)

                visuals = {
                    'frame': inputs[('color', 0, 0)][0].cpu().detach().numpy().transpose([1, 2, 0]).astype(np.uint8),
                    'pred': n_depth,
                    'gt': gt
                }

                idx = 1
                for label, image_numpy in visuals.items():
                    vis.image(
                        image_numpy.transpose([2, 0, 1]),
                        opts=dict(title=label),
                        win=str(idx + 1)
                    )
                    idx += 1

                if "depth_gt" in inputs.keys():
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()

        # 输出loss
        epoch_loss = {
            'total_loss': (sum_of_loss / len(self.train_loader)),
            # 're_loss': (sum_of_re_loss / len(self.train_loader)),
            'gt_loss': (sum_of_gt_loss / len(self.train_loader)),
            'gt_pose_loss': (sum_of_gt_pose_loss / len(self.train_loader)),
        }
        self.plot_data['legend'] = list(epoch_loss.keys())
        self.plot_data['X'].append(self.epoch)
        self.plot_data['Y'].append([epoch_loss[k] for k in self.plot_data['legend']])

        vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win='1'
        )

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if isinstance(ipt, list):
                inputs[key] = [temp.to(self.device) for temp in ipt]
            else:
                inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            if self.opt.is_rgbd:
                features = self.models["encoder"](inputs["rgbsd", 0, 0])  # TODO:is_rgbd
            else:
                features = self.models["encoder"](inputs["color_aug", 0, 0])
                # geo
                ref_depths = {}
                ref_features = [self.models["encoder"](inputs[("color_aug", i, 0)]) for i in self.opt.frame_ids[1:]]
                ref_dips = [self.models["depth"](i) for i in ref_features]
                for i in range(len(ref_features)):
                    for scale in self.opt.scales:
                        ref_depths[('ref_depth', self.opt.frame_ids[i + 1], scale)] = 1 / ref_dips[i][('disp', scale)]
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features, outputs))

        self.generate_images_pred(inputs, outputs, ref_depths)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features, outputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs_num = [f_i, 0]
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs_num = [0, f_i]
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)
                    elif self.opt.pose_model_type == "msfusion":
                        rgb_inputs = torch.cat(pose_inputs, dim=1)
                        with torch.no_grad():
                            prev_disp = self.models["depth"](
                                self.models["encoder"](inputs["color_aug", pose_inputs_num[0], 0]))
                            cur_disp = self.models["depth"](
                                self.models["encoder"](inputs["color_aug", pose_inputs_num[1], 0]))
                            disp_inputs = torch.cat([prev_disp[('disp', 0)].repeat(1, 3, 1, 1),
                                                     cur_disp[('disp', 0)].repeat(1, 3, 1, 1)], dim=1)
                        # prev_disp = inputs[('dgt', pose_inputs_num[0])]
                        # cur_disp = inputs[('dgt', pose_inputs_num[1])]
                        # disp_inputs = torch.cat([prev_disp.repeat(1, 3, 1, 1),
                        #                          cur_disp.repeat(1, 3, 1, 1)], dim=1)
                        pose_inputs = [rgb_inputs, disp_inputs]

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    if self.opt.pose_model_type == "msfusion":
                        outputs[("cam_T_cam", 0, f_i)] = from_euler_t(axisangle, translation, invert=(f_i < 0))
                    else:
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    inputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        inputs['axi_gt'][:, 0], inputs['tra_gt'][:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
           # inputs = self.val_iter.next()
           inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            #inputs = self.val_iter.next()

            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs.keys():
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def val_pap(self):
        """
        Validate the model after a epoch
        """
        self.set_eval()

        average_meter = AverageMeter()
        end = time.time()
        for batch_idx, inputs in enumerate(self.val_loader):
            if len(inputs[("dgt", 0)]) != self.opt.batch_size:
                break
            torch.cuda.synchronize()
            data_time = time.time() - end

            # compute output
            end = time.time()
            with torch.no_grad():
                outputs, losses = self.process_batch(inputs)

            torch.cuda.synchronize()
            gpu_time = time.time() - end

            if ('dgt', 0) in inputs.keys():
                # _, depth_pred = disp_to_depth(outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                depth_pred = outputs[("depth", 0, 0)].detach()
                # depth_pred = torch.clamp(F.interpolate(
                #     depth_pred, [320, 320], mode="bilinear", align_corners=False), 1e-3, 80)
                # depth_pred = depth_pred.detach()
                depth_gt = inputs[("dgt", 0)]

                mask = depth_gt > 0
                # mask = np.logical_and(depth_gt > self.opt.min_depth, depth_gt < self.opt.max_depth).bool()
                depth_gt = depth_gt[mask]
                depth_pred = depth_pred[mask]
                depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
                depth_pred = torch.clamp(depth_pred, min=self.opt.min_depth, max=self.opt.max_depth)

                result = Result()
                result.evaluate(depth_pred, depth_gt)
                average_meter.update(result, gpu_time, data_time, depth_pred.size(0))
                end = time.time()

        avg = average_meter.average()

        print_string = "DEPTH:  | rmse: {average.rmse:.5f} | absrel: {average.absrel:.5f} | mae: {average.mae:.5f}" + \
                       " | a1: {average.delta1:.5f} | a2: {average.delta2:.5f} | a3: {average.delta3:.5f}" + \
                       " | data_time: {average.data_time:.5f} | gpu_time: {average.gpu_time:.5f}"
        print(print_string.format(average=avg))

        # dataset = self.dataset(self.opt.data_path, 'val', self.opt.height, self.opt.height, [0, 1], 4)
        #
        # dataloader = DataLoader(dataset, self.opt.batch_size, shuffle=False,
        #                         num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)

        pred_poses = []
        frame_ids = [0, 1]  # pose network only takes two frames as input
        for inputs in self.val_loader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in frame_ids], 1)
            with torch.no_grad():
                if self.opt.pose_model_type == "separate_resnet":
                    features = [self.models["pose_encoder"](all_color_aug)]
                    axisangle, translation = self.models["pose"](features)
                    pred_poses.append(
                        transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
                elif self.opt.pose_model_type == "msfusion":
                    # pred_disps = self.models["depth"](self.models["encoder"](all_color_aug))
                    # prev_disp = self.models["depth"](self.models["encoder"](all_color_aug[0]))
                    cur_disp = self.models["depth"](self.models["encoder"](inputs[("color_aug",0,0)]))
                    prev_disp = self.models["depth"](self.models["encoder"](inputs[("color_aug",1,0)]))
                    disp_inputs = torch.cat([prev_disp[('disp', 0)].repeat(1, 3, 1, 1),
                                             cur_disp[('disp', 0)].repeat(1, 3, 1, 1)], dim=1)
                    axisangle, translation = self.models["pose"]([all_color_aug, disp_inputs])
                    pred_poses.append(from_euler_t(axisangle, translation).cpu().numpy())

        pred_poses = np.concatenate(pred_poses)

        mean_errors, std_errors = compute_atere(self.pose_gt, pred_poses)
        print_string = "POSE:  | ATE: {:.5f} ± {:.5f} | RE: {:.5f} ± {:.5f}"
        print(print_string.format(mean_errors[0], std_errors[0], mean_errors[1], std_errors[1]))
        print('')

        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        save_path = os.path.join(save_folder, "poses.npy")
        np.save(save_path, pred_poses)
        with open(self.test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            '''writer.writerow({'ate_mean': mean_errors[0], 'ate_std': std_errors[0],
                             're_mean': mean_errors[1], 're_std': std_errors[1],
                             'rmse': avg.rmse, 'absrel': avg.absrel, 'mae': avg.mae,
                             'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                             'mse': avg.mse, 'lg10': avg.lg10, 'data_time': avg.data_time, 'gpu_time': avg.gpu_time})'''
            writer.writerow({'ate_mean': mean_errors[0], 'ate_std': std_errors[0],
                            're_mean': mean_errors[1], 're_std': std_errors[1],
                            'rmse': avg.rmse, 'absrel': avg.absrel, 'mae': avg.mae,
                            'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                            'mse': avg.mse, 'lg10': avg.lg10, 'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
        # self.log("val", inputs, outputs, losses)
        del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs, ref_depths):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            scaled_disp, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth
            if self.test != self.epoch:
                print("max:", np.max(depth[0, ...].cpu().detach().numpy()))
                print("min:", np.min(depth[0, ...].cpu().detach().numpy()))
                print("median:", np.median(depth[0, ...].cpu().detach().numpy()))
                self.test = self.epoch

            # # normalize
            # scaled_disp = scaled_disp/(scaled_disp.mean())

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                # cam_points = self.backproject_depth[source_scale](
                #     1 / scaled_disp, inputs[("inv_K", source_scale)])
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords, computed_depth = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                # rgbm
                outputs[("rgbm", frame_id, scale)] = F.grid_sample(
                    torch.cat([inputs[("color", frame_id, source_scale)],
                               inputs[("lnm", frame_id, source_scale)][:, :2, ...]], dim=1),
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                # geometry
                # outputs[("computed_depth", frame_id, scale)] = computed_depth
                # outputs[("projected_depth", frame_id, scale)] = F.grid_sample(
                #     ref_depths["ref_depth", frame_id, scale], pix_coords, padding_mode="border", align_corners=False)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target, weight_mask=None):
        """Computes reprojection loss between a batch of predicted and target images
        """
        # TODO:光度对齐
        # pred2 = brightnes_equator(pred, target)
        # pred = pred2

        # TODO:光线对齐
        m_mask1 = target[:, 4:5, ...] == 0
        m_mask2 = pred[:, 4:5, ...] == 0
        m_mask = m_mask1 * m_mask2

        target = (target / (target[:, 3:4, ...] + 0.001)) * (pred[:, 3:4, ...] + 0.001)  # 防止除零
        target = target[:, :3, ...]
        pred = pred[:, :3, ...]

        abs_diff = torch.abs(target - pred)
        if weight_mask is not None:
            abs_diff = abs_diff * weight_mask
        l1_loss = abs_diff.mean(1, True)
        loss = l1_loss

        # # TODO:尺度不变
        # alpha = torch.log(target) - torch.log(pred)
        # valid_mask = (torch.abs(alpha) < 10000).detach()
        # alpha = alpha[valid_mask]
        # alpha = alpha.mean()
        # sq_diff = torch.sqrt(torch.log(pred) - torch.log(target) + alpha)
        # valid_mask = (torch.abs(sq_diff) < 10000).detach()
        # sq_diff = sq_diff[valid_mask]
        # l2_loss = sq_diff.mean()
        # loss = l2_loss

        if self.opt.no_ssim:
            reprojection_loss = loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * loss

        return reprojection_loss, m_mask

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}

        total_loss = 0
        # total_re_loss = 0
        gt_loss = 0
        gt_pose_loss = 0

        geo_losses = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            m_masks = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]

            # target = inputs[("color", 0, source_scale)]
            target = torch.cat([inputs[("color", 0, source_scale)],
                                inputs[("lnm", 0, source_scale)][:, :2, ...]], dim=1)

            if scale == min(self.opt.scales):
                # depth gt
                gt = inputs[("dgt", 0)]
                # gt_disp = 1 / gt
                pred = outputs[("depth", 0, scale)]

                valid_mask = (gt > 0).detach()
                # diff = gt_disp - disp
                diff = gt - pred
                diff = diff[valid_mask]
                # L1
                gt_loss += diff.abs().mean()

                # gt pose loss (mae)
                angle_loss1 = inputs[('cam_T_cam', 0, -1)][:, :, :3] - outputs[('cam_T_cam', 0, -1)][:, :, :3]
                trans_loss1 = inputs[('cam_T_cam', 0, -1)][:, :, 3:] - outputs[('cam_T_cam', 0, -1)][:, :, 3:]
                pose_loss1 = 100 * angle_loss1.abs().mean() + 100 * trans_loss1.abs().mean()

                angle_loss2 = inputs[('cam_T_cam', 0, 1)][:, :, :3] - outputs[('cam_T_cam', 0, 1)][:, :, :3]
                trans_loss2 = inputs[('cam_T_cam', 0, 1)][:, :, 3:] - outputs[('cam_T_cam', 0, 1)][:, :, 3:]
                pose_loss2 = 100 * angle_loss2.abs().mean() + 100 * trans_loss2.abs().mean()

                gt_pose_loss += (pose_loss1 + pose_loss2)  # TODO: 更换loss的参数

            for frame_id in self.opt.frame_ids[1:]:
                # pred = outputs[("color", frame_id, scale)]
                pred = outputs[("rgbm", frame_id, scale)]
                # geo
                # diff_depth = \
                #     (outputs["computed_depth", frame_id, scale] - outputs["projected_depth", frame_id, scale]).abs() / \
                #     (outputs["computed_depth", frame_id, scale] + outputs["projected_depth", frame_id, scale]).clamp(0, 1)
                # valid_points = outputs["sample", frame_id, scale].abs().max(dim=-1)[0] <= 1
                # geo_mask = valid_points.unsqueeze(1).float()
                # weight_mask = (1 - diff_depth)
                # geo_losses += diff_depth.mean()

                # reprojection_loss = self.compute_reprojection_loss(pred, target, weight_mask)
                # reprojection_loss = self.compute_reprojection_loss(pred, target)
                reprojection_loss, m_mask = self.compute_reprojection_loss(pred, target)
                m_masks.append(m_mask)
                reprojection_losses.append(reprojection_loss)

            reprojection_losses = torch.cat(reprojection_losses, 1)
            m_masks = torch.cat(m_masks, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    # pred = inputs[("color", frame_id, source_scale)]
                    pred = torch.cat([inputs[("color", frame_id, source_scale)],
                                      inputs[("lnm", frame_id, source_scale)][:, :3, ...]], dim=1)
                    temp, _ = self.compute_reprojection_loss(pred, target)
                    identity_reprojection_losses.append(temp)
                    # identity_reprojection_losses.append(
                    #     self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:  # TODO:如何添加mask
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            # loss += to_optimise.mean()
            # TODO:masked重投影
            optimise_mask = m_masks[:, 0, ...] + m_masks[:, 1, ...]
            # optimise_mask = m_masks[:, 0, ...]
            # temp = to_optimise[optimise_mask].sum()
            # temp = optimise_mask.sum()
            masked_reprojection = to_optimise[optimise_mask].sum() / optimise_mask.sum()
            loss += masked_reprojection

            # total_re_loss += to_optimise.mean()  # TODO: 输出重投影损失[暂时弃用]

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        total_loss = 0.01 * total_loss  # 超参

        # gt depth loss
        total_loss += gt_loss

        # pose gt loss
        total_loss += gt_pose_loss

        # TODO: geo loss
        # geo_losses /= self.num_scales
        # total_loss += (geo_losses * 0.1)  # constraint=0.1

        # TODO: silm loss
        if self.opt.pose_model_type == "msfusion":
            silm_loss = 0.0002 * sum([slim_penalty(m).cuda() for m in self.slim_params])
            total_loss += silm_loss

        # total_re_loss /= self.num_scales

        losses["loss"] = total_loss
        # losses["reprojection_loss"] = total_re_loss
        losses["gt_loss"] = gt_loss
        losses["gt_pose_loss"] = gt_pose_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        # depth_pred = torch.clamp(F.interpolate(
        #     depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [320, 320], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs[("dgt", 0)]
        mask = depth_gt > 0

        # TODO: 去除garg/eigen mask
        # garg/eigen crop
        # crop_mask = torch.zeros_like(mask)
        # crop_mask[:, :, 153:371, 44:1197] = 1
        # mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)
        #test-zmm
        #print(depth_gt,depth_pred)
        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} || batch {:>6} || examples/s: {:5.1f}" + \
                       " || loss: {:.5f} || time elapsed: {} || time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
