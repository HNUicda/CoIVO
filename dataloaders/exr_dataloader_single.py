

import dataloaders.transforms as transforms
from torchvision import transforms as transforms_tv
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
import torch.utils.data as data
import torch
import numpy as np
from imageio import imread
from path import Path
from PIL import Image
import matplotlib.image as mping



import os
#from os import environ
#os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "true"
#environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
import cv2
#import imageio
import OpenEXR
import matplotlib.pyplot
matplotlib.pyplot.switch_backend('agg')


to_tensor = transforms.ToTensor()


def load_255(path):
    temp = imread(path).astype(np.float32)
    temp = temp[:, :, :3]
    return temp


def load_exr(path):
    #print("test"+path)
    #opencv读不出来exr
    temp = cv2.imread(path, cv2.IMREAD_UNCHANGED) #包含alpha通道
    #temp=imageio.imread(path, 'exr')
    # print(temp)
    #print(temp.shape)

    temp = temp[:, :, 0]  # disp
    # temp = np.expand_dims(temp, axis=2)
    # print(temp)
    # print(temp.shape)

    return temp


class MyDataloader(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd']  # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    # 无效变量：loader
    def __init__(self, root, type, sparsifier=None, modality='rgb', loader=None):

        # 新初始化数据库（改自sequence_folders）
        self.root = Path(root)
        self.frame_idxs = [0, -1, 1]
        self.num_scales = 4
        self.height = 320
        self.width = 320

        if type == 'train':
            scene_list_path = self.root/'train.txt'
            self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
            self.transform = self.train_transform
        elif type == 'val':
            scene_list_path = self.root/'val.txt'
            self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))

        self.crawl_folders()
        self.loader = loader
        self.sparsifier = sparsifier

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                                  "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

        self.interp = Image.ANTIALIAS
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms_tv.Resize((self.height // s, self.width // s), interpolation=self.interp)

    def crawl_folders(self):
        sequence_set = []
        for scene in self.scenes:
            # intrinsics = np.genfromtxt(scene / 'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.png'))
            depth_path = scene / 'GT'
            depths = sorted(depth_path.files('*.exr'))

            for i in range(-min(self.frame_idxs), len(imgs) - max(self.frame_idxs)):
                tgts = []
                gts = []
                for j in self.frame_idxs:
                    tgts.append(imgs[i + j])
                    gts.append(depths[i + j])
                sample = {'tgt': tgts, 'GTs': gts}
                sequence_set.append(sample)

            # for i in range(len(imgs)):
            #     sample = {'tgt': imgs[i], 'GT': depths[i]}
            #     sequence_set.append(sample)
        self.imgs = sequence_set

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            # frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            # if "rgbsd" in k:
            #     n, im, i = k
            #     for i in range(self.num_scales):
            #         inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            # if "rgbm" in k:
            #     n, im, i = k
            #     for i in range(self.num_scales):
            #         inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            # if "lnm" in k:
            #     n, im, i = k
            #     for i in range(self.num_scales):
            #         inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                # inputs[(n, im, i)] = self.to_tensor(f)
                # inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
                inputs[(n, im, i)] = to_tensor(np.asarray(f))
                inputs[(n + "_aug", im, i)] = to_tensor(np.asarray(color_aug(f)))
            # if "rgbsd" in k:
            #     n, im, i = k
            #     # inputs[(n, im, i)] = self.to_tensor(f)
            #     # inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            #     inputs[(n, im, i)] = to_tensor(np.asarray(f))
            #     inputs[(n + "_aug", im, i)] = to_tensor(np.asarray(color_aug(f)))
            # if "rgbm" in k:
            #     n, im, i = k
            #     # inputs[(n, im, i)] = self.to_tensor(f)
            #     # inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            #     inputs[(n, im, i)] = to_tensor(np.asarray(f))
            #     inputs[(n + "_aug", im, i)] = to_tensor(np.asarray(color_aug(f)))
            # if "lnm" in k:
            #     n, im, i = k
            #     # inputs[(n, im, i)] = self.to_tensor(f)
            #     # inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            #     inputs[(n, im, i)] = to_tensor(np.asarray(f))
            #     inputs[(n + "_aug", im, i)] = to_tensor(np.asarray(color_aug(f)))

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(self, rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def create_sparse_depth(self, rgb, depth):
        if self.sparsifier is None:
            return depth
        else:
            mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
            sparse_depth = np.zeros(depth.shape)
            sparse_depth[mask_keep] = depth[mask_keep]
            return sparse_depth

    def create_rgbd(self, rgb, depth):
        sparse_depth = self.create_sparse_depth(rgb, depth)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd

    def __getitem__(self, index):
        sample = self.imgs[index]
        rgbs = [load_255(i) for i in sample['tgt']]
        depth_np = load_exr(sample['GTs'][0])
        # inputs = {}

        # for i in range(len(self.frame_idxs)):
        #     # 生成RGB
        #     rgb_np = load_255(sample['tgt'][i])
        #     inputs[("color", self.frame_idxs[i], -1)] = Image.fromarray(rgb_np)
        #
        #     # 生成GT
        #     gt_np = load_exr(sample['GT'][i])
        #     depth_tensor = to_tensor(gt_np)
        #     depth_tensor = depth_tensor.unsqueeze(0)
        #     inputs[("dgt", self.frame_idxs[i])] = depth_tensor
        #
        # color_aug = (lambda x: x)
        # self.preprocess(inputs, color_aug)

        # for i in self.frame_idxs:
        #     del inputs[("color", i, -1)]
        #     del inputs[("color_aug", i, -1)]
        #     # del inputs[("rgbsd", i, -1)]
        #     # del inputs[("rgbm", i, -1)]

        # if self.transform is not None:
        #     rgb_np, depth_np = self.transform(rgb, depth)
        # else:
        #     raise(RuntimeError("transform not defined"))

        # color normalization
        # rgb_tensor = normalize_rgb(rgb_tensor)
        # rgb_np = normalize_np(rgb_np)

        # if self.modality == 'rgb':
        #     input_np = rgb_np
        # elif self.modality == 'rgbd':
        #     input_np = self.create_rgbd(rgb_np, depth_np)
        # elif self.modality == 'd':
        #     input_np = self.create_sparse_depth(rgb_np, depth_np)

        tensors = [to_tensor(i) for i in rgbs]
        input_tensor = [tensors[0], [tensors[1], tensors[2]]]

        # while input_tensor.dim() < 3:
        #     input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor
        # return inputs

    def __len__(self):
        return len(self.imgs)


class aDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(aDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (320, 320)
        self.iheight = 320  # image height

    # TODO: 关闭transform
    def train_transform(self, rgb, depth):
        rgb_np = np.asfarray(rgb, dtype='float')
        depth_np = depth

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        rgb_np = np.asfarray(rgb, dtype='float')

        return rgb_np, depth_np