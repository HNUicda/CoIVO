import dataloaders.transforms as transforms
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
import torch.utils.data as data
import torch
import h5py
import numpy as np
from imageio import imread
from path import Path
import matplotlib.image as mping

import matplotlib.pyplot
matplotlib.pyplot.switch_backend('agg')


to_tensor = transforms.ToTensor()


def h5_loader(path, index):  # 目前只能读取最前排的scenes里的文件
    path = Path(path[0])
    path = path / 'fusion_data.hdf5'
    h5f = h5py.File(path, "r")
    # rgb = np.array(h5f['color'][index])
    rgb = h5f['color'][index]
    rgb = rgb.astype(np.float32)
    rgb = rgb[:, :, (2, 1, 0)]
    # depth = np.array(h5f['mean_depth'][index])
    depth = h5f['mean_depth'][index]
    depth = depth.squeeze().astype(np.float32)
    return rgb, depth


class MyDataloader(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd']  # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    # 无效变量：loader
    def __init__(self, root, type, sparsifier=None, modality='rgb'):

        # 新初始化数据库（改自sequence_folders）
        self.root = Path(root)

        # 读取mask
        path = self.root / 'temp' / 'fusion_data1.hdf5'
        h5f = h5py.File(path, "r")
        print(h5f.keys())
        self.mask = h5f['mask'][0].squeeze(2)
        h5f.close()

        if type == 'train':
            scene_list_path = self.root/'train.txt'
            self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
            self.transform = self.train_transform
            path = self.scenes[0] / 'fusion_data.hdf5'
            h5f = h5py.File(path, "r")
            self.lenc = len(h5f['color'])
        elif type == 'val':
            scene_list_path = self.root/'val.txt'
            self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
            self.transform = self.val_transform
            path = self.scenes[0] / 'fusion_data.hdf5'
            h5f = h5py.File(path, "r")
            self.lenc = len(h5f['color'])
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))

        self.sparsifier = sparsifier

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                                  "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform(self, rgb, depth, mask):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(self, rgb, depth, mask):
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

        rgb, depth = h5_loader(self.scenes, index)

        if self.transform is not None:
            rgb_np, depth_np, mask_np = self.transform(rgb, depth, self.mask)
        else:
            raise(RuntimeError("transform not defined"))

        mask_tensor = to_tensor(mask_np)
        mask_tensor = mask_tensor.unsqueeze(0)

        # color normalization
        # rgb_tensor = normalize_rgb(rgb_tensor)
        # rgb_np = normalize_np(rgb_np)

        if self.modality == 'rgb':
            input_np = rgb_np
        elif self.modality == 'rgbd':
            input_np = self.create_rgbd(rgb_np, depth_np)
        elif self.modality == 'd':
            input_np = self.create_sparse_depth(rgb_np, depth_np)

        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor, mask_tensor

    def __len__(self):
        return self.lenc


class hDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(hDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (256, 320)
        self.iheight = 256  # image height

    # TODO: 关闭transform
    def train_transform(self, rgb, depth, mask):
        s = np.random.uniform(1.0, 1.5)  # random scaling

        depth_np = depth / s
        # depth_np = depth

        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(300.0 / self.iheight),  # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np)  # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)
        mask_np = transform(mask)

        # rgb_np = np.asfarray(rgb, dtype='float') / 255
        # depth_np = depth

        return rgb_np, depth_np, mask_np

    def val_transform(self, rgb, depth, mask):
        depth_np = depth
        transform = transforms.Compose([
            # transforms.Resize(300.0 / self.iheight),
            transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)
        mask_np = transform(mask)

        # rgb_np = np.asfarray(rgb, dtype='float') / 255

        return rgb_np, depth_np, mask_np