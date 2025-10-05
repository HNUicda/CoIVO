
import dataloaders.transforms as transforms
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
import torch.utils.data as data
import torch
import numpy as np
from imageio import imread
from path import Path
import matplotlib.image as mping

import matplotlib.pyplot
matplotlib.pyplot.switch_backend('agg')


to_tensor = transforms.ToTensor()


def load_255(path):
    temp = imread(path).astype(np.float32)
    temp = temp[:, :, :3]
    return temp


def load_gt(path):
    temp = mping.imread(path).astype(np.float32)
    if temp.ndim == 3:
        temp = temp[:, :, 0]
        temp = 0.02 / (2 - 1.99 * (1 - temp))
    return temp


def load_dso(path):
    temp = mping.imread(path).astype(np.float32)
    if temp.ndim == 3:
        temp = temp[:, :, 0]
        temp = 1 / temp
    return temp


class MyDataloader(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd']  # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    # 无效变量：loader
    def __init__(self, root, type, sparsifier=None, modality='rgb', loader=None, with_gt=True):

        # 新初始化数据库（改自sequence_folders）
        self.root = Path(root)

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

        self.with_gt = with_gt

        self.crawl_folders()
        self.loader = loader
        self.sparsifier = sparsifier

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                                  "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality


    def crawl_folders(self):
        sequence_set = []
        for scene in self.scenes:
            # intrinsics = np.genfromtxt(scene / 'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.png'))
            if self.with_gt:
                # n_data, type_data = str(scene).split('/')[3:]
                # if n_data == 'data3':
                #     n_data = 'colon221020'
                # depth_path = scene.parent.parent.parent / n_data / type_data / 'GT'
                #
                # depths = [depth_path / i.name for i in imgs]  # 获取与rgb一一对应的Ground Truth

                depth_path = scene / 'GT'
                depths = sorted(depth_path.files('*.png'))

                dso_path = scene / 'DSO'
                dso = sorted(dso_path.files('*.png'))  # DSO

                for i in range(len(imgs)):
                    sample = {'tgt': imgs[i], 'GT': depths[i], 'dso': dso[i]}
                    sequence_set.append(sample)
            else:
                dso_path = scene / 'DSO'
                dso = sorted(dso_path.files('*.png'))  # DSO

                for i in range(len(imgs)):
                    sample = {'tgt': imgs[i], 'dso': dso[i]}
                    sequence_set.append(sample)

        self.imgs = sequence_set

    def train_transform(self, rgb, depth, dso):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(self, rgb, depth, dso):
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
        rgb = load_255(sample['tgt'])
        dso = load_dso(sample['dso'])

        if self.with_gt:
            depth = load_gt(sample['GT'])
            if self.transform is not None:
                rgb_np, depth_np, dso_np = self.transform(rgb, depth, dso)
            else:
                raise(RuntimeError("transform not defined"))
        else:
            depth = np.ones(dso.shape)
            if self.transform is not None:
                rgb_np, depth_np, dso_np = self.transform(rgb, depth, dso)
            else:
                raise(RuntimeError("transform not defined"))

        # color normalization
        # rgb_tensor = normalize_rgb(rgb_tensor)
        # rgb_np = normalize_np(rgb_np)

        if self.modality == 'rgb':
            input_np = rgb_np
        elif self.modality == 'rgbd':
            input_np = np.append(rgb_np, np.expand_dims(dso_np, axis=2), axis=2)
        # elif self.modality == 'rgbd':
        #     input_np = self.create_rgbd(rgb_np, depth_np)
        # elif self.modality == 'd':
        #     input_np = self.create_sparse_depth(rgb_np, depth_np)

        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor, (depth_tensor > 0)

    def __len__(self):
        return len(self.imgs)


class aDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb', with_gt=True):
        super(aDataset, self).__init__(root, type, sparsifier, modality, with_gt)
        self.output_size = (320, 320)
        self.iheight = 320  # image height

    # TODO: 关闭transform
    def train_transform(self, rgb, depth, dso):
        # s = np.random.uniform(1.0, 1.5)  # random scaling
        # depth_np = depth / s
        # dso_np = dso / s
        # angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        # do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
        #
        # # perform 1st step of data augmentation
        # transform = transforms.Compose([
        #     # transforms.Resize(250.0 / self.iheight), this is for computational efficiency, since rotation can be slow
        #     transforms.Rotate(angle),
        #     transforms.Resize(s),
        #     transforms.CenterCrop(self.output_size),
        #     transforms.HorizontalFlip(do_flip)
        # ])
        # rgb_np = transform(rgb)
        # rgb_np = self.color_jitter(rgb_np)  # random color jittering
        # rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        # depth_np = transform(depth_np)

        rgb_np = np.asfarray(rgb, dtype='float') / 255
        depth_np = depth
        dso_np = dso

        return rgb_np, depth_np, dso_np

    def val_transform(self, rgb, depth, dso):
        depth_np = depth
        dso_np = dso

        # transform = transforms.Compose([
        #     # transforms.Resize(240.0 / self.iheight),
        #     transforms.CenterCrop(self.output_size),
        # ])
        # rgb_np = transform(rgb)
        # rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        # depth_np = transform(depth_np)
        # dso_np = transform(dso_np)

        rgb_np = np.asfarray(rgb, dtype='float') / 255

        return rgb_np, depth_np, dso_np