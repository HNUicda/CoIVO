"""
from GTLN-Centos
"""
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
from scipy.spatial.transform import Rotation as R
import re

from .dense_to_sparse import UniformSampling as uar
import cv2
from os import environ
environ['OPENCV_IO_ENABLE_OPENEXR'] = "true"

import matplotlib.pyplot
matplotlib.pyplot.switch_backend('agg')


to_tensor = transforms.ToTensor()


def load_255(path):
    # temp = imread(path).astype(np.float32)
    temp = imread(path)
    temp = temp[:, :, :3]
    return temp


def load_exr(path):
    temp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # print(temp)
    # print(temp.shape)
    temp = temp[:, :, 2]
    # temp = np.expand_dims(temp, axis=2)
    # print(temp)
    # print(temp.shape)

    return temp


def get_csd_pose(txt_path1, txt_path2=None):
    """
    load txt：
    Camera Position Data.txt
    Camera Quaternion Rotation Data.txt
    ->> translation，rotation, pose
    """
    # 读取
    txt1 = []
    txt2 = []
    if not isinstance(txt_path1, Path):
        txt_path1 = Path(txt_path1)
    txt_path1 = txt_path1 / 'Camera Position Data.txt'
    with open(txt_path1, "r") as f:
        txt1 = f.readlines()
    if txt_path2 is not None:
        if not isinstance(txt_path2, Path):
            txt_path2 = Path(txt_path2)
        txt_path2 = txt_path2 / 'Camera Quaternion Rotation Data.txt'
        with open(txt_path2, "r") as f:
            txt2 = f.readlines()

    translations = []
    rots = []
    poses = []
    loc_x = re.search(r'X=(-?[\d.]+)', txt1[0])
    loc_y = re.search(r'Y=(-?[\d.]+)', txt1[0])
    loc_z = re.search(r'Z=(-?[\d.]+)', txt1[0])
    x = float(loc_x.group(1))
    y = float(loc_y.group(1))
    z = float(loc_z.group(1))
    init_location = np.array([x, y, z])
    for i in range(len(txt1)):
        # translation
        loc_x = re.search(r'X=(-?[\d.]+)', txt1[i])
        loc_y = re.search(r'Y=(-?[\d.]+)', txt1[i])
        loc_z = re.search(r'Z=(-?[\d.]+)', txt1[i])
        x = float(loc_x.group(1))
        y = float(loc_y.group(1))
        z = float(loc_z.group(1))
        location = np.array([x, y, z])
        translations.append(location - init_location)

        if txt_path2 is not None:
            # rotation
            rot_x = re.search(r'X=(-?[\d.]+)', txt2[i])
            rot_y = re.search(r'Y=(-?[\d.]+)', txt2[i])
            rot_z = re.search(r'Z=(-?[\d.]+)', txt2[i])
            rot_w = re.search(r'W=(-?[\d.]+)', txt2[i])
            x = float(rot_x.group(1))
            y = float(rot_y.group(1))
            z = float(rot_z.group(1))
            w = float(rot_w.group(1))
            r = R.from_quat(np.array([x, y, z, w])).as_matrix()
            rots.append(r)

            # pose
            poses.append(np.concatenate([r, np.expand_dims(location, 1)], 1))

    return [np.array(translations), np.array(rots), np.array(poses)]


class MyDataloader(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd']  # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, type, height, width, frame_idxs, num_scales, modality='rgb'):

        # 新初始化数据库（改自sequence_folders）
        self.root = Path(root)

        self.height = height
        self.width = width
        self.frame_idxs = frame_idxs
        self.num_scales = num_scales
        self.gt_poses = []

        if type == 'train':
            scene_list_path = self.root/'train.txt'
            self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
            self.transform = self.train_transform
            self.s_num = 0
        elif type == 'val':
            scene_list_path = self.root/'val.txt'
            self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
            self.transform = self.val_transform
            self.s_num = 0
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))

        self.crawl_folders()
        # self.sparsifier = uar(1000)
        self.sparsifier = None

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                                  "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

        # self.interp = Image.ANTIALIAS
        self.interp = Image.LANCZOS  # pillow 10.0.0
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms_tv.Resize((self.height // s, self.width // s), interpolation=self.interp)

    def crawl_folders(self):
        sequence_set = []
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene / 'cam.txt').astype(np.float32).reshape((3, 3))
            intrinsics = np.append(intrinsics, np.zeros((3, 1)), axis=1)
            imgs = sorted(scene.files('*.png'))
            depth_path = scene / 'GT'
            depths = sorted(depth_path.files('*.exr'))
            m_path = scene / 'm'
            m_maps = sorted(m_path.files('*.png'))

            pose_count = 0
            translations, rotations, poses = get_csd_pose(scene, scene)
            translations = translations[pose_count:]
            rotations = rotations[pose_count:]
            for i in range(-min(self.frame_idxs), len(imgs) - max(self.frame_idxs)):
                tgts = []
                gts = []
                lnms = []
                gt_rota = []
                gt_tran = []
                for j in self.frame_idxs:
                    tgts.append(imgs[i + j])
                    gts.append(depths[i + j])
                    lnms.append(m_maps[i + j])
                    if i + j != i:
                        t_vec = translations[i] - translations[i + j]
                        t_vec = np.expand_dims(t_vec, 1).reshape(-1, 3)
                        gt_tran.append(t_vec)

                        r_mat1 = rotations[i]
                        r_mat2 = rotations[i + j]
                        r_mat2 = np.linalg.inv(r_mat2)
                        r_mat = np.dot(r_mat1, r_mat2)
                        # r_vec, _ = cv2.Rodrigues(r_mat)
                        r_vec = R.from_matrix(r_mat).as_rotvec()
                        r_vec = np.expand_dims(r_vec, 1).reshape(-1, 3)
                        gt_rota.append(r_vec)

                sample = {'tgt': tgts, 'GT': gts, 'lnm': lnms,
                          'gt_rota': np.array(gt_rota), 'gt_tran': np.array(gt_tran),
                          'intrinsics': intrinsics}
                sequence_set.append(sample)
            self.gt_poses.append(poses)
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
            if "rgbm" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            if "lnm" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

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
            if "rgbm" in k:
                n, im, i = k
                # inputs[(n, im, i)] = self.to_tensor(f)
                # inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
                inputs[(n, im, i)] = to_tensor(np.asarray(f))
                inputs[(n + "_aug", im, i)] = to_tensor(np.asarray(color_aug(f)))
            if "lnm" in k:
                n, im, i = k
                # inputs[(n, im, i)] = self.to_tensor(f)
                # inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
                inputs[(n, im, i)] = to_tensor(np.asarray(f))
                inputs[(n + "_aug", im, i)] = to_tensor(np.asarray(color_aug(f)))

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
        # rgbs = [load_255(i) for i in sample['tgt']]
        # depth_np = load_exr(sample['GTs'][0])
        inputs = {}

        for i in range(len(self.frame_idxs)):
            # gren RGB
            rgb_np = load_255(sample['tgt'][i])
            inputs[("color", self.frame_idxs[i], -1)] = Image.fromarray(rgb_np)

            # gren GT
            gt_np = load_exr(sample['GT'][i])
            depth_tensor = to_tensor(gt_np)
            depth_tensor = depth_tensor.unsqueeze(0)
            inputs[("dgt", self.frame_idxs[i])] = depth_tensor

            # gren m
            m_np = load_255(sample['lnm'][i])
            inputs[("lnm", self.frame_idxs[i], -1)] = Image.fromarray(m_np)

        color_aug = (lambda x: x)
        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            # del inputs[("rgbsd", i, -1)]
            # del inputs[("rgbm", i, -1)]

        for scale in range(self.num_scales):
            K = sample['intrinsics'].copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

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

        # tensors = [to_tensor(i) for i in rgbs]
        # input_tensor = [tensors[0], [tensors[1], tensors[2]]]

        # while input_tensor.dim() < 3:
        #     input_tensor = input_tensor.unsqueeze(0)
        # depth_tensor = to_tensor(depth_np)
        # depth_tensor = depth_tensor.unsqueeze(0)

        inputs["axi_gt"] = torch.from_numpy(sample['gt_rota'].copy()).float()
        inputs["tra_gt"] = torch.from_numpy(sample['gt_tran'].copy()).float()

        # return input_tensor, depth_tensor
        return inputs

    def __len__(self):
        return len(self.imgs)


class aDataset(MyDataloader):
    def __init__(self, root, type, height, width, frame_idxs, num_scales, modality='rgb'):
        super(aDataset, self).__init__(root, type, height, width, frame_idxs, num_scales, modality=modality)
        self.output_size = (320, 320)
        self.iheight = 320  # image height

    # TODO: close transform
    def train_transform(self, rgb, depth):
        rgb_np = np.asfarray(rgb, dtype='float')
        depth_np = depth

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        rgb_np = np.asfarray(rgb, dtype='float')

        return rgb_np, depth_np