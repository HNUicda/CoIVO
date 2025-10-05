import dataloaders.transforms as transforms
from torchvision import transforms as transforms_tv

from scipy.spatial.transform import Rotation as R

from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
import torch.utils.data as data
import torch
import numpy as np
from imageio import imread
from path import Path
import matplotlib.image as mping
from PIL import Image

from .dense_to_sparse import UniformSampling as uar
import matplotlib.pyplot
import cv2
from PDutils import get_csd_pose, get_csv_pose

from os import environ
environ['OPENCV_IO_ENABLE_OPENEXR'] = "true"

matplotlib.pyplot.switch_backend('agg')

to_tensor = transforms.ToTensor()


def load_255(path):
    temp = imread(path).astype(np.float32)
    # temp = imread(path)
    temp = temp[:, :, :3] / 255
    return temp


def load_float(path):
    temp = mping.imread(path).astype(np.float32)
    temp = temp[:, :, 0]
    # temp = 0.02 / (2 - 1.99 * (1 - temp))
    # temp = 2 / (1 + 199 * temp)
    # temp = (1 + 199 * temp) / 2
    temp = 0.5 + (100 - 0.5) * temp  # or
    # temp = (2 - 1.99 * (1 - temp)) / 0.02  # inv_depth
    return temp


def load_exr(path):
    temp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    temp = temp[:, :, 0]  # disp
    return temp


def quaternion2matrix(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    r = R.from_euler('xyz', euler, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix


class MyDataloader(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd']  # , 'g', 'gd'
    data_type_names = ['png', 'exr']
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    # 无效变量：loader
    def __init__(self, root, type, height, width, frame_idxs, num_scales,
                 sparsifier=None, modality='rgb', data_type='png'):

        # 新初始化数据库（改自sequence_folders）
        self.root = Path(root)

        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.frame_idxs = frame_idxs  # [0, -1, 1]

        self.type = type
        if self.type == 'train':
            scene_list_path = self.root / 'train.txt'
            self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
            self.transform = self.train_transform
            # self.s_num = START_NUM_TARIN  # 2000
        elif self.type == 'val':
            scene_list_path = self.root / 'val.txt'
            self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
            self.transform = self.val_transform
            # self.s_num = START_NUM_VAL  # 10
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                                                  "Supported dataset types are: train, val"))
        # self.loader = loader
        # self.sparsifier = uar(1000)

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                                  "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

        assert (data_type in self.data_type_names), "Invalid data type: " + data_type + "\n" + \
                                                    "Supported data types are: " + ''.join(self.data_type_names)
        self.data_type = data_type
        self.crawl_folders()

        # self.interp = Image.ANTIALIAS
        # self.resize = {}
        # for i in range(self.num_scales):
        #     s = 2 ** i
        #     self.resize[i] = transforms_tv.Resize((self.height // s, self.width // s), interpolation=self.interp)

        # self.to_sensor = to_tensor

    def crawl_folders(self):
        sequence_set = []
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene / 'cam.txt').astype(np.float32).reshape((3, 3))
            # intrinsics = np.append(intrinsics, np.zeros((3, 1)), axis=1)
            imgs = sorted(scene.files('*.png'))
            depth_path = scene / 'GT'

            translations = []
            rotations = []
            if self.data_type == 'png':
                depths = sorted(depth_path.files('*.png'))
                translations, rotations, _ = get_csv_pose(scene, self.type)
            elif self.data_type == 'exr':
                depths = sorted(depth_path.files('*.exr'))
                translations, rotations, _ = get_csd_pose(scene, True)

            # pose_count = 0
            # translation_list = []
            # rotation_list = []
            # # with open(self.root + "/position_rotation.csv", 'r') as f_position:  # smalldata mode
            # with open(scene + "/position_rotation.csv", 'r') as f_position:  # seq mode
            #     f_position_reader = np.loadtxt(f_position, delimiter=',', skiprows=1)
            #     for row in f_position_reader:
            #         if pose_count >= self.s_num:
            #             translation_vector = row[0:3]
            #             rotation_quaternion = row[3:7]
            #             translation_list.append(translation_vector)
            #             rotation_list.append(rotation_quaternion)
            #         pose_count = pose_count + 1

            for i in range(-min(self.frame_idxs), len(imgs) - max(self.frame_idxs)):  # 掐头去尾
                tgts = []
                gts = []
                gt_rota = []
                gt_tran = []
                poses = []
                for j in self.frame_idxs:
                    tgts.append(imgs[i + j])
                    gts.append(depths[i + j])
                    if i + j != i:
                        t_vec = translations[i] - translations[i + j]
                        # t_vec = np.expand_dims(t_vec, 1).reshape(-1, 3)
                        gt_tran.append(t_vec)

                        # r_mat1 = quaternion2matrix(rotation_list[i])
                        # r_mat2 = quaternion2matrix(rotation_list[i + j])
                        r_mat1 = rotations[i]
                        r_mat2 = rotations[i + j]
                        r_mat2 = np.linalg.inv(r_mat2)
                        r_mat = np.dot(r_mat1, r_mat2)
                        r_vec, _ = cv2.Rodrigues(r_mat)
                        # r_vec = np.expand_dims(r_vec, 1).reshape(-1, 3)
                        r_vec = r_vec.squeeze()
                        gt_rota.append(r_vec)

                        poses.append(np.concatenate([t_vec, r_vec]))

                sample = {'tgt': tgts, 'GT': gts, 'gt_rota': gt_rota, 'gt_tran': gt_tran,
                          'poses': poses, 'intrinsics': intrinsics}
                sequence_set.append(sample)
        self.imgs = sequence_set

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(self, rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    # def create_sparse_depth(self, rgb, depth):
    #     if self.sparsifier is None:
    #         return depth
    #     else:
    #         mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
    #         sparse_depth = np.zeros(depth.shape)
    #         sparse_depth[mask_keep] = depth[mask_keep]
    #         return sparse_depth

    # def create_rgbd(self, rgb, depth):
    #     sparse_depth = self.create_sparse_depth(rgb, depth)
    #     rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
    #     return rgbd

    # def preprocess(self, inputs, color_aug):
    #     """Resize colour images to the required scales and augment if required
    #
    #     We create the color_aug object in advance and apply the same augmentation to all
    #     images in this item. This ensures that all images input to the pose network receive the
    #     same augmentation.
    #     """
    #     for k in list(inputs):
    #         # frame = inputs[k]
    #         if "color" in k:
    #             n, im, i = k
    #             for i in range(self.num_scales):
    #                 inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
    #         if "rgbsd" in k:
    #             n, im, i = k
    #             for i in range(self.num_scales):
    #                 inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
    #
    #     for k in list(inputs):
    #         f = inputs[k]
    #         if "color" in k:
    #             n, im, i = k
    #             # inputs[(n, im, i)] = self.to_tensor(f)
    #             # inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
    #             inputs[(n, im, i)] = to_tensor(np.asarray(f))
    #             inputs[(n + "_aug", im, i)] = to_tensor(np.asarray(color_aug(f)))
    #         if "rgbsd" in k:
    #             n, im, i = k
    #             # inputs[(n, im, i)] = self.to_tensor(f)
    #             # inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
    #             inputs[(n, im, i)] = to_tensor(np.asarray(f))
    #             inputs[(n + "_aug", im, i)] = to_tensor(np.asarray(color_aug(f)))

    def __getitem__(self, index):
        def to_tensor(images):
            tensors = []
            for im in images:
                # put it from HWC to CHW format
                im = np.transpose(im, (2, 0, 1))
                # handle numpy array
                tensors.append(torch.from_numpy(im).float())
            return tensors

        sample = self.imgs[index]
        rgbs = to_tensor([load_255(rgb) for rgb in sample['tgt']])
        if self.data_type == 'png':
            depths = to_tensor([np.expand_dims(load_float(depth), axis=2) for depth in sample['GT']])
        elif self.data_type == 'exr':
            depths = to_tensor([np.expand_dims(load_exr(depth), axis=2) for depth in sample['GT']])

        tgt_img = rgbs[0]
        ref_imgs = rgbs[1:]
        tgt_img_gt = depths[0].squeeze()
        ref_img_gts = depths[1:]
        intrinsics = np.copy(sample['intrinsics'])
        intrinsics_inv = torch.from_numpy(np.linalg.inv(intrinsics))

        # axi_gt = torch.from_numpy(sample['gt_rota'].copy()).float()
        # tra_gt = torch.from_numpy(sample['gt_tran'].copy()).float()

        poses_gt = [torch.from_numpy(pose) for pose in sample['poses']]

        return tgt_img, ref_imgs, torch.from_numpy(intrinsics), intrinsics_inv, tgt_img_gt, ref_img_gts, poses_gt

        # inputs = {}

        # for i in range(len(self.frame_idxs)):
        #     rgb_np = load_255(sample['tgt'][i])
        #     # if self.transform is not None:
        #     #     temp, _ = self.transform(temp, depth)
        #     # else:
        #     #     raise (RuntimeError("transform not defined"))
        #     inputs[("color", self.frame_idxs[i], -1)] = Image.fromarray(rgb_np)
        #     gt_np = load_float(sample['GT'][i])
        #     depth_tensor = to_tensor(gt_np)
        #     depth_tensor = depth_tensor.unsqueeze(0)
        #     inputs[("dgt", self.frame_idxs[i])] = depth_tensor
        #     # rgbsd_np = self.create_rgbd(rgb_np, gt_np)  # 暂时不乘255
        #     # inputs[("rgbsd", self.frame_idxs[i], -1)] = Image.fromarray(rgbsd_np.astype(np.uint8))

        # color_aug = (lambda x: x)

        # self.preprocess(inputs, color_aug)

        # for i in self.frame_idxs:
        #     del inputs[("color", i, -1)]
        #     del inputs[("color_aug", i, -1)]
        #     del inputs[("rgbsd", i, -1)]

        # for scale in range(self.num_scales):
        #     K = sample['intrinsics'].copy()
        #
        #     K[0, :] *= self.width // (2 ** scale)
        #     K[1, :] *= self.height // (2 ** scale)
        #
        #     inv_K = np.linalg.pinv(K)
        #
        #     inputs[("K", scale)] = torch.from_numpy(K)
        #     inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # if self.transform is not None:
        #     rgb_np, depth_np = self.transform(rgb, depth)
        # else:
        #     raise (RuntimeError("transform not defined"))

        # color normalization
        # rgb_tensor = normalize_rgb(rgb_tensor)
        # rgb_np = normalize_np(rgb_np)

        # if self.modality == 'rgb':
        #     input_np = rgb_np
        # elif self.modality == 'rgbd':
        #     input_np = self.create_rgbd(rgb_np, depth_np)
        # elif self.modality == 'd':
        #     input_np = self.create_sparse_depth(rgb_np, depth_np)

        # input_tensor = to_tensor(input_np)
        # while input_tensor.dim() < 3:
        #     input_tensor = input_tensor.unsqueeze(0)
        # depth_tensor = to_tensor(depth_np)
        # depth_tensor = depth_tensor.unsqueeze(0)

        # depths_tensor = []
        # for depth in depths:
        #     depth_tensor = to_tensor(depth)
        #     depth_tensor = depth_tensor.unsqueeze(0)
        #     depths_tensor.append(depth_tensor)

        # inputs
        # inputs["depth_gt"] = depths_tensor
        # inputs["axi_gt"] = torch.from_numpy(sample['gt_rota'].copy()).float()
        # inputs["tra_gt"] = torch.from_numpy(sample['gt_tran'].copy()).float()

        # return input_tensor, depth_tensor, (depth_tensor > 0)
        # return inputs

    def __len__(self):
        return len(self.imgs)


class aDataset(MyDataloader):
    def __init__(self, root, type, height=320, width=320, frame_idxs=None, num_scales=1,
                 sparsifier=None, modality='rgb', data_type='png'):
        if frame_idxs is None:
            frame_idxs = [0, -1, 1]
        super(aDataset, self).__init__(root, type, height, width, frame_idxs, num_scales,
                                       sparsifier=sparsifier, modality=modality, data_type=data_type)
        self.output_size = (320, 320)
        self.iheight = 320  # image height

    # TODO: 关闭transform
    def train_transform(self, rgb, depth):
        # s = np.random.uniform(1.0, 1.5)  # random scaling
        # depth_np = depth / s
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

        # rgb_np = np.asfarray(rgb, dtype='float') / 255
        rgb_np = np.asfarray(rgb, dtype='float')
        depth_np = depth

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth

        # transform = transforms.Compose([
        #     # transforms.Resize(240.0 / self.iheight),
        #     transforms.CenterCrop(self.output_size),
        # ])
        # rgb_np = transform(rgb)
        # rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        # depth_np = transform(depth_np)

        # rgb_np = np.asfarray(rgb, dtype='float') / 255
        rgb_np = np.asfarray(rgb, dtype='float')

        return rgb_np, depth_np
