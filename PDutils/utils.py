import re
import numpy as np
from scipy.spatial.transform import Rotation as R
from path import Path
from .DEFINE import *


def get_csd_pose(txt_path, has_rotation=False):
    """
    load txt：
    Camera Position Data.txt
    Camera Quaternion Rotation Data.txt
    ->> translation，rotation, pose
    """
    # load
    # txt1 = []
    # txt2 = []
    if not isinstance(txt_path, Path):
        txt_path = Path(txt_path)
    txt_path1 = txt_path / 'Camera Position Data.txt'
    with open(txt_path1, "r") as f:
        txt1 = f.readlines()
    if has_rotation:
        txt_path2 = txt_path / 'Camera Quaternion Rotation Data.txt'
        with open(txt_path2, "r") as f:
            txt2 = f.readlines()

    translations = []
    rotations = []
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

        if has_rotation:
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
            rotations.append(r)

            # pose
            poses.append(np.concatenate([r, np.expand_dims(location, 1)], 1))

    return [np.array(translations), np.array(rotations), np.array(poses)]


def get_csv_pose(csv_poses_path, dataset_type):
    """
    load gt pose
    input: csv_poses_path: str
           dataset_type: str in ['train', 'val']
    output: locations: np.array(n, 3)
            r: np.array(n, 3, 3)
            pose: np.array(n, 3, 4)
    """
    if not isinstance(csv_poses_path, Path):
        csv_poses_path = Path(csv_poses_path)
    # gt_poses_path = './position_rotation.csv'
    csv_path1 = csv_poses_path / 'position_rotation.csv'
    with open(csv_path1, 'r') as f_position:
        f_position_reader = np.loadtxt(f_position, delimiter=',', skiprows=1)
    poses = []
    translations = []
    rotations = []
    scale = 1  # pose scale
    s_num = 0  # total num
    if dataset_type == 'train':
        start_num = START_FRAME_SEQ3
    elif dataset_type == 'val':
        start_num = START_FRAME_SEQ4
    else:
        raise (RuntimeError("Invalid dataset type: " + dataset_type + "\n"
                                                                      "Supported dataset types are: train, val"))
    # init_pose = np.array([[0.0, 1.0, 0.0],
    #                       [0.0, 0.0, -1.0],
    #                       [-1.0, 0.0, 0.0]])
    # start_rot = np.eye(3)
    for row in f_position_reader:
        if s_num >= start_num:  # seq4
            if s_num == start_num:
                start_x = row[0]
                start_y = row[1]
                start_z = row[2]
            # dataxyz = [(row[0] - 1.791779) * scale,
            #            (row[1] - 9.152053) * scale,
            #            (row[2] + 2.348475) * scale]
            x = (row[0] - start_x) * scale
            y = (row[1] - start_y) * scale
            z = (row[2] - start_z) * scale
            dataxyz = [x, y, z]
            translations.append(np.array(dataxyz))

            dataxyz = np.expand_dims(np.array(dataxyz), axis=1)
            mat = R.from_quat(row[3:7]).as_matrix()
            rotations.append(mat)
            # mat = quaternion2matrix(row[3:7])  # quat to matrix
            # mat = np.dot(mat, init_pose)
            temp = np.concatenate([mat, dataxyz], axis=1)
            poses.append(temp)
        s_num += 1
    return [np.array(translations), np.array(rotations), np.array(poses)]
