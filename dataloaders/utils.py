import numpy as np


def to_3d(depth_map):  # 将点投影到三维空间
    # Load camera parameters
    fx = 159.5906
    cx = 161.9297
    fy = 159.2241
    cy = 163.1686

    # K = np.array([[fx, 0, cx, 0],
    #               [0, fy, cy, 0],
    #               [0, 0, 1, 0],
    #               [0, 0, 0, 1]])  # 内参矩阵

    # Get image size
    height, width = depth_map.shape[:2]
    # depth_map = np.expand_dims(depth_map, axis=2)

    # Compute image coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Convert image coordinates to camera coordinates
    x = (u - cx) / fx * depth_map
    y = (v - cy) / fy * depth_map
    z = depth_map

    # Stack camera coordinates into a single array
    # points = np.stack((x, y, z, np.ones_like(depth_map)), axis=-1)
    points = np.stack((x, y, z), axis=-1)

    return points


def to_vector(depth_map, K):
    """
    [[fx,  0, cx,  0],
     [ 0, fy, cy,  0],
     [ 0,  0,  1,  0]]
    """
    # Load camera parameters
    fx = K[0, 0]  # 159.5906
    cx = K[0, 2]  # 161.9297
    fy = K[1, 1]  # 159.2241
    cy = K[1, 2]  # 163.1686

    # Get image size
    height, width = depth_map.shape[:2]

    # Compute image coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Convert image coordinates to camera coordinates
    x = (u - cx) / fx * depth_map
    y = (v - cy) / fy * depth_map
    z = depth_map

    # Stack camera coordinates into a single array
    # points = np.stack((x, y, z, np.ones_like(depth_map)), axis=-1)
    points = np.stack((x, y, z), axis=-1)

    world_points = points  # 目标点和中心均处在相机坐标系下

    # Convert camera coordinates to world coordinates
    # world_points = np.matmul(points, np.linalg.inv(K))
    # world_points = np.matmul(world_points, np.hstack((R, T[:, np.newaxis])))

    # Compute direction vectors
    norms = np.linalg.norm(world_points, axis=-1)
    vectors = -world_points / np.expand_dims(norms, axis=-1)
    # vectors = -world_points
    # vectors = -points

    return vectors


def pca(points):  # 根据点和近临计算法向量
    data = points.transpose()
    temp = data.mean(axis=1, keepdims=True)
    data = data - temp

    data_T = data.transpose()
    H = np.matmul(data, data_T)

    evectors, evalues, _ = np.linalg.svd(H, full_matrices=True)

    # 最小特征对应的特征向量为法向量
    sort = evalues.argsort()[::-1]
    # evalues = evalues[sort]
    evectors = evectors[:, sort]
    evector = evectors[:, -1]

    return evector


def to_normal_pca(depth_map):
    # 将深度图转换为点图
    points_cam = to_3d(depth_map)

    # 点图的长宽获取
    height, width = points_cam.shape[:2]

    # 初始化法向量图
    normal_map = np.zeros((height, width, 3))
    kernel_size = 1

    # print('使用PCA方法生成法向量')

    # 计算法向量
    for i in range(kernel_size, height-kernel_size):
        for j in range(kernel_size, width-kernel_size):
            # 获取邻域点
            points = points_cam[i-kernel_size:i+kernel_size+1, j-kernel_size:j+kernel_size+1].reshape(-1, 3)
            # d = depth_map[i - 2:i + 3, j - 2:j + 3]
            # d = np.expand_dims(d, axis=1)

            # 使用 PCA 方法计算法向量
            v = pca(points)
            # u, s, v = np.linalg.svd(d - np.mean(d))
            # normal = v[2]

            # 将法向量存储到法向量图中
            normal_map[i, j] = -v

    # 计算mask
    normals_mask1 = np.abs(normal_map) == (0, 0, 1)
    normals_mask1 = np.sum(normals_mask1, axis=-1) == 3
    normals_mask2 = np.abs(normal_map) == (0, 0, 0)
    normals_mask2 = np.sum(normals_mask2, axis=-1) == 3
    normals_mask = (normals_mask1 + normals_mask2) > 0

    return normal_map, normals_mask


def to_mmap(depth_map, K):
    """
    depth_map.shape = (H, W, 1)
    m_map.shape = (H, W, 1)
    K.shape = (3, 4)
    """
    # 生成表面法向量和光线向量
    # depth_map = depth_map[..., 0]
    normal, normals_mask = to_normal_pca(depth_map)  # 使用pca方法生成法向量
    vector = to_vector(depth_map, K)

    # vectors = vectors[1:vectors.shape[0]-1, 1:vectors.shape[1]-1, :3]

    assert normal.shape == vector.shape, \
        "normal.shape:{}, vector.shape:{}".format(normal.shape, vector.shape)  # 两图尺寸必须相同
    nh, nw, _ = normal.shape
    m_map = np.zeros((nh, nw, 1))

    for i in range(nh):
        for j in range(nw):
            m_map[i, j, 0] = np.dot(normal[i, j], vector[i, j].T).sum()

    # 归一化到0-255
    nm_map = (m_map - np.min(m_map)) * 255 / (np.max(m_map) - np.min(m_map))

    # nm_map[normals_mask] = 0  # 将无效区域直接标识在光照图中
    normals_mask_255 = np.zeros((nh, nw, 1))
    normals_mask_255[normals_mask] = 255  # 無效區域標記為255

    return nm_map, normals_mask_255
