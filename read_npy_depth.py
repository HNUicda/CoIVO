# -*- codeing = utf-8 -*-
# @time:2024-06-28 15:20
# @Author:周明明
# @File:read_npy_depth.py
# @Software:PyCharm
import numpy as np
import matplotlib.pyplot as plt
import os
#result_path="E:\experiment\ColVO-20240408\GT_LN_MS\mono_model\models\weights_8"
#result_name=os.path.join(result_path,"poses.npy")
#depthmap = np.load(result_name)  # 使用numpy载入npy文件
#print(depthmap)
#depthmap=np.array([1,2,3])
#plt.imshow(depthmap)  # 执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
#plt.colorbar()                   #添加colorbar
#plt.savefig('depthmap.jpg')  # 执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
#plt.show()  # 在线显示图像
import matplotlib.pyplot as plt
import numpy as np

# 假设 n_depth_new 是经过 np.repeat 操作后的数组，形状为 (a, b, 3c)
# 我们将其重塑为 (a, b, c, 3) 以符合彩色图像的格式

# 重塑数组以符合彩色图像格式
color_image = n_depth_new.reshape((n_depth_new.shape[0], n_depth_new.shape[1], -1, 3))

# 选择一个切片进行可视化，例如第一个切片
slice_index = 0
slice = color_image[slice_index, :, :, :]

# 使用 imshow 函数显示图像
plt.imshow(slice)
plt.colorbar()  # 显示颜色条
plt.title('Color Visualization')
plt.show()