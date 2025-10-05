import numpy as np
import cv2
import os
# 将灰色图转成伪彩色图并保存
def image_grey_save(n_depth,filename):
    # 将Numpy矩阵转换成OpenCV图像
    img_grey = cv2.cvtColor(n_depth, cv2.COLOR_BGR2RGB)
    # 转成伪彩色图
    image_cv = cv2.applyColorMap(img_grey, cv2.COLORMAP_TURBO)
    image_cv2 = cv2.applyColorMap(img_grey, cv2.COLORMAP_JET)
    filepath = "E:\experiment\ColVO-20240408\grey_image"
    folder = os.path.exists(filepath)
    # 判断是否存在文件夹如果不存在则创建为文件夹
    if not folder:
        os.makedirs(filepath)
        print("111")
    filename1 = os.path.join(filepath, "TURBO_"+filename+".jpg")
    print(filename1)
    filename2 = os.path.join(filepath, "JET_"+filename+".jpg")
    cv2.imwrite(filename1, image_cv)
    cv2.imwrite(filename2, image_cv2)

#测试用例
#n_depth = np.load("test.npy")
#image_grey_save(n_depth,"test")

'''n_depth = np.load("test.npy")
# 将Numpy矩阵转换成OpenCV图像
img_grey = cv2.cvtColor(n_depth, cv2.COLOR_BGR2RGB)
# 转成伪彩色图
image_cv = cv2.applyColorMap(img_grey, cv2.COLORMAP_TURBO)
image_cv2 = cv2.applyColorMap(img_grey, cv2.COLORMAP_JET)
# 显示OpenCV图像
cv2.imshow('image', img_grey)
cv2.imshow('image2', image_cv)
cv2.imshow('image3', image_cv2)
#cv2.waitKey(0)
cv2.destroyAllWindows()
filepath="E:\experiment\ColVO-20240408\grey_image"
folder = os.path.exists(filepath)
# 判断是否存在文件夹如果不存在则创建为文件夹
if not folder:
    os.makedirs(filepath)
    print("111")
filename1 =os.path.join(filepath,"image1.jpg")
filename2 =os.path.join(filepath,"image2.jpg")
cv2.imwrite(filename1, image_cv)
cv2.imwrite(filename2, image_cv2)'''