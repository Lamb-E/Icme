import cv2
import numpy as np

# 定义一个函数来计算图像的模糊程度
def is_blurry(image, threshold=50.0):
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算拉普拉斯变换，获取边缘信息
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 根据方差判断是否模糊，方差低说明图像较模糊
    return laplacian_var < threshold, laplacian_var


# 读取并处理图片
image_path = r'/home/zcy/attack/mayiwen/DreamStudio/output_images/9m-18d-2h-9m-37s/0.png'  # 替换为实际的图片路径
image = cv2.imread(image_path)

# 检测是否模糊
blurry, laplacian_var = is_blurry(image)

# 打印结果
if blurry:
    print(f"fuzzy: {laplacian_var}")
else:
    print(f"clear: {laplacian_var}")
