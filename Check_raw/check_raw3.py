import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- 加载您的RAW文件 ---
WIDTH = 1280
HEIGHT = 800
DTYPE = np.uint16
raw_file_path = r"C:\librealsense\rs_convert_res\raw_\raw_frame_Color_1758105205861.21386718750000.raw" # 替换路径

bayer_raw_1d = np.fromfile(raw_file_path, dtype=DTYPE)
raw_image = bayer_raw_1d.reshape((HEIGHT, WIDTH))

# --- 选择一个您认为有高对比度边缘的区域 (ROI: Region of Interest) ---
# 这是一个示例坐标，您需要根据您的图像内容调整 x, y 的值
# 比如，选择键盘上的一个亮区和暗区的交界
x, y, w, h = 1060, 185, 10, 10  # (左上角x, 左上角y, 宽度, 高度)
roi = raw_image[y:y+h, x:x+w]

# --- 使用matplotlib可视化这个小区块 ---
plt.figure(figsize=(8, 8))
# interpolation='nearest' 是关键，它能让我们看清每一个独立的像素块
plt.imshow(roi, cmap='gray', interpolation='nearest')

# 在每个像素上显示其16位的亮度值，方便分析
for i in range(h):
    for j in range(w):
        plt.text(j, i, roi[i, j], ha="center", va="center", color="red", fontsize=8)

plt.title(f"Zoomed-in RAW values at ({x},{y})")
plt.show()