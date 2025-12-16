import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# --- 1. 基本参数设置 ---
WIDTH = 1280
HEIGHT = 800
DTYPE = np.uint16
raw_file_path = r"ISPpipline/raw_data/raw_data_1/raw_frame_Color_1762949434706.22973632812500.raw"

# D455 通常是 BGGR 或 RGGB，可根据实际情况切换
BAYER_PATTERN = cv2.COLOR_BAYER_GB2BGR  # GBRG    OpenCV使用BGR通道进行处理和显示，为了方便OpenCV的显示和写入，使用BGR进行存储。
                                        # 后续处理前还是要先转化为RGB

# --- 2. 加载 RAW 文件 ---
if not os.path.exists(raw_file_path):
    raise FileNotFoundError(f"错误: 文件不存在 {raw_file_path}")

bayer_raw_1d = np.fromfile(raw_file_path, dtype=DTYPE)
raw_image = bayer_raw_1d.reshape((HEIGHT, WIDTH))
print("RAW 文件加载成功。")

# --- 3. 去马赛克 ---
bgr_16bit = cv2.cvtColor(raw_image, BAYER_PATTERN)    ## OpenCV直接输出BGR格式
print("去马赛克完成。")

rgb_16bit = cv2.cvtColor(bgr_16bit, cv2.COLOR_BGR2RGB)    ## 转换为RGB格式


# 保存原始去马赛克结果 (16位PNG)
cv2.imwrite("ISPpipline/Check_raw/demosaic_raw16.png", bgr_16bit)
print("保存原始去马赛克图像 -> demosaic_raw16.png")

# --- 4. 自动白平衡 (灰世界算法) ---
bgr_float = bgr_16bit.astype(np.float32)
b, g, r = cv2.split(bgr_float)

avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
avg_gray = avg_g

gain_b = avg_gray / avg_b
gain_r = avg_gray / avg_r

# 限制增益范围，避免过度
gain_b = np.clip(gain_b, 0.5, 2.0)
gain_r = np.clip(gain_r, 0.5, 2.0)

b_balanced = b * gain_b
r_balanced = r * gain_r
balanced_bgr_float = cv2.merge([b_balanced, g, r_balanced])

# 裁剪到16位范围并转回uint16
wb_bgr_16bit = np.clip(balanced_bgr_float, 0, 65535).astype(np.uint16)  # BGR格式
print("自动白平衡完成。")

# 保存白平衡后的 16 位结果
cv2.imwrite("ISPpipline/Check_raw/wb_raw16.png", wb_bgr_16bit)
print("保存白平衡矫正图像 (16位) -> wb_raw16.png")

# --- 5. 转换到8位以供显示 ---
display_8bit = cv2.normalize(wb_bgr_16bit, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 保存并显示结果
cv2.imwrite("ISPpipline/Check_raw/wb_display_8bit.png", display_8bit)   
print("保存显示用图像 (8位) -> wb_display_8bit.png")


# cv2.imshow("Corrected Image_cvBGR (8-bit)", display_8bit)  # BGR格式
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.imshow(cv2.cvtColor(display_8bit, cv2.COLOR_BGR2RGB))   # 转换为RGB格式
# plt.title("Corrected Image_pltRGB (8-bit)")    
# plt.show()      