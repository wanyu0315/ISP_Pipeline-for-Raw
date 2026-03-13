import numpy as np
import cv2
import os

# --- 参数 ---
WIDTH = 1280
HEIGHT = 800
DTYPE = np.uint16
raw_file_path = r"/home/lizize/pyVHR_for_ISP/ISPpipline/raw_data/baseenvironment_rawframe/raw_lzz/raw_frame_Color_1765956387838.58398437500000.raw"

# --- 加载 RAW ---
if not os.path.exists(raw_file_path):
    raise FileNotFoundError(f"文件不存在: {raw_file_path}")

raw = np.fromfile(raw_file_path, dtype=DTYPE).reshape((HEIGHT, WIDTH))
print(f"RAW 加载成功, shape={raw.shape}, dtype={raw.dtype}")

# =============================================================
# Bayer Pattern 检测
# 原理：对图像中亮区域（白色/灰色）采样，
#       在 2x2 Bayer 格子中，R/B 像素值应明显低于 G 像素值。
#       通过比较四个位置的均值，推断哪个位置是 R/G/B。
#
# 2x2 Bayer 格子位置：
#   (row%2==0, col%2==0) → 位置 [0,0]
#   (row%2==0, col%2==1) → 位置 [0,1]
#   (row%2==1, col%2==0) → 位置 [1,0]
#   (row%2==1, col%2==1) → 位置 [1,1]
# =============================================================

# 提取四个子采样通道（每隔一行一列取一个）
ch00 = raw[0::2, 0::2].astype(np.float64)  # 位置 (0,0)
ch01 = raw[0::2, 1::2].astype(np.float64)  # 位置 (0,1)
ch10 = raw[1::2, 0::2].astype(np.float64)  # 位置 (1,0)
ch11 = raw[1::2, 1::2].astype(np.float64)  # 位置 (1,1)

means = {
    '(0,0)': ch00.mean(),
    '(0,1)': ch01.mean(),
    '(1,0)': ch10.mean(),
    '(1,1)': ch11.mean(),
}

print("\n--- Bayer Pattern 检测 ---")
print("各位置均值（越高越可能是 G，最低的两个是 R 和 B）：")
for pos, val in sorted(means.items(), key=lambda x: -x[1]):
    print(f"  位置 {pos}: {val:.1f}")

# 找出最高的两个（G 通道有两个）和最低的两个（R/B 各一个）
sorted_pos = sorted(means.items(), key=lambda x: -x[1])
g_positions = [sorted_pos[0][0], sorted_pos[1][0]]
rb_positions = [sorted_pos[2][0], sorted_pos[3][0]]

print(f"\n推断 G 通道位置: {g_positions}")
print(f"推断 R/B 通道位置: {rb_positions}（需要彩色参考图区分 R 和 B）")

# 用四种 pattern 分别去马赛克，保存结果，肉眼判断哪个颜色正确
# 注意：OpenCV Bayer 命名与 EXIF 标准相反，此处已做正确映射
patterns = {
    'RGGB': cv2.COLOR_BAYER_BG2BGR,
    'GRBG': cv2.COLOR_BAYER_GB2BGR,
    'GBRG': cv2.COLOR_BAYER_GR2BGR,
    'BGGR': cv2.COLOR_BAYER_RG2BGR,
}

def gray_world_wb(bgr_uint16):
    """灰世界白平衡，以 G 通道为基准"""
    f = bgr_uint16.astype(np.float32)
    b, g, r = cv2.split(f)
    avg_g = g.mean()
    f[:, :, 0] = np.clip(b * (avg_g / b.mean()), 0, 65535)  # B
    f[:, :, 2] = np.clip(r * (avg_g / r.mean()), 0, 65535)  # R
    return f.astype(np.uint16)

os.makedirs("ISPpipline/Check_raw/pattern_test", exist_ok=True)

for name, code in patterns.items():
    bgr = cv2.cvtColor(raw, code)
    bgr_wb = gray_world_wb(bgr)
    display = cv2.normalize(bgr_wb, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    out_path = f"ISPpipline/Check_raw/pattern_test/{name}.png"
    cv2.imwrite(out_path, display)
    print(f"已保存: {out_path}")

print("\n请打开 pattern_test/ 目录，找到颜色看起来正常（肤色/天空/植物颜色正确）的那张图，")
print("对应的文件名即为正确的 Bayer Pattern。")
