import cv2
import numpy as np

def evaluate_pattern(img):
    # 转 HSV，检查颜色分布
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_var = np.var(hsv[:,:,1])  # 饱和度方差
    
    # Laplacian 锐度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 绿色通道平滑度
    green = img[:,:,1]
    grad_x = cv2.Sobel(green, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(green, cv2.CV_64F, 0, 1, ksize=3)
    green_grad = np.mean(np.abs(grad_x) + np.abs(grad_y))
    
    # 综合分数（可调权重）
    score = green_grad - 0.1*sharpness + 0.05*color_var
    return score

def test_bayer_patterns(raw, width, height):
    raw = raw.reshape((height, width))
    patterns = {
        "RGGB": cv2.COLOR_BAYER_RG2BGR,
        "BGGR": cv2.COLOR_BAYER_BG2BGR,
        "GRBG": cv2.COLOR_BAYER_GR2BGR,
        "GBRG": cv2.COLOR_BAYER_GB2BGR,
    }
    
    results = {}
    for name, code in patterns.items():
        rgb = cv2.cvtColor(raw, code)
        score = evaluate_pattern(rgb)
        results[name] = (score, rgb)
        print(f"{name} score: {score:.2f}")
        # cv2.imwrite(f"{name}.png", rgb)  # 保存结果图方便肉眼对比
    
    # 找最优
    best_pattern = min(results, key=lambda k: results[k][0])
    print(f"\n✅ Best guess: {best_pattern}")
    return best_pattern, results

# 例子：读取 RAW 文件（假设 16-bit, 1280x800）
# raw = np.fromfile("frame.raw", dtype=np.uint16)
# best, results = test_bayer_patterns(raw, 1280, 800)
# 读取 RAW 文件（假设 16-bit, 1280x800）
raw = np.fromfile("C:\\librealsense\\rs_convert_res\\raw_\\raw_frame_Color_1758105205861.21386718750000.raw", dtype=np.uint16)
best, results = test_bayer_patterns(raw, 1280, 800)
