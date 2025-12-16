# gamma_correction.py
import numpy as np

class GammaCorrection:
    """
    Gamma校正模块
    """
    def execute(self, rgb_image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """
        执行Gamma校正。
        
        Args:
            rgb_image: 输入的RGB图像。
            gamma: Gamma值，通常为2.2。

        Returns:
            校正后的图像。
        """
        print(f"Executing Gamma Correction with gamma: {gamma}")
        # 归一化到 [0, 1]，且幂运算（Gamma校正）需要在浮点数上进行。
        max_val = np.iinfo(rgb_image.dtype).max
        img_normalized = rgb_image.astype(np.float32) / max_val
        
        # 应用Gamma校正
        corrected_img = np.power(img_normalized, 1.0 / gamma)
        
        # 转换回原始数据范围
        return (corrected_img * max_val).astype(rgb_image.dtype)