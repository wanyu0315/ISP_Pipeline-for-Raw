# gamma_correction.py
import numpy as np

class GammaCorrection:
    """
    工业级 Gamma 校正模块 (适配全浮点零损耗管线)
    """
    def execute(self, rgb_image: np.ndarray, gamma: float = 2.2, method: str = 'simple') -> np.ndarray:
        """
        执行 Gamma 校正
        
        Args:
            rgb_image: 输入的 RGB 图像 (严格期望 float32, [0.0, 1.0+])
            gamma: Gamma 值 (通常为 2.2 或 1.0)
            method: 'simple' (纯幂律) 或 'srgb' (工业标准 sRGB 曲线，自带暗部噪声抑制)
        
        Returns:
            校正后的图像 (float32)
        """
        print(f"Executing Gamma Correction with gamma: {gamma}, method: {method}")
        
        # 提前拦截纯线性映射，保证 100% 物理透传
        if gamma == 1.0 and method == 'simple':
            return rgb_image.copy()

        # ===================================================================
        # 🌟 零损耗数据流：防御负数 NaN，保留高光 HDR
        # CCM 可能会产生负数，这会导致 np.power 产生 NaN，必须将底线锁死在 0.0。
        # 上限设为 None，允许 > 1.0 的超亮像素存活到管线最后。
        # ===================================================================
        img_safe = np.clip(rgb_image, 0.0, None)

        # 核心数学计算
        if method == 'srgb' and gamma == 2.2:
            # 真实的 sRGB 曲线标准（带有暗部线性保护段）
            linear_mask = img_safe <= 0.0031308
            gamma_mask = img_safe > 0.0031308
            
            corrected_img = np.zeros_like(img_safe)
            corrected_img[linear_mask] = img_safe[linear_mask] * 12.92
            corrected_img[gamma_mask] = 1.055 * np.power(img_safe[gamma_mask], 1.0 / 2.4) - 0.055
        else:
            # 基础的简单幂律曲线
            corrected_img = np.power(img_safe, 1.0 / gamma)

        # ===================================================================
        # 🌟 零损耗数据流：直接返回 Float32，无任何向下量化截断
        # ===================================================================
        return corrected_img
