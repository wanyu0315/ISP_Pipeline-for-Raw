# ===================================================================
# 基础ISP模块：黑电平校正、镜头阴影校正、坏点校正
# ===================================================================

import numpy as np
from scipy.ndimage import median_filter


# ===================================================================
# 1. 黑电平校正 (Black Level Correction - BLC)
# ===================================================================

class BlackLevelCorrection:
    """
    黑电平校正模块
    消除传感器暗电流，建立真正的"0"点
    """
    
    def execute(self, raw_data: np.ndarray, 
                black_level: int = None,
                auto_detect: bool = False,
                **kwargs) -> np.ndarray:
        """
        执行黑电平校正
        
        Args:
            raw_data: 输入Raw数据 (2D Bayer数组)
            black_level: 黑电平值
                - 如果为None且auto_detect=False，根据位深自动推断
                - 如果auto_detect=True，从图像暗区自动检测
            auto_detect: 是否自动检测黑电平
        
        Returns:
            校正后的Raw数据
        """
        print(f"Executing Black Level Correction")
        
        # 1. 确定黑电平值
        if auto_detect:
            black_level = self._detect_black_level(raw_data)
            print(f"  Auto-detected black level: {black_level}")
        elif black_level is None:
            black_level = self._estimate_black_level(raw_data)
            print(f"  Estimated black level: {black_level}")
        else:
            print(f"  Manual black level: {black_level}")
        
        # 2. 执行校正
        corrected = raw_data.astype(np.float32) - black_level
        
        # 3. 裁剪负值
        corrected = np.maximum(corrected, 0)
        
        # 4. 转换回原类型
        return corrected.astype(raw_data.dtype)
    
    def _estimate_black_level(self, raw_data: np.ndarray) -> int:
        """
        根据位深估计典型黑电平值
        """
        max_val = np.iinfo(raw_data.dtype).max
        
        if max_val <= 255:  # 8-bit
            return 16
        elif max_val <= 1023:  # 10-bit
            return 64
        elif max_val <= 4095:  # 12-bit
            return 256
        elif max_val <= 16383:  # 14-bit
            return 1024
        else:  # 16-bit
            return 2048
    
    def _detect_black_level(self, raw_data: np.ndarray, 
                           percentile: float = 0.1) -> int:
        """
        从图像最暗区域自动检测黑电平
        
        Args:
            percentile: 使用最暗的百分位数
        """
        # 使用最暗的0.1%像素的平均值作为黑电平
        black_level = int(np.percentile(raw_data, percentile))
        
        # 确保不会过度校正
        estimated = self._estimate_black_level(raw_data)
        black_level = min(black_level, estimated * 1.5)
        
        return black_level


# ===================================================================
# 2. 镜头阴影校正 (Lens Shading Correction - LSC)
# ===================================================================

class LensShadingCorrection:
    """
    镜头阴影校正模块
    补偿镜头渐晕效应（中心亮、边缘暗）
    """
    
    def execute(self, raw_data: np.ndarray,
                method: str = 'polynomial',
                gain_map: np.ndarray = None,
                **kwargs) -> np.ndarray:
        """
        执行镜头阴影校正
        
        Args:
            raw_data: 输入Raw数据
            method: 校正方法
                - 'polynomial': 多项式拟合（自动生成增益图）
                - 'gain_map': 使用预先标定的增益图
                - 'radial': 径向模型
            gain_map: 预先标定的增益图（当method='gain_map'时必须提供）
        
        Returns:
            校正后的Raw数据
        """
        print(f"Executing Lens Shading Correction using method: {method}")
        
        if method == 'gain_map' and gain_map is not None:
            return self._apply_gain_map(raw_data, gain_map)
        elif method == 'polynomial':
            return self._polynomial_correction(raw_data, **kwargs)
        elif method == 'radial':
            return self._radial_correction(raw_data, **kwargs)
        else:
            print("  Warning: No LSC applied (no gain map provided)")
            return raw_data
    
    def _apply_gain_map(self, raw_data: np.ndarray, 
                        gain_map: np.ndarray) -> np.ndarray:
        """
        应用预先标定的增益图
        
        Args:
            gain_map: 增益图，尺寸必须与raw_data相同
        """
        if gain_map.shape != raw_data.shape:
            # 如果尺寸不匹配，需要插值
            from scipy.ndimage import zoom
            zoom_factors = (raw_data.shape[0] / gain_map.shape[0],
                          raw_data.shape[1] / gain_map.shape[1])
            gain_map = zoom(gain_map, zoom_factors, order=1)
        
        corrected = raw_data.astype(np.float32) * gain_map
        
        max_val = np.iinfo(raw_data.dtype).max
        return np.clip(corrected, 0, max_val).astype(raw_data.dtype)
    
    def _polynomial_correction(self, raw_data: np.ndarray,
                               strength: float = 0.3) -> np.ndarray:
        """
        使用多项式模型生成增益图
        适用于没有预先标定的情况
        
        Args:
            strength: 校正强度 (0-1)
        """
        h, w = raw_data.shape
        
        # 生成归一化坐标网格 [-1, 1]
        y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
        
        # 计算到中心的距离
        r_squared = x**2 + y**2
        
        # 多项式模型：gain = 1 + a*r^2 + b*r^4
        # 边缘增益大于1，中心接近1
        a = 0.2 * strength
        b = 0.1 * strength
        
        gain_map = 1.0 + a * r_squared + b * (r_squared ** 2)
        
        # 限制增益范围
        gain_map = np.clip(gain_map, 1.0, 2.0)
        
        print(f"  Generated polynomial gain map (strength={strength:.2f})")
        print(f"  Gain range: [{gain_map.min():.3f}, {gain_map.max():.3f}]")
        
        corrected = raw_data.astype(np.float32) * gain_map
        
        max_val = np.iinfo(raw_data.dtype).max
        return np.clip(corrected, 0, max_val).astype(raw_data.dtype)
    
    def _radial_correction(self, raw_data: np.ndarray,
                          vignetting_coefficient: float = 0.3) -> np.ndarray:
        """
        径向渐晕模型
        
        Args:
            vignetting_coefficient: 渐晕系数
        """
        h, w = raw_data.shape
        
        # 归一化坐标
        y, x = np.ogrid[0:h, 0:w]
        center_y, center_x = h / 2, w / 2
        
        # 计算归一化距离
        max_radius = np.sqrt(center_y**2 + center_x**2)
        r = np.sqrt((y - center_y)**2 + (x - center_x)**2) / max_radius
        
        # 径向渐晕模型: gain = 1 / (1 - k*r^2)
        k = vignetting_coefficient
        gain_map = 1.0 / (1.0 - k * r**2 + 1e-6)
        
        # 限制增益
        gain_map = np.clip(gain_map, 1.0, 2.5)
        
        print(f"  Applied radial correction (k={k:.2f})")
        
        corrected = raw_data.astype(np.float32) * gain_map
        
        max_val = np.iinfo(raw_data.dtype).max
        return np.clip(corrected, 0, max_val).astype(raw_data.dtype)
    
    def calibrate_gain_map(self, flat_field_image: np.ndarray) -> np.ndarray:
        """
        从平场图像标定增益图
        
        Args:
            flat_field_image: 均匀光照下拍摄的图像（如拍摄白墙）
        
        Returns:
            增益图
        """
        # 计算目标亮度（使用中心区域的平均值）
        h, w = flat_field_image.shape
        center_region = flat_field_image[
            h//4:3*h//4,
            w//4:3*w//4
        ]
        target_value = np.mean(center_region)
        
        # 生成增益图
        gain_map = target_value / (flat_field_image.astype(np.float32) + 1e-6)
        
        # 限制增益范围
        gain_map = np.clip(gain_map, 0.5, 3.0)
        
        print(f"Calibrated gain map from flat field")
        print(f"  Target value: {target_value:.1f}")
        print(f"  Gain range: [{gain_map.min():.3f}, {gain_map.max():.3f}]")
        
        return gain_map


