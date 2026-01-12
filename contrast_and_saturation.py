# contrast_saturation.py
import numpy as np
import cv2
from scipy.ndimage import uniform_filter


class ContrastSaturation:
    """
    对比度与饱和度调整模块
    
    所有方法均假设输入为 np.float32 [0, 1] YUV图像，
    输出也为 np.float32 [0, 1] YUV图像。
    """
    
    def execute(self, yuv_image: np.ndarray, 
                contrast_method: str = 'sigmoid', 
                saturation_method: str = 'linear',
                contrast_factor: float = 1.0,
                saturation_factor: float = 1.0,
                **kwargs) -> np.ndarray:
        """
        执行对比度和饱和度调整。
        
        Args:
            yuv_image: 输入的YUV图像 (np.float32, 范围 [0, 1])
            contrast_method: 对比度调整方法 ('linear', 'histogram_equalization', 'clahe', 'gamma', 'sigmoid'(S-curve), 'adaptive')
            saturation_method: 饱和度调整方法 ('linear', 'hsv', 'vibrance', 'selective')
            contrast_factor: 对比度因子 (1.0为原始)
            saturation_factor: 饱和度因子 (1.0为原始)
            **kwargs: 其他参数 (例如 clahe 的 clip_limit)
        
        Returns:
            调整后的YUV图像 (np.float32, 范围 [0, 1])
        """
        print(f"Executing Contrast & Saturation adjustment")
        print(f"  Contrast method: {contrast_method}, factor: {contrast_factor}")
        print(f"  Saturation method: {saturation_method}, factor: {saturation_factor}")
        
        # 确保输入是 float32
        if yuv_image.dtype != np.float32:
            print(f"警告: ContrastSaturation 模块接收到非 float32 图像 (dtype: {yuv_image.dtype})。")
            if yuv_image.max() > 1.0:
                 yuv_image = yuv_image.astype(np.float32) / 65535.0
            else:
                 yuv_image = yuv_image.astype(np.float32)

        # 复制一份用于操作
        result = yuv_image.copy()
        
        # 1. 对比度调整（作用于Y通道）
        # (检查 'linear' 是为了防止 factor=1.0 但 method='clahe' 时被跳过)
        if contrast_factor != 1.0 or contrast_method != 'linear':
            y_channel = result[:, :, 0]
            y_adjusted = self._adjust_contrast(
                y_channel, 
                method=contrast_method, 
                factor=contrast_factor,
                **kwargs
            )
            result[:, :, 0] = y_adjusted
        
        # 2. 饱和度调整（作用于UV通道或整个图像）
        if saturation_factor != 1.0 or saturation_method != 'linear':
            # 饱和度函数可能会修改整个 YUV 图像 (例如 hsv 方法)
            # 因此我们将已调整了Y通道的 'result' 传入
            result = self._adjust_saturation(
                result,
                method=saturation_method,
                factor=saturation_factor,
                **kwargs
            )
        
        # 最终裁剪以确保安全
        return np.clip(result, 0, 1)
    
    # ==================== 对比度调整方法 ====================
    
    def _adjust_contrast(self, y_channel: np.ndarray, method: str, 
                         factor: float, **kwargs) -> np.ndarray:
        """调整对比度的统一接口"""
        if method == 'linear':
            # 'factor' 是对比度因子
            return self._linear_contrast(y_channel, factor)
        elif method == 'histogram_equalization':
            return self._histogram_equalization(y_channel)
        elif method == 'clahe':
            return self._clahe(y_channel, **kwargs)
        elif method == 'gamma':
            # 'factor' 在这里作为 'gamma' 值
            return self._gamma_contrast(y_channel, factor)
        elif method == 'sigmoid':
            # 'factor' 在这里作为 'strength'
            return self._sigmoid_contrast(y_channel, factor, **kwargs)
        elif method == 'adaptive':
            # 'factor' 在这里作为 'strength'
            return self._adaptive_contrast(y_channel, factor, **kwargs)
        else:
            raise ValueError(f"Unknown contrast method: {method}")
    
    def _linear_contrast(self, channel: np.ndarray, factor: float) -> np.ndarray:
        """
        线性对比度调整 - (float32 兼容)
        公式: output = factor * (input - 0.5) + 0.5 (Y通道中心为0.5)
        """
        # Y 通道的中心点是 0.5 (对应灰阶)
        mean = 0.5
        
        # channel 已经是 float32 [0, 1]
        adjusted = factor * (channel - mean) + mean
        return np.clip(adjusted, 0, 1)
    
    def _histogram_equalization(self, channel: np.ndarray, **kwargs) -> np.ndarray:
        """
        直方图均衡化 - 增强全局对比度(需转uint8)
        
        注意: cv2.equalizeHist 只接受 uint8 图像
        """
        ## --- 兼容性核心：执行局部 float32 -> uint8 -> float32 转换 ---
        # 1. 转换为 [0, 255] uint8
        channel_uint8 = (channel * 255.0).astype(np.uint8)
        
        # 2. 在 uint8 上执行处理
        equalized_uint8 = cv2.equalizeHist(channel_uint8)
        
        # 3. 转换回 [0, 1] float32
        return equalized_uint8.astype(np.float32) / 255.0
        ## --- 转换结束 ---
    
    def _clahe(self, channel: np.ndarray, clip_limit: float = 2.0, 
               tile_grid_size: tuple = (8, 8), **kwargs) -> np.ndarray:
        """
        CLAHE - 局部对比度增强 (需转uint8)
        
        Args:
            clip_limit: 对比度限制
            tile_grid_size: 网格大小

        注意: clahe.apply 只接受 uint8 (或 uint16)
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        ## --- 兼容性核心：执行局部 float32 -> uint8 -> float32 转换 ---
        # 1. 转换为 [0, 255] uint8
        channel_uint8 = (channel * 255.0).astype(np.uint8)
        
        # 2. 在 uint8 上执行处理
        equalized_uint8 = clahe.apply(channel_uint8)
        
        # 3. 转换回 [0, 1] float32
        return equalized_uint8.astype(np.float32) / 255.0
        ## --- 转换结束 ---
    
    def _gamma_contrast(self, channel: np.ndarray, gamma: float, **kwargs) -> np.ndarray:
        """
        Gamma对比度调整 - (float32 兼容)
        
        注意: factor 被用作 gamma 值。gamma < 1 增加对比度。
        """
        if gamma <= 0:
            gamma = 0.01 # 防止除零
            
        # channel 已经是 float32 [0, 1]
        # 使用 1.0 / gamma 是标准 Gamma 校正
        # 如果想 'factor > 1' 增加对比度, 应该使用 '1.0 / factor'
        adjusted = np.power(channel, 1.0 / gamma)
        return np.clip(adjusted, 0, 1)
    
    def _sigmoid_contrast(self, channel: np.ndarray, strength: float = 5.0, 
                          midpoint: float = 0.5, **kwargs) -> np.ndarray:
        """
        S曲线（Sigmoid）对比度调整 - (float32 兼容)
        参数 Strength (Contrast Strength / Gain) —— 控制“S”的弯曲程度
            Strength越大： 曲线越陡峭（S形越明显）。
            效果： 暗部更暗，亮部更亮，中间调层次拉开。
        参数 midpoint —— 控制“S”的中心位置
        """
        # channel 已经是 float32 [0, 1]
        
        # Sigmoid函数
        adjusted = 1 / (1 + np.exp(-strength * (channel - midpoint)))
        
        # 归一化到 [0, 1] (如原始代码)
        adj_min = adjusted.min()
        adj_max = adjusted.max()
        range = adj_max - adj_min
        if range > 1e-6:
            adjusted = (adjusted - adj_min) / range
        else:
            adjusted = np.full_like(adjusted, adj_min) # 纯色
            
        return np.clip(adjusted, 0, 1)
    
    def _adaptive_contrast(self, channel: np.ndarray, factor: float, 
                           window_size: int = 31, **kwargs) -> np.ndarray:
        """
        自适应对比度增强 - (float32 兼容) 根据局部统计调整
        
        Args:
            factor: 增强因子
            window_size: 局部窗口大小
        """
        # channel 已经是 float32 [0, 1]
        
        # 计算局部均值和标准差
        local_mean = uniform_filter(channel, size=window_size)
        local_mean_sq = uniform_filter(channel**2, size=window_size)
        local_variance = local_mean_sq - local_mean**2
        local_std = np.sqrt(np.maximum(local_variance, 0))
        
        # 自适应对比度增强
        adjusted = local_mean + factor * (channel - local_mean) * (1 + local_std)
        return np.clip(adjusted, 0, 1)
    
    # ==================== 饱和度调整方法 ====================
    
    def _adjust_saturation(self, yuv_image: np.ndarray, method: str, 
                           factor: float, **kwargs) -> np.ndarray:
        """调整饱和度的统一接口"""
        if method == 'linear':
            return self._linear_saturation(yuv_image, factor)
        elif method == 'hsv':
            return self._hsv_saturation(yuv_image, factor)
        elif method == 'vibrance':
            return self._vibrance(yuv_image, factor, **kwargs)
        elif method == 'selective':
            return self._selective_saturation(yuv_image, factor, **kwargs)
        else:
            raise ValueError(f"Unknown saturation method: {method}")
    
    def _linear_saturation(self, yuv_image: np.ndarray, factor: float, **kwargs) -> np.ndarray:
        """
        线性饱和度调整 - 直接缩放UV通道(float32 兼容)
        
        注意: 这是在 YUV 域中最高效的方法
        """
        result = yuv_image.copy()
        
        # yuv_image 已经是 float32 [0, 1]
        # UV 通道中心为 0.5
        u_centered = result[:, :, 1] - 0.5
        v_centered = result[:, :, 2] - 0.5
        
        # 中心化后缩放
        u_adjusted = u_centered * factor + 0.5
        v_adjusted = v_centered * factor + 0.5
        
        # 裁剪并写回
        result[:, :, 1] = np.clip(u_adjusted, 0, 1)
        result[:, :, 2] = np.clip(v_adjusted, 0, 1)
        
        return result
    
    def _vibrance(self, yuv_image: np.ndarray, factor: float, 
                  skin_protection: float = 0.5, **kwargs) -> np.ndarray:
        """
        智能饱和度调整（Vibrance）- 保护已饱和区域和肤色 (float32 兼容)
        
        Args:
            factor: 饱和度因子
            skin_protection: 肤色保护强度 (0-1)

        注意: 这是在 YUV 域中高效的实现
        """
        result = yuv_image.copy()
        
        # yuv_image 已经是 float32 [0, 1]
        u_centered = result[:, :, 1] - 0.5
        v_centered = result[:, :, 2] - 0.5
        
        # 计算当前饱和度（在 [0, ~0.707] 范围内）
        current_saturation = np.sqrt(u_centered**2 + v_centered**2)
        
        # 自适应调整：低饱和度区域调整更多
        # (对 'current_saturation' 进行归一化以获得更好的效果)
        sat_norm = current_saturation * 1.414 # 近似归一化到 [0, 1]
        adaptive_factor = 1 + (factor - 1) * (1 - np.clip(sat_norm, 0, 1))
        
        # 肤色检测和保护（简化版）
        # 肤色通常在特定的UV范围内 (这些值在 [0, 1] 范围内)
        is_skin = (np.abs(u_centered) < 0.1) & (v_centered > 0.1) & (v_centered < 0.2)
        skin_mask = is_skin.astype(np.float32)
        
        # 混合肤色保护
        final_factor = adaptive_factor * (1 - skin_protection * skin_mask) + (factor * skin_mask)
        
        # 应用调整
        u_adjusted = u_centered * final_factor + 0.5
        v_adjusted = v_centered * final_factor + 0.5
        
        result[:, :, 1] = np.clip(u_adjusted, 0, 1)
        result[:, :, 2] = np.clip(v_adjusted, 0, 1)
        
        return result
    
    def _hsv_saturation(self, yuv_image: np.ndarray, factor: float, **kwargs) -> np.ndarray:
        """
        HSV空间饱和度调整 - 更符合人眼感知 (需复杂转换)
        
        警告: 此方法效率极低 (YUV->RGB->HSV->RGB->YUV)
        """
        ## --- 兼容性核心：执行完整的 YUV -> RGB -> HSV -> RGB -> YUV 转换 ---
        
        # 1. YUV [0, 1] -> RGB [0, 1] (使用 float32 原生辅助函数)
        rgb_image_float = self._yuv_to_rgb(yuv_image)
        
        # 2. RGB [0, 1] -> RGB [0, 255] uint8
        rgb_uint8 = (rgb_image_float * 255.0).astype(np.uint8)
        
        # 3. RGB uint8 -> HSV uint8
        #    H [0, 179], S [0, 255], V [0, 255]
        hsv_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
        hsv_float = hsv_uint8.astype(np.float32)
        
        # 4. 调整饱和度通道 S
        hsv_float[:, :, 1] = np.clip(hsv_float[:, :, 1] * factor, 0, 255)
        
        # 5. HSV -> RGB
        hsv_adjusted_uint8 = hsv_float.astype(np.uint8)
        rgb_adjusted_uint8 = cv2.cvtColor(hsv_adjusted_uint8, cv2.COLOR_HSV2RGB)
        
        # 6. RGB [0, 255] uint8 -> RGB [0, 1] float32
        rgb_adjusted_float = rgb_adjusted_uint8.astype(np.float32) / 255.0
        
        # 7. RGB [0, 1] -> YUV [0, 1] (使用 float32 原生辅助函数)
        return self._rgb_to_yuv(rgb_adjusted_float)
        ## --- 转换结束 ---
    
    def _selective_saturation(self, yuv_image: np.ndarray, factor: float,
                            target_hue: float = 0, hue_range: float = 30, **kwargs) -> np.ndarray:
        """
        选择性饱和度调整 - 只调整特定颜色范围 (需复杂转换)
        
        Args:
            factor: 饱和度因子
            target_hue: 目标色相 (0-360)，None表示全局
            hue_range: 色相范围
        """
        ## --- 兼容性核心：与 _hsv_saturation 相同的转换 ---
        
        # 1. YUV -> RGB (float32)
        rgb_image_float = self._yuv_to_rgb(yuv_image)
        
        # 2. RGB -> HSV (uint8)
        rgb_uint8 = (rgb_image_float * 255.0).astype(np.uint8)
        hsv_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
        hsv_float = hsv_uint8.astype(np.float32)
        
        # 3. 创建色相掩模
        hue_channel = hsv_float[:, :, 0] # 范围 [0, 179]
        
        # 【BUG 修复】: 将 [0, 360] 范围的输入 转换为 OpenCV [0, 179] 范围
        target_hue_cv = (target_hue / 360.0) * 179.0
        hue_range_cv = (hue_range / 360.0) * 179.0
        
        # 计算色相差异 (处理 179 -> 0 的环绕)
        hue_diff = np.abs(hue_channel - target_hue_cv)
        hue_diff = np.minimum(hue_diff, 180.0 - hue_diff)
        
        # 创建高斯掩模
        # (使用 1/3 范围作为标准差)
        std_dev = hue_range_cv / 3.0
        if std_dev < 1e-6:
             mask = (hue_diff < 1.0).astype(np.float32)
        else:
            mask = np.exp(-(hue_diff**2) / (2 * (std_dev**2)))
        
        # 4. 选择性调整饱和度
        # S' = S * (1 + (factor - 1) * mask)
        s_channel = hsv_float[:, :, 1]
        s_adjusted = s_channel * (1.0 + (factor - 1.0) * mask)
        hsv_float[:, :, 1] = np.clip(s_adjusted, 0, 255)
        
        # 5. HSV -> RGB -> YUV
        hsv_adjusted_uint8 = hsv_float.astype(np.uint8)
        rgb_adjusted_uint8 = cv2.cvtColor(hsv_adjusted_uint8, cv2.COLOR_HSV2RGB)
        rgb_adjusted_float = rgb_adjusted_uint8.astype(np.float32) / 255.0
        return self._rgb_to_yuv(rgb_adjusted_float)
    
    # ==================== 辅助函数 (float32 原生) ====================
    
    def _yuv_to_rgb(self, yuv_float: np.ndarray) -> np.ndarray:
        """YUV转RGB (BT.601) - float32 [0, 1] 原生"""
        
        # UV去偏移
        y = yuv_float[:, :, 0]
        u = yuv_float[:, :, 1] - 0.5
        v = yuv_float[:, :, 2] - 0.5
        
        # BT.601 转换矩阵 (Y'UV to R'G'B')
        # R = Y + 1.402 * V
        # G = Y - 0.344136 * U - 0.714136 * V
        # B = Y + 1.772 * U
        r = y + 1.402 * v
        g = y - 0.344136 * u - 0.714136 * v
        b = y + 1.772 * u
        
        rgb = np.stack((r, g, b), axis=-1)
        return np.clip(rgb, 0, 1)
    
    def _rgb_to_yuv(self, rgb_float: np.ndarray) -> np.ndarray:
        """RGB转YUV (BT.601) - float32 [0, 1] 原生"""
        
        r = rgb_float[:, :, 0]
        g = rgb_float[:, :, 1]
        b = rgb_float[:, :, 2]
        
        # BT.601 转换矩阵 (R'G'B' to Y'UV)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.168736 * r - 0.331264 * g + 0.5 * b
        v = 0.5 * r - 0.418688 * g - 0.081312 * b
        
        # UV加偏移
        yuv = np.stack((y, u + 0.5, v + 0.5), axis=-1)
        return np.clip(yuv, 0, 1)