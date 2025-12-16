# # white_balance.py
import numpy as np
# white_balance_improved.py
import numpy as np


class WhiteBalanceRaw:
    """
    Raw域白平衡处理模块
    在Bayer Raw数据上直接进行白平衡，这是专业ISP的标准做法
    """

    def __init__(self):
        """初始化白平衡模块"""
        self.gain_limits = (0.25, 4.0)  # 增益范围限制
    
    def execute(self, image: np.ndarray, algorithm: str = 'gray_world_green', 
                bayer_pattern: str = 'GBRG', **kwargs) -> np.ndarray:
        """
        执行Raw域白平衡操作。
        
        Args:
            image: 输入Raw Bayer数据 (2D数组, HxW)
            algorithm: 白平衡算法
                - 'gray_world': 标准灰度世界
                - 'gray_world_green': 以绿色为基准（推荐）
                - 'perfect_reflector': 完美反射法
                - 'max_white': 最大白法
                - 'manual': 手动指定增益
            bayer_pattern: Bayer模式 ('RGGB', 'BGGR', 'GRBG', 'GBRG')
            **kwargs: 算法特定参数
        
        Returns:
            白平衡后的Raw图像
        """
        if bayer_pattern is None:
            raise ValueError("Raw域白平衡必须提供 'bayer_pattern' 参数")
        
        if image.ndim != 2:
            raise ValueError(f"Raw域白平衡输入必须是2D数组，当前维度: {image.ndim}")
        
        print(f"Executing Raw White Balance: {algorithm}, Pattern: {bayer_pattern}")
        
        if algorithm == 'gray_world':
            return self._gray_world_raw(image, bayer_pattern)
        elif algorithm == 'gray_world_green':
            return self._gray_world_green_raw(image, bayer_pattern)
        elif algorithm == 'perfect_reflector':
            return self._perfect_reflector_raw(image, bayer_pattern, **kwargs)
        elif algorithm == 'max_white':
            return self._max_white_raw(image, bayer_pattern)
        elif algorithm == 'manual':
            return self._manual_wb_raw(image, bayer_pattern, **kwargs)
        else:
            raise ValueError(f"不支持的白平衡算法: {algorithm}")
    
    def _get_bayer_channels(self, image: np.ndarray, bayer_pattern: str) -> dict:
        """
        提取Bayer图像的R, Gr, Gb, B四个通道
        
        Returns:
            {'R': array, 'Gr': array, 'Gb': array, 'B': array, 
             'slices': {'R': slice_tuple, ...}}
        """
        pattern = bayer_pattern.upper()
        
        # 定义每种模式的位置
        patterns = {
            'RGGB': {'R': (0, 0), 'Gr': (0, 1), 'Gb': (1, 0), 'B': (1, 1)},
            'BGGR': {'B': (0, 0), 'Gb': (0, 1), 'Gr': (1, 0), 'R': (1, 1)},
            'GRBG': {'Gr': (0, 0), 'R': (0, 1), 'B': (1, 0), 'Gb': (1, 1)},
            'GBRG': {'Gb': (0, 0), 'B': (0, 1), 'R': (1, 0), 'Gr': (1, 1)}
        }
        
        if pattern not in patterns:
            raise ValueError(f"不支持的Bayer模式: {pattern}")
        
        layout = patterns[pattern]
        channels = {}
        slices = {}
        
        for color, (row_offset, col_offset) in layout.items():
            slice_tuple = (slice(row_offset, None, 2), slice(col_offset, None, 2))
            channels[color] = image[slice_tuple].copy()
            slices[color] = slice_tuple
        
        return {'channels': channels, 'slices': slices}
    
    def _gray_world_raw(self, image: np.ndarray, bayer_pattern: str) -> np.ndarray:
        """
        标准灰度世界算法（Raw域）
        假设R、G、B的平均值应该相等
        """
        img_float = image.astype(np.float32)
        bayer_data = self._get_bayer_channels(image, bayer_pattern)
        channels = bayer_data['channels']
        slices = bayer_data['slices']
        
        # 计算各通道平均值
        avg_r = np.mean(channels['R'])
        avg_gr = np.mean(channels['Gr'])
        avg_gb = np.mean(channels['Gb'])
        avg_b = np.mean(channels['B'])
        avg_g = (avg_gr + avg_gb) / 2.0
        
        # 全局灰度平均
        gray_avg = (avg_r + avg_g + avg_b) / 3.0
        
        # 计算增益
        gain_r = gray_avg / (avg_r + 1e-6)
        gain_gr = gray_avg / (avg_gr + 1e-6)
        gain_gb = gray_avg / (avg_gb + 1e-6)
        gain_b = gray_avg / (avg_b + 1e-6)
        
        # 限制增益范围
        gain_r = np.clip(gain_r, *self.gain_limits)
        gain_gr = np.clip(gain_gr, *self.gain_limits)
        gain_gb = np.clip(gain_gb, *self.gain_limits)
        gain_b = np.clip(gain_b, *self.gain_limits)
        
        print(f"  Gains - R: {gain_r:.3f}, Gr: {gain_gr:.3f}, Gb: {gain_gb:.3f}, B: {gain_b:.3f}")
        
        # 应用增益
        img_float[slices['R']] *= gain_r
        img_float[slices['Gr']] *= gain_gr
        img_float[slices['Gb']] *= gain_gb
        img_float[slices['B']] *= gain_b
        
        # 裁剪并转换回原类型
        max_val = np.iinfo(image.dtype).max
        return np.clip(img_float, 0, max_val).astype(image.dtype)
    
    def _gray_world_green_raw(self, image: np.ndarray, bayer_pattern: str) -> np.ndarray:
        """
        以绿色为基准的灰度世界算法（推荐）
        保持绿色通道不变，调整R和B，最大化保留动态范围
        
        改进点：
        1. Gr和Gb可能略有不同，先对它们进行轻微均衡
        2. 然后以绿色平均值为基准调整R和B
        """
        img_float = image.astype(np.float32)
        bayer_data = self._get_bayer_channels(image, bayer_pattern)
        channels = bayer_data['channels']
        slices = bayer_data['slices']
        
        # 计算各通道平均值
        avg_r = np.mean(channels['R'])
        avg_gr = np.mean(channels['Gr'])
        avg_gb = np.mean(channels['Gb'])
        avg_b = np.mean(channels['B'])
        
        # 绿色通道均衡（可选，轻微调整Gr和Gb的差异）
        avg_g = (avg_gr + avg_gb) / 2.0
        gain_gr = avg_g / (avg_gr + 1e-6)
        gain_gb = avg_g / (avg_gb + 1e-6)
        
        # 限制绿色通道的调整幅度（只做轻微调整）
        gain_gr = np.clip(gain_gr, 0.95, 1.05)
        gain_gb = np.clip(gain_gb, 0.95, 1.05)
        
        # 以绿色为基准计算R和B的增益
        gain_r = avg_g / (avg_r + 1e-6)
        gain_b = avg_g / (avg_b + 1e-6)
        
        # 限制R和B的增益范围
        gain_r = np.clip(gain_r, *self.gain_limits)
        gain_b = np.clip(gain_b, *self.gain_limits)
        
        print(f"  Gains - R: {gain_r:.3f}, Gr: {gain_gr:.3f}, Gb: {gain_gb:.3f}, B: {gain_b:.3f}")
        
        # 应用增益
        img_float[slices['R']] *= gain_r
        img_float[slices['Gr']] *= gain_gr
        img_float[slices['Gb']] *= gain_gb
        img_float[slices['B']] *= gain_b
        
        # 饱和度保护：检查是否有溢出
        max_val = np.iinfo(image.dtype).max
        overflow_ratio = np.sum(img_float > max_val) / img_float.size * 100
        if overflow_ratio > 1.0:
            print(f"  Warning: {overflow_ratio:.2f}% pixels will be clipped")
        
        return np.clip(img_float, 0, max_val).astype(image.dtype)
    
    def _perfect_reflector_raw(self, image: np.ndarray, bayer_pattern: str,
                               percentile: float = 99.0) -> np.ndarray:
        """
        完美反射法（Perfect Reflector）
        假设场景中最亮的点是白色的
        
        Args:
            percentile: 使用的百分位数（避免极端亮点）
        """
        img_float = image.astype(np.float32)
        bayer_data = self._get_bayer_channels(image, bayer_pattern)
        channels = bayer_data['channels']
        slices = bayer_data['slices']
        
        # 使用高百分位数而非最大值（更robust）
        max_r = np.percentile(channels['R'], percentile)
        max_gr = np.percentile(channels['Gr'], percentile)
        max_gb = np.percentile(channels['Gb'], percentile)
        max_b = np.percentile(channels['B'], percentile)
        max_g = (max_gr + max_gb) / 2.0
        
        # 找到最大通道作为参考
        max_val_all = max(max_r, max_g, max_b)
        
        # 计算增益
        gain_r = max_val_all / (max_r + 1e-6)
        gain_gr = max_val_all / (max_gr + 1e-6)
        gain_gb = max_val_all / (max_gb + 1e-6)
        gain_b = max_val_all / (max_b + 1e-6)
        
        # 限制增益
        gain_r = np.clip(gain_r, *self.gain_limits)
        gain_gr = np.clip(gain_gr, *self.gain_limits)
        gain_gb = np.clip(gain_gb, *self.gain_limits)
        gain_b = np.clip(gain_b, *self.gain_limits)
        
        print(f"  Gains - R: {gain_r:.3f}, Gr: {gain_gr:.3f}, Gb: {gain_gb:.3f}, B: {gain_b:.3f}")
        
        # 应用增益
        img_float[slices['R']] *= gain_r
        img_float[slices['Gr']] *= gain_gr
        img_float[slices['Gb']] *= gain_gb
        img_float[slices['B']] *= gain_b
        
        max_val = np.iinfo(image.dtype).max
        return np.clip(img_float, 0, max_val).astype(image.dtype)
    
    def _max_white_raw(self, image: np.ndarray, bayer_pattern: str) -> np.ndarray:
        """
        最大白法（Max White）
        将最亮的通道归一化，其他通道按比例调整
        """
        img_float = image.astype(np.float32)
        bayer_data = self._get_bayer_channels(image, bayer_pattern)
        channels = bayer_data['channels']
        slices = bayer_data['slices']
        
        # 找到每个通道的最大值
        max_r = np.max(channels['R'])
        max_gr = np.max(channels['Gr'])
        max_gb = np.max(channels['Gb'])
        max_b = np.max(channels['B'])
        
        # 找到最大通道
        max_all = max(max_r, max_gr, max_gb, max_b)
        
        # 计算增益
        gain_r = max_all / (max_r + 1e-6)
        gain_gr = max_all / (max_gr + 1e-6)
        gain_gb = max_all / (max_gb + 1e-6)
        gain_b = max_all / (max_b + 1e-6)
        
        # 限制增益
        gain_r = np.clip(gain_r, *self.gain_limits)
        gain_gr = np.clip(gain_gr, *self.gain_limits)
        gain_gb = np.clip(gain_gb, *self.gain_limits)
        gain_b = np.clip(gain_b, *self.gain_limits)
        
        print(f"  Gains - R: {gain_r:.3f}, Gr: {gain_gr:.3f}, Gb: {gain_gb:.3f}, B: {gain_b:.3f}")
        
        # 应用增益
        img_float[slices['R']] *= gain_r
        img_float[slices['Gr']] *= gain_gr
        img_float[slices['Gb']] *= gain_gb
        img_float[slices['B']] *= gain_b
        
        max_val = np.iinfo(image.dtype).max
        return np.clip(img_float, 0, max_val).astype(image.dtype)
    
    def _manual_wb_raw(self, image: np.ndarray, bayer_pattern: str,
                      gain_r: float = 1.0, gain_g: float = 1.0, 
                      gain_b: float = 1.0) -> np.ndarray:
        """
        手动指定增益
        
        Args:
            gain_r: R通道增益
            gain_g: G通道增益
            gain_b: B通道增益
        """
        img_float = image.astype(np.float32)
        bayer_data = self._get_bayer_channels(image, bayer_pattern)
        slices = bayer_data['slices']
        
        # 限制增益
        gain_r = np.clip(gain_r, *self.gain_limits)
        gain_g = np.clip(gain_g, *self.gain_limits)
        gain_b = np.clip(gain_b, *self.gain_limits)
        
        print(f"  Manual Gains - R: {gain_r:.3f}, G: {gain_g:.3f}, B: {gain_b:.3f}")
        
        # 应用增益
        img_float[slices['R']] *= gain_r
        img_float[slices['Gr']] *= gain_g
        img_float[slices['Gb']] *= gain_g
        img_float[slices['B']] *= gain_b
        
        max_val = np.iinfo(image.dtype).max
        return np.clip(img_float, 0, max_val).astype(image.dtype)
    
    def compute_wb_gains(self, image: np.ndarray, bayer_pattern: str,
                        algorithm: str = 'gray_world_green') -> dict:
        """
        仅计算白平衡增益，不应用到图像
        用于调试或保存增益参数
        
        Returns:
            {'R': gain_r, 'Gr': gain_gr, 'Gb': gain_gb, 'B': gain_b}
        """
        bayer_data = self._get_bayer_channels(image, bayer_pattern)
        channels = bayer_data['channels']
        
        avg_r = np.mean(channels['R'])
        avg_gr = np.mean(channels['Gr'])
        avg_gb = np.mean(channels['Gb'])
        avg_b = np.mean(channels['B'])
        avg_g = (avg_gr + avg_gb) / 2.0
        
        if algorithm == 'gray_world_green':
            gain_r = avg_g / (avg_r + 1e-6)
            gain_b = avg_g / (avg_b + 1e-6)
            gain_gr = 1.0
            gain_gb = 1.0
        else:
            gray_avg = (avg_r + avg_g + avg_b) / 3.0
            gain_r = gray_avg / (avg_r + 1e-6)
            gain_gr = gray_avg / (avg_gr + 1e-6)
            gain_gb = gray_avg / (avg_gb + 1e-6)
            gain_b = gray_avg / (avg_b + 1e-6)
        
        gains = {
            'R': np.clip(gain_r, *self.gain_limits),
            'Gr': np.clip(gain_gr, *self.gain_limits),
            'Gb': np.clip(gain_gb, *self.gain_limits),
            'B': np.clip(gain_b, *self.gain_limits)
        }
        
        return gains


# class WhiteBalance:
#     """
#     白平衡处理模块
#     包含多种白平衡算法。
#     """
#     def _gray_world(self, image: np.ndarray) -> np.ndarray:
#         """灰度世界算法"""
#         # 将图像转换为浮点数进行计算
#         img_f = image.astype(np.float32)
        
#         # 计算每个通道的平均值
#         r_avg = np.mean(img_f[:, :, 0])
#         g_avg = np.mean(img_f[:, :, 1])
#         b_avg = np.mean(img_f[:, :, 2])
        
#         # 计算所有通道的全局平均值（灰度值）
#         gray_avg = (r_avg + g_avg + b_avg) / 3
        
#         # 计算每个通道的增益
#         r_gain = gray_avg / r_avg
#         g_gain = gray_avg / g_avg
#         b_gain = gray_avg / b_avg
        
#         # 应用增益
#         img_f[:, :, 0] *= r_gain
#         img_f[:, :, 1] *= g_gain
#         img_f[:, :, 2] *= b_gain
        
#         # 裁剪到有效范围并转换回原始数据类型
#         # 注意：假设输入是16位图像，最大值为65535
#         max_val = np.iinfo(image.dtype).max
#         return np.clip(img_f, 0, max_val).astype(image.dtype)
    
#     def _gray_world_green(self, image: np.ndarray) -> np.ndarray:
#         """灰度世界算法 (以绿色通道为基准)"""
#         img_f = image.astype(np.float32)
        
#         # 计算每个通道的平均值
#         r_avg = np.mean(img_f[:, :, 0])
#         g_avg = np.mean(img_f[:, :, 1])
#         b_avg = np.mean(img_f[:, :, 2])
        
#         # 以绿色通道为基准
#         r_gain = g_avg / r_avg
#         g_gain = 1.0  # 绿色通道不变
#         b_gain = g_avg / b_avg

#         # 限制增益范围，避免过度
#         b_gain = np.clip(b_gain, 0.5, 2.0)
#         r_gain = np.clip(r_gain, 0.5, 2.0)

#         # 应用增益
#         img_f[:, :, 0] *= r_gain
#         img_f[:, :, 1] *= g_gain
#         img_f[:, :, 2] *= b_gain
        
#         # 根据输入图像image的数据类型裁剪并转换
#         max_val = np.iinfo(image.dtype).max
#         return np.clip(img_f, 0, max_val).astype(image.dtype)

#     def _perfect_reflector(self, image: np.ndarray, percentile: float = 99.5) -> np.ndarray:
#         """完美反射算法 (也叫白点算法)"""
#         img_f = image.astype(np.float32)
        
#         # 找到每个通道的“最亮点”
#         # 我们使用百分位数来避免噪点或过曝区域的影响
#         r_white = np.percentile(img_f[:, :, 0], percentile)
#         g_white = np.percentile(img_f[:, :, 1], percentile)
#         b_white = np.percentile(img_f[:, :, 2], percentile)
        
#         # 假设最亮的点是白色 (R=G=B=max_val)
#         # 我们以G通道为基准（通常G通道信噪比最好）
#         # 或者以图像类型的最大值为基准
#         max_val = np.iinfo(image.dtype).max
        
#         r_gain = max_val / r_white
#         g_gain = max_val / g_white
#         b_gain = max_val / b_white
        
#         # 应用增益
#         img_f[:, :, 0] *= r_gain
#         img_f[:, :, 1] *= g_gain
#         img_f[:, :, 2] *= b_gain
        
#         return np.clip(img_f, 0, max_val).astype(image.dtype)


#     def execute(self, rgb_image: np.ndarray, algorithm: str = 'gray_world', **kwargs) -> np.ndarray:
#         """
#         执行白平衡操作。
        
#         Args:
#             rgb_image: 输入的RGB图像 (Numpy array)。
#             algorithm: 'gray_world' 或 'perfect_reflector'。
#             **kwargs: 传递给特定算法的额外参数 (例如 'percentile' for perfect_reflector)。

#         Returns:
#             经过白平衡处理的RGB图像。
#         """
#         print(f"Executing White Balance with algorithm: {algorithm}")
#         if algorithm == 'gray_world':
#             return self._gray_world(rgb_image)
#         elif algorithm == 'perfect_reflector':
#             percentile = kwargs.get('percentile', 99.5)
#             return self._perfect_reflector(rgb_image, percentile)
#         else:
#             raise ValueError(f"不支持的白平衡算法: {algorithm}")
