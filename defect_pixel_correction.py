import numpy as np
from scipy.ndimage import median_filter

# ===================================================================
# 坏点校正 (Defect Pixel Correction - DPC)
# ===================================================================
class DefectPixelCorrection:
    """
    坏点校正模块
    检测并修复Dead/Hot/Stuck像素
    """
    
    def execute(self, raw_data: np.ndarray,
                method: str = 'median',
                defect_map: np.ndarray = None,
                auto_detect: bool = True,
                **kwargs) -> np.ndarray:
        """
        执行坏点校正
        
        Args:
            raw_data: 输入Raw数据
            method: 校正方法
                - 'median': 中值滤波替换
                - 'mean': 均值替换
                - 'gradient': 基于梯度的插值
            defect_map: 预先标定的坏点位置图（bool数组，True表示坏点）
            auto_detect: 是否自动检测坏点
        
        Returns:
            校正后的Raw数据
        """
        print(f"Executing Defect Pixel Correction using method: {method}")
        
        # 1. 获取坏点位置
        if defect_map is not None:
            print(f"  Using provided defect map: {np.sum(defect_map)} defects")
        elif auto_detect:
            defect_map = self._detect_defects(raw_data, **kwargs)
            print(f"  Auto-detected: {np.sum(defect_map)} defects")
        else:
            print("  No defects detected/provided")
            return raw_data
        
        # 2. 执行校正
        if method == 'median':
            corrected = self._median_correction(raw_data, defect_map)
        elif method == 'mean':
            corrected = self._mean_correction(raw_data, defect_map)
        elif method == 'gradient':
            corrected = self._gradient_correction(raw_data, defect_map)
        else:
            raise ValueError(f"Unknown DPC method: {method}")
        
        return corrected
    
    def _detect_defects(self, raw_data: np.ndarray,
                       threshold_factor: float = 3.0) -> np.ndarray:
        """
        自动检测坏点
        
        Args:
            threshold_factor: 检测阈值因子（倍数于局部标准差）
        
        Returns:
            坏点掩模（bool数组）
        """
        # 计算局部统计量
        from scipy.ndimage import uniform_filter
        
        window_size = 5
        local_mean = uniform_filter(raw_data.astype(np.float32), size=window_size)
        local_mean_sq = uniform_filter(raw_data.astype(np.float32)**2, size=window_size)
        local_var = local_mean_sq - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # 检测异常值
        deviation = np.abs(raw_data.astype(np.float32) - local_mean)
        threshold = threshold_factor * (local_std + 1.0)  # +1避免除零
        
        defect_map = deviation > threshold
        
        # 额外检测极端值（Dead/Hot pixels）
        max_val = np.iinfo(raw_data.dtype).max
        defect_map |= (raw_data == 0)  # Dead pixels
        defect_map |= (raw_data >= max_val * 0.98)  # Hot pixels
        
        return defect_map
    
    def _median_correction(self, raw_data: np.ndarray,
                          defect_map: np.ndarray) -> np.ndarray:
        """
        使用中值滤波修复坏点
        最常用的方法，robust且效果好
        """
        corrected = raw_data.copy()
        
        # 对每个坏点，用5×5邻域的中值替换
        defect_coords = np.where(defect_map)
        
        for y, x in zip(*defect_coords):
            # 提取邻域（排除坏点本身）
            y_min = max(0, y - 2)
            y_max = min(raw_data.shape[0], y + 3)
            x_min = max(0, x - 2)
            x_max = min(raw_data.shape[1], x + 3)
            
            neighborhood = raw_data[y_min:y_max, x_min:x_max].copy()
            
            # 计算中值（排除其他坏点）
            valid_pixels = neighborhood[~defect_map[y_min:y_max, x_min:x_max]]
            
            if len(valid_pixels) > 0:
                corrected[y, x] = np.median(valid_pixels)
        
        return corrected
    
    def _mean_correction(self, raw_data: np.ndarray,
                        defect_map: np.ndarray) -> np.ndarray:
        """
        使用均值替换坏点
        比中值快，但对极端值不够robust
        """
        corrected = raw_data.copy()
        defect_coords = np.where(defect_map)
        
        for y, x in zip(*defect_coords):
            y_min = max(0, y - 1)
            y_max = min(raw_data.shape[0], y + 2)
            x_min = max(0, x - 1)
            x_max = min(raw_data.shape[1], x + 2)
            
            neighborhood = raw_data[y_min:y_max, x_min:x_max].copy()
            valid_pixels = neighborhood[~defect_map[y_min:y_max, x_min:x_max]]
            
            if len(valid_pixels) > 0:
                corrected[y, x] = int(np.mean(valid_pixels))
        
        return corrected
    
    def _gradient_correction(self, raw_data: np.ndarray,
                            defect_map: np.ndarray) -> np.ndarray:
        """
        基于梯度的插值修复
        考虑边缘方向，更好地保留细节
        """
        corrected = raw_data.copy().astype(np.float32)
        defect_coords = np.where(defect_map)
        
        for y, x in zip(*defect_coords):
            if y == 0 or y == raw_data.shape[0]-1 or x == 0 or x == raw_data.shape[1]-1:
                # 边界像素用简单均值
                neighbors = []
                if y > 0: neighbors.append(raw_data[y-1, x])
                if y < raw_data.shape[0]-1: neighbors.append(raw_data[y+1, x])
                if x > 0: neighbors.append(raw_data[y, x-1])
                if x < raw_data.shape[1]-1: neighbors.append(raw_data[y, x+1])
                
                if neighbors:
                    corrected[y, x] = np.mean(neighbors)
            else:
                # 计算水平和垂直梯度
                h_grad = abs(raw_data[y, x-1] - raw_data[y, x+1])
                v_grad = abs(raw_data[y-1, x] - raw_data[y+1, x])
                
                # 选择梯度小的方向进行插值（保留边缘）
                if h_grad < v_grad:
                    # 水平方向平滑，使用水平邻居
                    corrected[y, x] = (raw_data[y, x-1] + raw_data[y, x+1]) / 2.0
                else:
                    # 垂直方向平滑，使用垂直邻居
                    corrected[y, x] = (raw_data[y-1, x] + raw_data[y+1, x]) / 2.0
        
        return corrected.astype(raw_data.dtype)
    
    def create_defect_map_from_dark_frame(self, dark_frame: np.ndarray,
                                         threshold: int = None) -> np.ndarray:
        """
        从暗场图像创建坏点图
        
        Args:
            dark_frame: 盖上镜头盖拍摄的暗场图像
            threshold: 检测阈值，None则自动确定
        
        Returns:
            坏点掩模
        """
        if threshold is None:
            # 自动确定阈值：使用3倍标准差
            threshold = int(np.mean(dark_frame) + 3 * np.std(dark_frame))
        
        # Hot pixels: 暗场中仍然很亮的像素
        hot_pixels = dark_frame > threshold
        
        # Dead pixels: 始终为0
        dead_pixels = dark_frame == 0
        
        defect_map = hot_pixels | dead_pixels
        
        print(f"Created defect map from dark frame")
        print(f"  Hot pixels: {np.sum(hot_pixels)}")
        print(f"  Dead pixels: {np.sum(dead_pixels)}")
        print(f"  Total defects: {np.sum(defect_map)}")
        
        return defect_map