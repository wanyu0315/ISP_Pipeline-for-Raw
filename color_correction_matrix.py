# color_correction_matrix.py
import numpy as np
import cv2


class ColorCorrectionMatrix:
    """
    颜色校正矩阵（CCM）模块
    将传感器的原始RGB色彩空间转换到标准色彩空间（如sRGB）
    """
    
    def __init__(self):
        """初始化CCM模块，预定义常用的校正矩阵"""
        # 预定义的标准CCM矩阵（示例）
        self.predefined_matrices = {
            'identity': np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]),
            
            # 典型相机传感器到sRGB的转换矩阵（示例）
            'sensor_to_srgb': np.array([
                [ 1.5, -0.3, -0.2],
                [-0.2,  1.3, -0.1],
                [-0.1, -0.4,  1.5]
            ]),
            
            # D65光源下的典型CCM
            'd65_standard': np.array([
                [ 1.6, -0.4, -0.2],
                [-0.3,  1.4, -0.1],
                [-0.1, -0.5,  1.6]
            ]),
            
            # 更温暖的色调
            'warm_tone': np.array([
                [ 1.3, -0.2, -0.1],
                [-0.1,  1.2, -0.1],
                [-0.2, -0.3,  1.5]
            ]),
            
            # 更冷的色调
            'cool_tone': np.array([
                [ 1.4, -0.3, -0.1],
                [-0.2,  1.3, -0.1],
                [-0.1, -0.2,  1.3]
            ]),
            
            # 增强饱和度的CCM
            'vibrant': np.array([
                [ 1.8, -0.5, -0.3],
                [-0.4,  1.6, -0.2],
                [-0.2, -0.6,  1.8]
            ]),

            # 光源 A (白炽灯/钨丝灯, ~2856K)
            # 修正钨丝灯的极度偏黄，大幅提升蓝色增益，抑制红色
            'illuminant_a': np.array([
                [ 1.25, -0.15, -0.10],
                [-0.35,  1.45, -0.10],
                [-0.10, -0.65,  1.75]
            ]),

            # 光源 TL84/F11 (商场/办公室常见的冷白荧光灯, ~4000K)
            # 修正荧光灯特有的“绿调”
            'fluorescent_tl84': np.array([
                [ 1.55, -0.45, -0.10],
                [-0.15,  1.35, -0.20],
                [-0.10, -0.30,  1.40]
            ]),

            # ==========================================
            # rPPG 研究专用矩阵 (Signal Enhancement)
            # ==========================================

            # rPPG - 绿色光谱锐化 (Spectral Sharpening)
            # 目的：减少 R 和 B 对 G 通道的串扰，让 G 通道更纯净（含氧血红蛋白吸收峰在 G）
            # 这是一个强分离矩阵，会增加噪声但提升信号分离度
            'rppg_green_isolation': np.array([
                [ 1.0,  0.0,  0.0],
                [-0.3,  1.6, -0.3], # 强行扣除 R/B 分量，突出 G
                [ 0.0,  0.0,  1.0]
            ]),

            # rPPG - 血液对比度增强 (Hemoglobin Contrast)
            # 实验性矩阵：尝试拉大 G 通道与 R 通道的反差
            # 这种非自然色彩对人眼不友好，但对机器视觉提取脉搏可能有效
            'rppg_high_contrast': np.array([
                [ 1.0, -0.5,  0.0], # 压暗 R 中混入的 G
                [-0.2,  1.4, -0.2], # 提亮 G
                [ 0.0, -0.5,  1.0]  # 压暗 B 中混入的 G
            ]),
        }
    
    def execute(self, rgb_image: np.ndarray, method: str = 'sensor_to_srgb', 
                custom_matrix: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        执行颜色校正矩阵变换。
        
        Args:
            rgb_image: 输入的RGB图像（白平衡后的线性RGB）
            method: CCM矩阵选择
                - 'identity': 恒等变换（不做任何改变）
                - 'sensor_to_srgb': 传感器到sRGB标准转换
                - 'd65_standard': D65光源标准CCM
                - 'warm_tone': 暖色调
                - 'cool_tone': 冷色调
                - 'vibrant': 高饱和度
                - 'rppg_green_isolation': rPPG绿色光谱锐化
                - 'rppg_high_contrast': rPPG血液对比度增强
                - 'custom': 使用自定义矩阵
                - 'auto_calibration': 自动校准（基于色卡）
                - 'least_squares': 最小二乘优化
            custom_matrix: 自定义的3×3 CCM矩阵
            **kwargs: 其他参数
        
        Returns:
            校正后的RGB图像
        """
        print(f"Executing Color Correction Matrix using method: {method}")
        
        if method == 'custom':
            if custom_matrix is None:
                raise ValueError("Custom matrix must be provided when method='custom'")
            ccm_matrix = custom_matrix
        elif method == 'auto_calibration':
            return self._auto_calibration_ccm(rgb_image, **kwargs)
        elif method == 'least_squares':
            return self._least_squares_ccm(rgb_image, **kwargs)
        elif method in self.predefined_matrices:
            ccm_matrix = self.predefined_matrices[method]
        else:
            raise ValueError(f"Unknown CCM method: {method}")
        
        return self._apply_ccm(rgb_image, ccm_matrix, **kwargs)
    
    def _apply_ccm(self, rgb_image: np.ndarray, ccm_matrix: np.ndarray, 
                   clip: bool = True) -> np.ndarray:
        """
        应用CCM矩阵到RGB图像
        
        Args:
            rgb_image: 输入RGB图像
            ccm_matrix: 3×3 CCM矩阵
            clip: 是否裁剪到有效范围
        
        Returns:
            校正后的RGB图像
        """
        # 归一化到 [0, 1]
        if rgb_image.dtype == np.uint16:
            max_val = 65535.0
            img_float = rgb_image.astype(np.float32) / max_val
        else:
            max_val = 255.0
            img_float = rgb_image.astype(np.float32) / max_val
        
        # 重塑为 (H*W, 3) 以进行矩阵乘法
        h, w, c = img_float.shape
        img_reshaped = img_float.reshape(-1, 3)
        
        # 应用CCM: RGB_out = CCM × RGB_in
        corrected = np.dot(img_reshaped, ccm_matrix.T)
        
        # 重塑回原始形状
        corrected = corrected.reshape(h, w, c)
        
        # 裁剪到有效范围
        if clip:
            corrected = np.clip(corrected, 0, 1)
        
        # 转换回原始数据类型
        return (corrected * max_val).astype(rgb_image.dtype)
    
    def _auto_calibration_ccm(self, rgb_image: np.ndarray, 
                             reference_patches: np.ndarray = None,
                             measured_patches: np.ndarray = None) -> np.ndarray:
        """
        基于色卡的自动CCM校准
        计算出CCM后直接应用校正
        
        Args:
            rgb_image: 输入RGB图像
            reference_patches: 参考色块的RGB值 (N, 3)，标准sRGB值
            measured_patches: 实际测量的色块RGB值 (N, 3)，是用来校准的“实际值”或“原始素材”，需要自己拍摄上传
        
        Returns:
            校正后的RGB图像
        """
        if reference_patches is None or measured_patches is None:
            print("Warning: No color chart data provided, using default CCM")
            return self._apply_ccm(rgb_image, self.predefined_matrices['sensor_to_srgb'])
        
        # 使用最小二乘法计算CCM
        # 目标: reference = CCM × measured
        # CCM = reference × measured^(-1)
        
        # 添加偏置项以处理黑电平
        measured_augmented = np.hstack([measured_patches, np.ones((measured_patches.shape[0], 1))])
        
        # 最小二乘求解
        ccm_augmented, residuals, rank, s = np.linalg.lstsq(
            measured_augmented, reference_patches, rcond=None
        )
        
        # 提取3×3矩阵
        ccm_matrix = ccm_augmented[:3, :].T
        
        print(f"Calibrated CCM matrix:\n{ccm_matrix}")
        print(f"Residual error: {np.sqrt(np.mean(residuals)) if len(residuals) > 0 else 'N/A'}")
        
        return self._apply_ccm(rgb_image, ccm_matrix)
    
    def _least_squares_ccm(self, rgb_image: np.ndarray,
                          target_illuminant: str = 'd65',
                          sensor_sensitivity: np.ndarray = None) -> np.ndarray:
        """
        基于传感器光谱响应的最小二乘CCM优化
        
        Args:
            rgb_image: 输入RGB图像
            target_illuminant: 目标光源 ('d65', 'a', 'd50')
            sensor_sensitivity: 传感器光谱灵敏度曲线 (可选)
        
        Returns:
            校正后的RGB图像
        """
        # 这是一个简化版本，实际应用中需要传感器的光谱响应数据
        print("Warning: Using approximate least squares CCM")
        
        # 使用预定义的近似矩阵
        if target_illuminant.lower() == 'd65':
            ccm_matrix = self.predefined_matrices['d65_standard']
        else:
            ccm_matrix = self.predefined_matrices['sensor_to_srgb']
        
        return self._apply_ccm(rgb_image, ccm_matrix)
    
    def compute_ccm_from_colorchecker(self, measured_rgb: np.ndarray, 
                                     reference_rgb: np.ndarray = None) -> np.ndarray:
        """
        从ColorChecker色卡数据计算CCM矩阵
        这是一个独立的工具函数。它的设计目的是在**校准阶段（离线）**使用。可在外部使用这个函数求出基于标准色卡的CCM矩阵。

        Args:
            measured_rgb: 实际拍摄的24个色块的RGB值 (24, 3)
            reference_rgb: 标准ColorChecker的RGB值 (24, 3)，如果为None则使用内置标准值
        
        Returns:
            计算得到的3×3 CCM矩阵
        """
        if reference_rgb is None:
            # ColorChecker Classic 24色卡的标准sRGB值（归一化到0-1）
            reference_rgb = np.array([
                [0.45, 0.31, 0.25],  # 1. 深棕色 (Dark Skin)
                [0.77, 0.57, 0.48],  # 2. 浅棕色 (Light Skin)
                [0.34, 0.42, 0.61],  # 3. 蓝天 (Blue Sky)
                [0.29, 0.37, 0.23],  # 4. 叶绿色 (Foliage)
                [0.47, 0.47, 0.69],  # 5. 蓝花 (Blue Flower)
                [0.42, 0.74, 0.63],  # 6. 蓝绿色 (Bluish Green)
                [0.87, 0.46, 0.19],  # 7. 橙色 (Orange)
                [0.26, 0.32, 0.62],  # 8. 紫蓝色 (Purplish Blue)
                [0.79, 0.33, 0.37],  # 9. 中等红色 (Moderate Red)
                [0.29, 0.17, 0.31],  # 10. 紫色 (Purple)
                [0.47, 0.69, 0.26],  # 11. 黄绿色 (Yellow Green)
                [0.87, 0.57, 0.18],  # 12. 橙黄色 (Orange Yellow)
                [0.15, 0.20, 0.48],  # 13. 蓝色 (Blue)
                [0.26, 0.57, 0.30],  # 14. 绿色 (Green)
                [0.67, 0.19, 0.20],  # 15. 红色 (Red)
                [0.91, 0.70, 0.13],  # 16. 黄色 (Yellow)
                [0.75, 0.27, 0.57],  # 17. 品红 (Magenta)
                [0.14, 0.40, 0.49],  # 18. 青色 (Cyan)
                [0.96, 0.96, 0.96],  # 19. 白色 (White)
                [0.78, 0.78, 0.78],  # 20. 中性8 (Neutral 8)
                [0.62, 0.62, 0.62],  # 21. 中性6.5 (Neutral 6.5)
                [0.44, 0.44, 0.44],  # 22. 中性5 (Neutral 5)
                [0.21, 0.21, 0.21],  # 23. 中性3.5 (Neutral 3.5)
                [0.06, 0.06, 0.06],  # 24. 黑色 (Black)
            ])
        
        # 确保输入数据是归一化的
        if measured_rgb.max() > 1.0:
            measured_rgb = measured_rgb / 255.0
        
        # 最小二乘求解: reference = CCM × measured
        ccm_matrix, residuals, rank, s = np.linalg.lstsq(
            measured_rgb, reference_rgb, rcond=None
        )
        
        ccm_matrix = ccm_matrix.T
        
        print(f"Computed CCM matrix from ColorChecker:")
        print(ccm_matrix)
        if len(residuals) > 0:
            print(f"Average color error (ΔE): {np.sqrt(np.mean(residuals)):.4f}")
        
        return ccm_matrix
    
    def evaluate_ccm(self, ccm_matrix: np.ndarray, 
                    measured_rgb: np.ndarray, 
                    reference_rgb: np.ndarray) -> dict:
        """
        评估CCM矩阵的性能
        
        Args:
            ccm_matrix: 待评估的CCM矩阵
            measured_rgb: 实测RGB值
            reference_rgb: 参考RGB值
        
        Returns:
            包含评估指标的字典
        """
        # 应用CCM
        corrected_rgb = np.dot(measured_rgb, ccm_matrix.T)
        
        # 计算色差 (ΔE)
        delta_e = np.sqrt(np.sum((corrected_rgb - reference_rgb)**2, axis=1))
        
        # 计算统计指标
        results = {
            'mean_delta_e': np.mean(delta_e),
            'max_delta_e': np.max(delta_e),
            'std_delta_e': np.std(delta_e),
            'median_delta_e': np.median(delta_e),
            'percentile_95_delta_e': np.percentile(delta_e, 95)
        }
        
        print("\n=== CCM Performance Evaluation ===")
        print(f"Mean ΔE:     {results['mean_delta_e']:.4f}")
        print(f"Median ΔE:   {results['median_delta_e']:.4f}")
        print(f"Max ΔE:      {results['max_delta_e']:.4f}")
        print(f"Std ΔE:      {results['std_delta_e']:.4f}")
        print(f"95th %ile ΔE: {results['percentile_95_delta_e']:.4f}")
        
        return results
    
    def adaptive_ccm(self, rgb_image: np.ndarray, 
                    scene_illuminant: str = 'auto') -> np.ndarray:
        """
        自适应CCM - 根据场景光源自动选择CCM
        
        Args:
            rgb_image: 输入RGB图像
            scene_illuminant: 场景光源类型
                - 'auto': 自动检测
                - 'd65': 日光
                - 'a': 白炽灯
                - 'f': 荧光灯
        
        Returns:
            校正后的RGB图像
        """
        if scene_illuminant == 'auto':
            # 简单的光源检测：基于图像的平均色温
            avg_rgb = np.mean(rgb_image.reshape(-1, 3), axis=0)
            if avg_rgb.dtype == np.uint16:
                avg_rgb = avg_rgb / 65535.0
            else:
                avg_rgb = avg_rgb / 255.0
            
            # 色温估计（简化）
            color_temp_ratio = avg_rgb[2] / (avg_rgb[0] + 1e-6)  # B/R比值
            
            if color_temp_ratio > 1.1:  # 偏蓝，高色温
                ccm_matrix = self.predefined_matrices['d65_standard']
                print("Detected: Daylight (D65)")
            elif color_temp_ratio < 0.9:  # 偏红，低色温
                ccm_matrix = self.predefined_matrices['warm_tone']
                print("Detected: Warm light")
            else:
                ccm_matrix = self.predefined_matrices['sensor_to_srgb']
                print("Detected: Neutral light")
        else:
            ccm_matrix = self.predefined_matrices.get(
                scene_illuminant, 
                self.predefined_matrices['sensor_to_srgb']
            )
        
        return self._apply_ccm(rgb_image, ccm_matrix)
    
    def set_custom_matrix(self, name: str, matrix: np.ndarray):
        """
        添加自定义CCM矩阵到预定义矩阵库
        
        Args:
            name: 矩阵名称
            matrix: 3×3 CCM矩阵
        """
        if matrix.shape != (3, 3):
            raise ValueError("CCM matrix must be 3×3")
        
        self.predefined_matrices[name] = matrix
        print(f"Added custom CCM matrix '{name}' to library")