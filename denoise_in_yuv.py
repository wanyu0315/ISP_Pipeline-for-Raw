# denoise.py
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# 尝试导入可选库
try:
    import pywt
    _pywt_available = True
except ImportError:
    _pywt_available = False

class Denoise:
    """
    降噪模块：提供多种降噪算法。
    
    所有方法均假设输入为 np.float32 [0, 1] YUV图像，
    输出也为 np.float32 [0, 1] YUV图像。
    """
    
    def execute(self, yuv_image: np.ndarray, algorithm: str = 'bilateral', **kwargs) -> np.ndarray:
        """
        执行图像降噪。
        
        Args:
            yuv_image: 输入的YUV图像 (np.float32, 范围 [0, 1])
            algorithm: 降噪算法选择
                'gaussian': 高斯滤波 (float32 兼容)
                'bilateral': 双边滤波 (保边降噪, 需转uint8)
                'nlm': 非局部均值降噪 (需转uint8)
                'nlm_fast': 快速非局部均值 (需转uint8)
                'median': 中值滤波 (需转uint8)
                'wavelet': 小波降噪 (float32 兼容)
                'anisotropic': 各向异性扩散 (float32 兼容)
            **kwargs: 算法特定参数
        
        Returns:
            降噪后的YUV图像 (np.float32, 范围 [0, 1])
        """
        print(f"Executing Denoising using algorithm: {algorithm}")
        
        # 确保输入是 float32
        if yuv_image.dtype != np.float32:
            print(f"警告: Denoise 模块接收到非 float32 图像 (dtype: {yuv_image.dtype})。")
            # 假设它来自 uint16
            if yuv_image.max() > 1.0:
                 yuv_image = yuv_image.astype(np.float32) / 65535.0
            else:
                 yuv_image = yuv_image.astype(np.float32)

        
        if algorithm == 'gaussian':
            return self._gaussian_denoise(yuv_image, **kwargs)
        elif algorithm == 'bilateral':
            return self._bilateral_denoise(yuv_image, **kwargs)
        elif algorithm == 'nlm':
            return self._nlm_denoise(yuv_image, **kwargs)
        elif algorithm == 'nlm_fast':
            return self._nlm_fast_denoise(yuv_image, **kwargs)
        elif algorithm == 'median':
            return self._median_denoise(yuv_image, **kwargs)
        elif algorithm == 'wavelet':
            return self._wavelet_denoise(yuv_image, **kwargs)
        elif algorithm == 'anisotropic':
            return self._anisotropic_denoise(yuv_image, **kwargs)
        else:
            raise ValueError(f"Unknown denoising algorithm: {algorithm}")
    
    def _gaussian_denoise(self, image: np.ndarray, sigma: float = 1.0, 
                          process_chroma: bool = True) -> np.ndarray:
        """
        高斯滤波降噪 - (float32 兼容)
        
        Args:
            image: float32 [0, 1] 图像
            sigma: 高斯核标准差
            process_chroma: 是否处理色度(U, V)通道
        """
        y, u, v = cv2.split(image)
        
        # 1. 总是处理 Y (亮度) 通道
        y_denoised = gaussian_filter(y, sigma=sigma)
        
        # 2. 可选处理 U, V (色度) 通道
        if process_chroma:
            # 色度通道通常使用较弱的降噪
            chroma_sigma = max(0.5, sigma * 0.5) 
            u_denoised = gaussian_filter(u, sigma=chroma_sigma)
            v_denoised = gaussian_filter(v, sigma=chroma_sigma)
        else:
            u_denoised, v_denoised = u, v
            
        denoised_image = cv2.merge([y_denoised, u_denoised, v_denoised])
        
        # 滤波可能会产生轻微超出 [0, 1] 范围的值
        return np.clip(denoised_image, 0, 1)
    
    def _bilateral_denoise(self, image: np.ndarray, d: int = 9, 
                           sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        双边滤波 - 保边降噪，适合保留细节(需转uint8)
        Args：
            d: 滤波器直径
            sigma_color: 颜色空间标准差
            sigma_space: 坐标空间标准差

        注意: sigma_color 参数 (如 75) 是针对 [0, 255] 范围设计的。
              在 [0, 1] 范围上应用它没有意义，因此必须进行局部转换。
        """
        ## --- 兼容性核心：执行局部 float32 -> uint8 -> float32 转换 ---
        # 1. 转换为 [0, 255] uint8
        img_uint8 = (image * 255.0).astype(np.uint8)
        
        # 2. 在 uint8 上执行处理
        denoised_uint8 = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)
        
        # 3. 转换回 [0, 1] float32
        return denoised_uint8.astype(np.float32) / 255.0
        ## --- 转换结束 ---
    
    def _nlm_denoise(self, image: np.ndarray, h: float = 10, 
                     template_window_size: int = 7, search_window_size: int = 21) -> np.ndarray:
        """
        非局部均值降噪 - - 高质量但较慢(需转uint8)

        Args:
            h: 滤波强度
            template_window_size: 模板窗口大小
            search_window_size: 搜索窗口大小
        
        注意: OpenCV 的 NLM 实现 (尤其是彩色) 期望 uint8 输入。
        """
        ## --- 兼容性核心：执行局部 float32 -> uint8 -> float32 转换 ---
        # 1. 转换为 [0, 255] uint8
        img_uint8 = (image * 255.0).astype(np.uint8)
        
        # 2. 在 uint8 上执行处理 (使用彩色版本以匹配原始代码)
        denoised_uint8 = cv2.fastNlMeansDenoisingColored(
            img_uint8, None, h, h, template_window_size, search_window_size
        )
        
        # 3. 转换回 [0, 1] float32
        return denoised_uint8.astype(np.float32) / 255.0
        ## --- 转换结束 ---
    
    def _nlm_fast_denoise(self, image: np.ndarray, h: float = 10) -> np.ndarray:
        """
        快速非局部均值降噪 - 平衡速度和质量(需转uint8)
        
        注意: OpenCV 的 NLM 实现 (尤其是彩色) 期望 uint8 输入。
        """
        ## --- 兼容性核心：执行局部 float32 -> uint8 -> float32 转换 ---
        # 1. 转换为 [0, 255] uint8
        img_uint8 = (image * 255.0).astype(np.uint8)
        
        # 2. 在 uint8 上执行处理
        denoised_uint8 = cv2.fastNlMeansDenoisingColored(img_uint8, None, h, h, 7, 21)
        
        # 3. 转换回 [0, 1] float32
        return denoised_uint8.astype(np.float32) / 255.0
        ## --- 转换结束 ---
    
    def _median_denoise(self, image: np.ndarray, ksize: int = 5) -> np.ndarray:
        """
        中值滤波 -  对椒盐噪声效果好(需转uint8)
        
        注意: cv2.medianBlur 在 float32 上行为不同且通常不用于此目的。
              标准的中值滤波是在 uint8 上执行的。
        """
        if ksize % 2 == 0:
            print(f"警告: 中值滤波核大小 ksize ({ksize}) 必须是奇数，已自动+1。")
            ksize += 1
            
        ## --- 兼容性核心：执行局部 float32 -> uint8 -> float32 转换 ---
        # 1. 转换为 [0, 255] uint8
        img_uint8 = (image * 255.0).astype(np.uint8)
        
        # 2. 在 uint8 上执行处理
        denoised_uint8 = cv2.medianBlur(img_uint8, ksize)
        
        # 3. 转换回 [0, 1] float32
        return denoised_uint8.astype(np.float32) / 255.0
        ## --- 转换结束 ---
    
    def _wavelet_denoise(self, image: np.ndarray, wavelet: str = 'db1', 
                         level: int = 1, threshold_scale: float = 1.0) -> np.ndarray:
        """
        小波降噪 - 多尺度分析(float32 兼容)

        Args:
            wavelet: 小波基类型
            level: 分解层数
            threshold_scale: 阈值缩放因子
        
        注意: pywt 库完美兼容 float32 数据。
        """
        if not _pywt_available:
            print("警告: pywt 库未安装。将回退到高斯降噪。")
            return self._gaussian_denoise(image, sigma=1.0, process_chroma=True)
            
        # image 已经是 float32 [0, 1]，无需归一化
        denoised = np.zeros_like(image)
        
        for i in range(3): # 独立处理 Y, U, V
            channel = image[:, :, i]
            
            # 检查图像尺寸是否足够进行小波分解
            min_size = 2 ** level
            if channel.shape[0] < min_size or channel.shape[1] < min_size:
                print(f"警告: 图像尺寸 ({channel.shape}) 太小，无法进行 {level} 级小波分解。跳过小波降噪。")
                denoised[:, :, i] = channel
                continue
                
            coeffs = pywt.wavedec2(channel, wavelet, level=level)
            
            # 计算阈值
            # (使用最后一个细节层 (cD) 的中位数绝对偏差来估计噪声)
            try:
                # detail_coeffs = coeffs[-1]
                # median_abs_dev = np.median(np.abs(np.concatenate([d for d in detail_coeffs])))
                
                # 更稳健的方法：仅使用最高频的对角细节 cD
                sigma = np.median(np.abs(coeffs[-1][-1])) / 0.6745
                
                # VisuShrink 阈值
                threshold = sigma * threshold_scale * np.sqrt(2 * np.log(channel.size))
            except (IndexError, ValueError):
                # 如果分解失败或返回空，则不进行阈值处理
                threshold = 0.0

            if threshold > 0:
                # 软阈值处理
                coeffs_denoised = [coeffs[0]] # 保留近似系数
                for detail_level in coeffs[1:]:
                    # (cH, cV, cD)
                    coeffs_denoised.append(tuple(pywt.threshold(d, threshold, mode='soft') for d in detail_level))
            else:
                coeffs_denoised = coeffs
            
            # 重建
            try:
                denoised[:, :, i] = pywt.waverec2(coeffs_denoised, wavelet)
            except ValueError:
                print(f"警告: 小波重建失败。跳过通道 {i} 的降噪。")
                denoised[:, :, i] = channel
        
        # 重建后可能超出 [0, 1] 范围
        return np.clip(denoised, 0, 1)
    
    def _anisotropic_denoise(self, image: np.ndarray, iterations: int = 10, 
                             kappa: float = 50, gamma: float = 0.1) -> np.ndarray:
        """
        各向异性扩散降噪 (Perona-Malik) - 强保边能力(float32 兼容)
        
        Args:
            iterations: 迭代次数
            kappa: 扩散系数
            gamma: 步长

        注意: 此算法是纯 NumPy 实现，完全兼容 float32。
              kappa 值 (50) 是为 [0, 255] 范围设计的。
              我们必须将其缩放到 [0, 1] 范围。
        """
        # image 已经是 float32 [0, 1]，无需归一化
        denoised = np.zeros_like(image)
        
        # ## --- 兼容性核心：缩放 kappa ---
        # 将 kappa 从 [0, 255] 范围缩放到 [0, 1] 范围
        kappa_scaled = kappa / 255.0
        
        for i in range(3): # 独立处理 Y, U, V
            channel = image[:, :, i].copy()
            
            for _ in range(iterations):
                # 计算梯度
                grad_n = np.roll(channel, -1, axis=0) - channel
                grad_s = np.roll(channel,  1, axis=0) - channel
                grad_e = np.roll(channel, -1, axis=1) - channel
                grad_w = np.roll(channel,  1, axis=1) - channel
                
                # 计算扩散系数 (Perona-Malik模型2)
                # 使用缩放后的 kappa_scaled
                c_n = np.exp(-(grad_n / kappa_scaled) ** 2)
                c_s = np.exp(-(grad_s / kappa_scaled) ** 2)
                c_e = np.exp(-(grad_e / kappa_scaled) ** 2)
                c_w = np.exp(-(grad_w / kappa_scaled) ** 2)
                
                # 更新
                channel += gamma * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)
            
            denoised[:, :, i] = channel
        
        # 迭代过程可能超出 [0, 1] 范围
        return np.clip(denoised, 0, 1)