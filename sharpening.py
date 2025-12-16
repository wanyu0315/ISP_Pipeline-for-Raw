# sharpen.py
import numpy as np
import cv2
from scipy.ndimage import convolve, gaussian_filter, uniform_filter


class Sharpen:
    """
    锐化模块：提供多种锐化算法。
    
    所有方法均假设输入为 np.float32 [0, 1] YUV图像，
    输出也为 np.float32 [0, 1] YUV图像。
    锐化操作将只在 Y (亮度) 通道上执行。
    """
    
    def execute(self, yuv_image: np.ndarray, algorithm: str = 'unsharp_mask', **kwargs) -> np.ndarray:
        """
        执行图像锐化。通常只对Y通道（亮度）进行锐化。
        
        Args:
            yuv_image: 输入的YUV图像 (np.float32, 范围 [0, 1])
            algorithm: 锐化算法选择
                'laplacian': 拉普拉斯锐化
                'unsharp_mask': 非锐化掩蔽 (USM)
                'high_pass': 高通滤波锐化
                'sobel': Sobel边缘增强
                'gaussian_unsharp': 高斯USM (更平滑)
                'adaptive': 自适应锐化
                'wiener': 维纳滤波锐化
            **kwargs: 算法特定参数
        
        Returns:
            锐化后的YUV图像 (np.float32, 范围 [0, 1])
        """
        print(f"Executing Sharpening using algorithm: {algorithm}")
        
        # 确保输入是 float32
        if yuv_image.dtype != np.float32:
            print(f"警告: Sharpen 模块接收到非 float32 图像 (dtype: {yuv_image.dtype})。")
            if yuv_image.max() > 1.0:
                 yuv_image = yuv_image.astype(np.float32) / 65535.0
            else:
                 yuv_image = yuv_image.astype(np.float32)

        # 只对Y通道进行锐化，保持UV通道不变
        y_channel, u_channel, v_channel = cv2.split(yuv_image)
        
        if algorithm == 'laplacian':
            y_sharpened = self._laplacian_sharpen(y_channel, **kwargs)
        elif algorithm == 'unsharp_mask':
            y_sharpened = self._unsharp_mask(y_channel, **kwargs)
        elif algorithm == 'high_pass':
            y_sharpened = self._high_pass_sharpen(y_channel, **kwargs)
        elif algorithm == 'sobel':
            y_sharpened = self._sobel_sharpen(y_channel, **kwargs)
        elif algorithm == 'gaussian_unsharp':
            y_sharpened = self._gaussian_unsharp(y_channel, **kwargs)
        elif algorithm == 'adaptive':
            y_sharpened = self._adaptive_sharpen(y_channel, **kwargs)
        elif algorithm == 'wiener':
            y_sharpened = self._wiener_sharpen(y_channel, **kwargs)
        else:
            raise ValueError(f"Unknown sharpening algorithm: {algorithm}")
        
        # 合并通道
        return cv2.merge([y_sharpened, u_channel, v_channel])
    
    def _laplacian_sharpen(self, channel: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        拉普拉斯锐化 - (float32 兼容)
        
        Args:
            channel: Y 通道 (float32, [0, 1])
            strength: 锐化强度 (0-2)
        """
        # 拉普拉斯核
        kernel = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        # channel 已经是 float32 [0, 1]
        
        # 计算拉普拉斯
        laplacian = convolve(channel, kernel, mode='reflect')
        
        # 添加回原图像
        sharpened = channel + strength * laplacian
        return np.clip(sharpened, 0, 1)
    
    def _unsharp_mask(self, channel: np.ndarray, radius: float = 1.0, 
                        amount: float = 1.0, threshold: float = 0) -> np.ndarray:
        """
        非锐化掩蔽 (USM) - (float32 兼容)
        
        Args:
            channel: Y 通道 (float32, [0, 1])
            radius: 高斯模糊半径
            amount: 锐化强度
            threshold: 阈值, 避免锐化噪声 (范围应为 [0, 1], e.g., 5/255.0)
        """
        # channel 已经是 float32 [0, 1]
        
        # 高斯模糊
        # cv2.GaussianBlur 完美支持 float32 [0, 1]
        blurred = cv2.GaussianBlur(channel, (0, 0), radius)
        
        # 计算锐化掩模
        mask = channel - blurred
        
        # 应用阈值
        if threshold > 0:
            mask = np.where(np.abs(mask) >= threshold, mask, 0)
        
        # 应用锐化
        sharpened = channel + amount * mask
        return np.clip(sharpened, 0, 1)
    
    def _high_pass_sharpen(self, channel: np.ndarray, radius: int = 5, 
                           strength: float = 1.5) -> np.ndarray:
        """
        高通滤波锐化 - (float32 兼容)
        
        Args:
            channel: Y 通道 (float32, [0, 1])
            radius: 滤波半径
            strength: 锐化强度
        """
        # channel 已经是 float32 [0, 1]
        
        # 低通滤波（模糊）
        # cv2.boxFilter 支持 float32 [0, 1] (ddepth=-1)
        low_pass = cv2.boxFilter(channel, -1, (radius, radius))
        
        # 高通 = 原图 - 低通
        high_pass = channel - low_pass
        
        # 增强高频
        sharpened = channel + strength * high_pass
        return np.clip(sharpened, 0, 1)
    
    def _sobel_sharpen(self, channel: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Sobel边缘增强 - (float32 兼容)
        
        Args:
            channel: Y 通道 (float32, [0, 1])
            strength: 锐化强度
        """
        # channel 已经是 float32 [0, 1]
        
        # Sobel算子
        # cv2.Sobel 支持 float32 [0, 1] 输入,
        # 并通过 cv2.CV_32F 指定 float32 输出
        sobel_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        
        # 梯度幅值
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 添加边缘信息
        sharpened = channel + strength * gradient
        return np.clip(sharpened, 0, 1)
    
    def _gaussian_unsharp(self, channel: np.ndarray, sigma: float = 1.0, 
                          amount: float = 1.5) -> np.ndarray:
        """
        高斯非锐化掩蔽 - (float32 兼容)
        
        Args:
            channel: Y 通道 (float32, [0, 1])
            sigma: 高斯标准差
            amount: 锐化强度
        """
        # channel 已经是 float32 [0, 1]
        
        # 高斯模糊
        # scipy.ndimage.gaussian_filter 完美支持 float32 [0, 1]
        blurred = gaussian_filter(channel, sigma=sigma)
        
        # 锐化
        sharpened = channel + amount * (channel - blurred)
        return np.clip(sharpened, 0, 1)
    
    def _adaptive_sharpen(self, channel: np.ndarray, strength: float = 1.0, 
                          window_size: int = 5) -> np.ndarray:
        """
        自适应锐化 - (float32 兼容)
        
        Args:
            channel: Y 通道 (float32, [0, 1])
            strength: 基础锐化强度
            window_size: 局部窗口大小
        """
        # channel 已经是 float32 [0, 1]
        
        # 计算局部标准差（对比度）
        # scipy.ndimage.uniform_filter 完美支持 float32 [0, 1]
        mean = uniform_filter(channel, size=window_size)
        mean_sq = uniform_filter(channel**2, size=window_size)
        variance = mean_sq - mean**2
        std_dev = np.sqrt(np.maximum(variance, 0))
        
        # 归一化标准差作为自适应权重
        std_max = std_dev.max()
        if std_max > 0:
            adaptive_weight = std_dev / (std_max + 1e-6)
        else:
            adaptive_weight = np.zeros_like(std_dev)
        
        # 拉普拉斯锐化
        kernel = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        laplacian = convolve(channel, kernel, mode='reflect')
        
        # 自适应应用锐化
        sharpened = channel + strength * adaptive_weight * laplacian
        return np.clip(sharpened, 0, 1)
    
    def _wiener_sharpen(self, channel: np.ndarray, noise_variance: float = 0.001, 
                        strength: float = 1.0) -> np.ndarray:
        """
        维纳滤波锐化 - (float32 兼容)
        
        Args:
            channel: Y 通道 (float32, [0, 1])
            noise_variance: 噪声方差估计 (应在 [0, 1] 范围内)
            strength: 锐化强度
        """
        # channel 已经是 float32 [0, 1]
        
        # FFT变换
        f_transform = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f_transform)
        
        # 创建理想的锐化滤波器（高通）
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        
        x = np.arange(cols) - ccol
        y = np.arange(rows) - crow
        X, Y = np.meshgrid(x, y)
        D = np.sqrt(X**2 + Y**2)
        D_max = D.max()
        
        # 维纳滤波器
        # H 是高频增强滤波器
        if D_max > 0:
            H = 1 + strength * (D / (D_max + 1e-6))
        else:
            H = np.ones_like(D)
            
        # 信号功率谱的粗略估计 |S(f)|^2
        signal_power = np.abs(f_shift)**2
        
        # 维纳滤波器: H_wiener = H * [ |S(f)|^2 / (|S(f)|^2 + |N(f)|^2) ]
        # |N(f)|^2 是噪声功率谱，我们用估计的 noise_variance 
        # H * [ 1 / (1 + |N(f)|^2 / |S(f)|^2) ]
        # 注意: 原代码中的 wiener_filter 形式是 H / (1 + N/S)，这是对的
        wiener_filter = H / (1 + noise_variance / (signal_power + 1e-6))
        
        # 应用滤波器
        f_filtered = f_shift * wiener_filter
        
        # 逆FFT
        f_ishift = np.fft.ifftshift(f_filtered)
        sharpened = np.fft.ifft2(f_ishift)
        sharpened = np.abs(sharpened) # 取实部
        
        return np.clip(sharpened, 0, 1)