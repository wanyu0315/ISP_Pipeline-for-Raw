# raw_denoise.py
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# 尝试导入可选库
try:
    import pywt
    _pywt_available = True
except ImportError:
    _pywt_available = False


class RawDenoise:
    """
    原始域降噪模块 (Raw Domain Denoising)
    接收一个 2D Bayer 数组，返回一个 2D Bayer 数组。
    """

    def __init__(self):
        # 用于存储时域降噪的上一帧数据
        self.prev_raw_float = None
    
    def execute(self, raw_data: np.ndarray, bayer_pattern: str = 'GBRG',
                algorithm: str = 'None', steps: list = None, **kwargs) -> np.ndarray:
        """
        对Bayer Raw数据执行降噪。
        
        Args:
            raw_data: 输入的Bayer Raw数据 (来自 RawLoader 的 2D 数组)
            ... (其他参数不变)
        """
        # 情况 A: 级联模式 (传入了 steps)
        if steps is not None and isinstance(steps, list) and len(steps) > 0:
            # 提取所有步骤的算法名称，组成一个列表字符串
            step_names = [s.get('algorithm', 'unknown') for s in steps]
            print(f"Executing Raw Domain Cascade: {step_names}")
            
        # 情况 B: 单一模式 (递归调用会进入这里，或者单独调用)
        elif algorithm is not None:
            print(f"  > Executing Raw Algo: {algorithm}")

        # --- 级联处理逻辑 ---
        if steps is not None and isinstance(steps, list) and len(steps) > 0:
            # print(f"RawDenoise: Executing cascaded pipeline with {len(steps)} steps...")
            current_data = raw_data
            
            for i, step_params in enumerate(steps):
                # 确保 step_params 是字典
                if not isinstance(step_params, dict):
                    continue
                
                # 继承默认的 bayer_pattern (如果 step 里没写)
                if 'bayer_pattern' not in step_params:
                    step_params['bayer_pattern'] = bayer_pattern
                
                # 递归调用 execute
                # 注意：我们将 step_params 拆包传入，这样它会进入下方的 "单一模式" 逻辑
                # 这样可以复用所有已有的算法分支
                current_data = self.execute(current_data, **step_params)
                
            return current_data

        # --- 常规单一处理逻辑 ---

        if algorithm == 'binning':
            return self._pixel_binning_raw(raw_data, **kwargs)
        elif algorithm == 'temporal':
            return self._temporal_denoise_raw(raw_data, **kwargs)
        # --- 空间降噪 (全部经过 Bayer 分离) ---
        elif algorithm == 'bilateral':
            return self._bilateral_raw(raw_data, bayer_pattern, **kwargs)
        elif algorithm == 'gaussian':
            return self._gaussian_raw(raw_data, bayer_pattern, **kwargs)
        elif algorithm == 'median':
            return self._median_raw(raw_data, bayer_pattern, **kwargs)
        elif algorithm == 'nlm':
            return self._nlm_raw(raw_data, **kwargs)
        elif algorithm == 'green_uniform':
            return self._green_uniform_denoise(raw_data, bayer_pattern, **kwargs)
        elif algorithm == 'adaptive':
            return self._adaptive_raw_denoise(raw_data, bayer_pattern, **kwargs)
        elif algorithm == 'wavelet':
            return self._wavelet_raw_denoise(raw_data, bayer_pattern, **kwargs)
        else:
            raise ValueError(f"Unknown raw denoising algorithm: {algorithm}")
        
    # ==================== 核心辅助函数：通道分离应用 ====================
    def _apply_channel_wise(self, raw_data: np.ndarray, bayer_pattern: str, 
                            func, **kwargs) -> np.ndarray:
        """
        [核心修正] 将降噪函数分别应用于 4 个 Bayer 子通道，然后合并。
        这是 Raw 域空间降噪唯一正确的方式。
        
        Args:
            raw_data: 原始 Bayer 图
            bayer_pattern: 模式 ('RGGB', 'GBRG' 等)
            func: 要应用的降噪函数 (接受 2D array 和 kwargs)
            **kwargs: 传递给 func 的参数
        """
        h, w = raw_data.shape
        output = np.zeros_like(raw_data)
        
        # 定义 4 个相位的切片位置 (Row Start, Col Start)
        # 无论什么 Pattern，我们总是处理 (0,0), (0,1), (1,0), (1,1) 这四个子网格
        # 只是它们代表的颜色不同，但对于"同色降噪"来说，我们只需要独立处理这4块即可。
        # 所以这里不需要解析 'GBRG' 到底哪个是 G，只需要把 4 个相位分开处理就行。
        slices = [
            (slice(0, h, 2), slice(0, w, 2)), # 00
            (slice(0, h, 2), slice(1, w, 2)), # 01
            (slice(1, h, 2), slice(0, w, 2)), # 10
            (slice(1, h, 2), slice(1, w, 2))  # 11
        ]
        
        for (r_slice, c_slice) in slices:
            sub_img = raw_data[r_slice, c_slice]
            
            # 对子图进行处理
            # 注意：传递 kwargs 给具体算法
            denoised_sub = func(sub_img, **kwargs)
            
            output[r_slice, c_slice] = denoised_sub
            
        return output
    
    # ==================== 新增算法：像素合并 (Binning) ====================
    def _pixel_binning_raw(self, raw_data: np.ndarray, mode: str = 'average') -> np.ndarray:
        """
        【rPPG 强力推荐】2x2 像素合并 (Downsampling)
        
        将分辨率长宽各减半，通过合并 4 个同色像素来提升信噪比 (SNR)。
        这比任何软件滤波都更物理、更真实地提升 rPPG 质量。
        
        Args:
            raw_data: 输入 Raw 图
            mode: 'average' (推荐)
        """
        h, w = raw_data.shape
        # 确保长宽是 4 的倍数，否则裁剪
        h_new = (h // 4) * 4
        w_new = (w // 4) * 4
        if h_new != h or w_new != w:
            raw_data = raw_data[:h_new, :w_new]
        
        # 转换为 float 避免溢出
        data_float = raw_data.astype(np.float32)

        # Bayer 格式解析：
        # 一个 2x2 的 Bayer 单元包含 R, Gr, Gb, B。
        # 我们要合并的是 *同色* 像素。
        # 在 4x4 的区域内，有 4 个 R，4 个 B，4 个 Gr，4 个 Gb。
        # 我们将这 4x4 区域 (16 pixels) 变成一个新的 2x2 区域 (4 pixels)。
        
        # 提取四个 Bayer 相位
        p00 = data_float[0::2, 0::2] # Phase 1 (e.g., G or B)
        p01 = data_float[0::2, 1::2] # Phase 2
        p10 = data_float[1::2, 0::2] # Phase 3
        p11 = data_float[1::2, 1::2] # Phase 4

        # 对每个相位内部进行 2x2 平均 (Stride=2)
        # 这样就把 Phase 1 的 2x2 邻域均值化了
        def bin_phase(phase_img):
            return (phase_img[0::2, 0::2] + phase_img[0::2, 1::2] + 
                    phase_img[1::2, 0::2] + phase_img[1::2, 1::2]) / 4.0

        new_p00 = bin_phase(p00)
        new_p01 = bin_phase(p01)
        new_p10 = bin_phase(p10)
        new_p11 = bin_phase(p11)

        # 重组为新的 Bayer 图 (尺寸为原图 1/2)
        new_h, new_w = h // 2, w // 2
        new_raw = np.zeros((new_h, new_w), dtype=np.float32)
        
        new_raw[0::2, 0::2] = new_p00
        new_raw[0::2, 1::2] = new_p01
        new_raw[1::2, 0::2] = new_p10
        new_raw[1::2, 1::2] = new_p11
        
        # 转换回原类型
        if raw_data.dtype == np.uint16:
            return np.clip(new_raw, 0, 65535).astype(np.uint16)
        else:
            return np.clip(new_raw, 0, 255).astype(np.uint8)

    # ==================== 新增算法：时域降噪 (Temporal) ====================
    def _temporal_denoise_raw(self, raw_data: np.ndarray, alpha: float = 0.5, 
                              motion_thresh: float = 0.05) -> np.ndarray:
        """
        【rPPG 核心】Raw 域时域滤波 (IIR)
        
        利用上一帧信息平滑当前帧。
        针对 rPPG 优化：包含简单的运动检测，防止 "拖影" (Ghosting)。
        
        Args:
            raw_data: 当前帧
            alpha: 混合权重 (0-1)。越小，降噪越强(依赖历史)，但拖影风险越大。建议 0.5-0.8。
            motion_thresh: 运动阈值 (0-1, 相对于量程)。超过此变化的像素被视为运动，不进行融合。
                           rPPG 信号变化极小 (<1%)，人脸转动变化大。
        """
        # 归一化处理
        if raw_data.dtype == np.uint16:
            max_val = 65535.0
        else:
            max_val = 255.0
            
        curr_float = raw_data.astype(np.float32) / max_val

        # 第一帧，直接初始化
        if self.prev_raw_float is None:
            self.prev_raw_float = curr_float.copy()
            return raw_data
        
        # 检查尺寸是否变化 (例如开启了 Binning 但上一帧是全尺寸)
        if self.prev_raw_float.shape != curr_float.shape:
             self.prev_raw_float = curr_float.copy()
             return raw_data

        # --- 运动自适应混合 ---
        # 1. 计算差异
        diff = np.abs(curr_float - self.prev_raw_float)
        
        # 2. 生成混合 Mask
        # 如果 diff < threshold, 认为是静态/微动 (rPPG信号)，使用 IIR 融合
        # 如果 diff > threshold, 认为是运动，直接使用当前帧 (alpha = 1.0) 以防拖影
        
        # 创建一个平滑的过渡 mask，而不是硬切断，避免伪影
        # 这里的 mask 表示 "使用当前帧的程度"
        # diff 小 -> mask 接近 alpha
        # diff 大 -> mask 接近 1.0
        
        motion_mask = np.clip(diff / motion_thresh, 0, 1) # 0~1 之间
        # 混合系数：在 alpha 和 1.0 之间根据运动程度插值
        effective_alpha = alpha + (1.0 - alpha) * motion_mask
        
        # 3. 执行 IIR 滤波: Out = alpha * Curr + (1-alpha) * Prev
        out_float = effective_alpha * curr_float + (1.0 - effective_alpha) * self.prev_raw_float
        
        # 4. 更新历史帧 (重要：更新为滤波后的结果，增强平滑延续性)
        self.prev_raw_float = out_float
        
        # 5. 输出
        result = (np.clip(out_float, 0, 1) * max_val).astype(raw_data.dtype)
        return result
    
    def _bilateral_raw(self, raw_data: np.ndarray, bayer_pattern: str, d: int = 5, 
                       sigma_color: float = 50, sigma_space: float = 50) -> np.ndarray:
        """
        RAW 域双边滤波，它在平坦区域进行高斯模糊，但在边缘区域停止模糊，从而防止背景颜色（如墙壁）“渗入”人脸皮肤。
参数含义：
        d：滤波过程中每个像素邻域的直径（整数），决定了计算某个像素的新值时，要参考周围多大范围内的像素。通常在 3 到 9 之间，越大降噪越多。
        sigma_color：
            定义：颜色空间的高斯函数标准差，这是双边滤波的灵魂参数。
            物理意义：决定了多大的颜色差异才会被算作边缘。
            值越小 (e.g., 10)：筛选非常严格。只有颜色极其接近的像素才会互相平滑，稍微有一点差异（如噪点或微弱的脉搏变化）就可能被保留或视为边缘。
            值越大 (e.g., 100)：筛选很宽松。颜色差异较大的像素也会被混合在一起，此时它退化成接近普通的高斯模糊。
        sigma_space (Space Sigma / 空间标准差)
            定义：坐标空间的高斯函数标准差。
            物理意义：决定了物理距离多远的像素可以相互影响。
        """
        def _process_sub(img, d_val=d, s_c=sigma_color, s_s=sigma_space):
            original_dtype = img.dtype
            if original_dtype == np.uint16:
                max_val = 65535.0
            else:
                max_val = 255.0
            
            # OpenCV bilateral 需要 float32 (0-1) 或 uint8
            img_float = img.astype(np.float32) / max_val
            
            # sigma_color 需要根据 0-1 范围调整
            s_c_scaled = s_c / 255.0 
            
            res = cv2.bilateralFilter(img_float, d_val, s_c_scaled, s_s)
            
            return (np.clip(res, 0, 1) * max_val).astype(original_dtype)

        return self._apply_channel_wise(raw_data, bayer_pattern, _process_sub)
    
    def _gaussian_raw(self, raw_data: np.ndarray, bayer_pattern: str, sigma: float = 1.0) -> np.ndarray:
        """
        RAW 域高斯降噪 (Corrected)
        """
        # 定义具体的单通道处理逻辑
        def _process_sub(img, s=sigma):
            data_float = img.astype(np.float32)
            res = gaussian_filter(data_float, sigma=s)
            # 保持数据类型一致
            if img.dtype == np.uint16:
                return np.clip(res, 0, 65535).astype(np.uint16)
            else:
                return np.clip(res, 0, 255).astype(np.uint8)
        
        # 调用通用分离逻辑
        return self._apply_channel_wise(raw_data, bayer_pattern, _process_sub)
    
    def _median_raw(self, raw_data: np.ndarray, bayer_pattern: str, ksize: int = 3) -> np.ndarray:
        """
        RAW 域中值滤波 (Corrected)
        """
        if ksize % 2 == 0: ksize += 1
        
        def _process_sub(img, k=ksize):
            return cv2.medianBlur(img, k)
            
        return self._apply_channel_wise(raw_data, bayer_pattern, _process_sub)
    
    def _nlm_raw(self, raw_data: np.ndarray, h: float = 10,
                   template_window_size: int = 7, search_window_size: int = 21) -> np.ndarray:
        """
        非局部均值降噪 - (需转uint8)
        
        警告: cv2.fastNlMeansDenoising *只* 支持 uint8。
             这将导致 16-bit 数据的精度损失。
        """
        original_dtype = raw_data.dtype
        
        if original_dtype == np.uint16:
            print("警告: NLM 降噪需要转为 uint8，将导致 16-bit 精度损失。")
            raw_uint8 = (raw_data.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
            
            denoised_uint8 = cv2.fastNlMeansDenoising(
                raw_uint8, None, h, template_window_size, search_window_size
            )
            
            # 转换回 16-bit 范围
            denoised_float = denoised_uint8.astype(np.float32) / 255.0
            return (denoised_float * 65535.0).astype(np.uint16)
        
        elif original_dtype == np.uint8:
            return cv2.fastNlMeansDenoising(
                raw_data, None, h, template_window_size, search_window_size
            )
        else:
            raise TypeError(f"NLM 不支持的数据类型: {original_dtype}")
    

    def _green_uniform_denoise(self, raw_data: np.ndarray, bayer_pattern: str,
                               balance_strength: float = 0.5) -> np.ndarray:
        """
        绿色通道均匀化降噪 - 处理Gr/Gb差异
        Bayer模式中有两个绿色通道，它们应该相似但可能有噪声差异
        
        Args:
            bayer_pattern: Bayer模式
            balance_strength: 均衡强度 (0-1)

            green_uniform算法原理： 强制让 Gr 和 Gb 的亮度趋于一致。
            它假设在一个很小的局部范围内（比如 2x2 邻域），光照强度是不变的，因此 Gr 和 Gb 应该相等。
            如果它们不相等，那就取它们的平均值作为基准，把两者都往平均值上拉。
            参数 balance_strength 控制了修正的力度，范围通常是 0.0 到 1.0 。
            1. strength = 0.0 (无操作)效果： G_r' = G_r，数据保持原样。
            后果： 保留原始的传感器数据，但如果传感器本身有 Gr/Gb 差异，去马赛克后会有网格噪点。
            2. strength = 1.0 (完全均衡 / 强制统一)效果： G_r' = G_{avg}$，$G_b' = G_{avg}$。
            含义： Gr 和 Gb 被强制变成完全一样的值。
            优点： 彻底消除了 Gr/Gb 差异带来的网格噪声，画面极其干净。
            缺点（副作用）：损失高频细节，会把这条真实的纹理抹平，导致图像锐度下降。
        """
        denoised = raw_data.copy().astype(np.float32)
        h, w = raw_data.shape
        
        # 提取两个绿色通道
        if bayer_pattern == 'RGGB':
            gr = raw_data[0::2, 1::2].astype(np.float32)  # R行的G
            gb = raw_data[1::2, 0::2].astype(np.float32)  # B行的G
        elif bayer_pattern == 'BGGR':
            gb = raw_data[0::2, 1::2].astype(np.float32)
            gr = raw_data[1::2, 0::2].astype(np.float32)
        elif bayer_pattern == 'GRBG':
            gr = raw_data[0::2, 0::2].astype(np.float32)
            gb = raw_data[1::2, 1::2].astype(np.float32)
        elif bayer_pattern == 'GBRG':
            gb = raw_data[0::2, 0::2].astype(np.float32)
            gr = raw_data[1::2, 1::2].astype(np.float32)
        else:
            return raw_data
        
        # 计算两个绿色通道的平均值
        min_h = min(gr.shape[0], gb.shape[0])
        min_w = min(gr.shape[1], gb.shape[1])
        
        gr_crop = gr[:min_h, :min_w]
        gb_crop = gb[:min_h, :min_w]
        
        g_mean = (gr_crop + gb_crop) / 2.0
        
        # 均衡化
        gr_balanced = gr_crop * (1 - balance_strength) + g_mean * balance_strength
        gb_balanced = gb_crop * (1 - balance_strength) + g_mean * balance_strength
        
        # 写回
        if bayer_pattern == 'RGGB':
            denoised[0::2, 1::2][:min_h, :min_w] = gr_balanced
            denoised[1::2, 0::2][:min_h, :min_w] = gb_balanced
        elif bayer_pattern == 'BGGR':
            denoised[0::2, 1::2][:min_h, :min_w] = gb_balanced
            denoised[1::2, 0::2][:min_h, :min_w] = gr_balanced
        elif bayer_pattern == 'GRBG':
            denoised[0::2, 0::2][:min_h, :min_w] = gr_balanced
            denoised[1::2, 1::2][:min_h, :min_w] = gb_balanced
        elif bayer_pattern == 'GBRG':
            denoised[0::2, 0::2][:min_h, :min_w] = gb_balanced
            denoised[1::2, 1::2][:min_h, :min_w] = gr_balanced
        
        return denoised.astype(raw_data.dtype)
    
    def _wavelet_raw_denoise(self, raw_data: np.ndarray, bayer_pattern: str, wavelet: str = 'db1',
                             level: int = 2, threshold_scale: float = 1.0) -> np.ndarray:
        """
        【修正】现在通过 _apply_channel_wise 正确处理 Bayer 图
        """
        if not _pywt_available:
            print("Warning: pywt not installed, fallback to bilateral")
            return self._bilateral_raw(raw_data, bayer_pattern)

        def _process_sub(img, w=wavelet, l=level, ts=threshold_scale):
            if img.dtype == np.uint16: max_val = 65535.0
            else: max_val = 255.0
            data_float = img.astype(np.float32) / max_val
            
            coeffs = pywt.wavedec2(data_float, w, level=l)
            sigma = np.median(np.abs(coeffs[-1][0])) / 0.6745
            threshold = sigma * ts * np.sqrt(2 * np.log(data_float.size))
            
            coeffs_denoised = [coeffs[0]]
            for detail in coeffs[1:]:
                coeffs_denoised.append(tuple(pywt.threshold(d, threshold, mode='soft') for d in detail))
            
            denoised = pywt.waverec2(coeffs_denoised, w)
            # 尺寸对齐
            denoised = denoised[:img.shape[0], :img.shape[1]]
            return (np.clip(denoised, 0, 1) * max_val).astype(img.dtype)

        return self._apply_channel_wise(raw_data, bayer_pattern, _process_sub)

    def _adaptive_raw_denoise(self, raw_data: np.ndarray, bayer_pattern: str,
                              base_strength: float = 1.0, edge_threshold: float = 50) -> np.ndarray:
        """
        【修正】现在通过 _apply_channel_wise 正确处理 Bayer 图
        这样梯度计算才是基于同色像素的，能够区分真正的边缘和Bayer图案。
        """
        def _process_sub(img, bs=base_strength, et=edge_threshold):
            if img.dtype == np.uint16: max_val = 65535.0
            else: max_val = 255.0
            data_float = img.astype(np.float32) / max_val
            
            grad_x = np.abs(np.diff(data_float, axis=1, prepend=data_float[:, :1]))
            grad_y = np.abs(np.diff(data_float, axis=0, prepend=data_float[:1, :]))
            edge_strength = np.sqrt(grad_x**2 + grad_y**2)
            edge_strength = edge_strength / (edge_strength.max() + 1e-6)
            
            denoise_weight = 1.0 - np.clip(edge_strength * et, 0, 1)
            denoised = gaussian_filter(data_float, sigma=bs)
            result = data_float * (1 - denoise_weight) + denoised * denoise_weight
            
            return (np.clip(result, 0, 1) * max_val).astype(img.dtype)
            
        return self._apply_channel_wise(raw_data, bayer_pattern, _process_sub)
    
    # # ==================== 辅助函数 ====================
    
    # def _extract_bayer_channels(self, raw_data: np.ndarray, 
    #                            bayer_pattern: str) -> dict:
    #     """
    #     从Bayer Raw数据中提取R/Gr/Gb/B四个通道
        
    #     Returns:
    #         字典: {'R': (data, mask), 'Gr': (data, mask), 'Gb': (data, mask), 'B': (data, mask)}
    #     """
    #     h, w = raw_data.shape
        
    #     # 根据Bayer模式定义位置
    #     patterns = {
    #         'RGGB': {'R': (0, 0), 'Gr': (0, 1), 'Gb': (1, 0), 'B': (1, 1)},
    #         'BGGR': {'B': (0, 0), 'Gb': (0, 1), 'Gr': (1, 0), 'R': (1, 1)},
    #         'GRBG': {'Gr': (0, 0), 'R': (0, 1), 'B': (1, 0), 'Gb': (1, 1)},
    #         'GBRG': {'Gb': (0, 0), 'B': (0, 1), 'R': (1, 0), 'Gr': (1, 1)}
    #     }
        
    #     if bayer_pattern not in patterns:
    #         raise ValueError(f"Unknown Bayer pattern: {bayer_pattern}")
        
    #     pattern = patterns[bayer_pattern]
    #     channels = {}
        
    #     for color, (row_offset, col_offset) in pattern.items():
    #         # 提取子通道
    #         channel_data = raw_data[row_offset::2, col_offset::2]
            
    #         # 创建mask
    #         mask = np.zeros((h, w), dtype=bool)
    #         mask[row_offset::2, col_offset::2] = True
            
    #         channels[color] = (channel_data, mask)
        
    #     return channels
    
    # def _reconstruct_bayer(self, channels: dict, bayer_pattern: str, 
    #                       h: int, w: int) -> np.ndarray:
    #     """
    #     从分离的通道重构Bayer Raw数据
        
    #     Args:
    #         channels: 字典包含处理后的通道数据
    #         bayer_pattern: Bayer模式
    #         h, w: 输出图像尺寸
        
    #     Returns:
    #         重构的Bayer Raw数据
    #     """
    #     reconstructed = np.zeros((h, w), dtype=np.float32)
        
    #     patterns = {
    #         'RGGB': {'R': (0, 0), 'Gr': (0, 1), 'Gb': (1, 0), 'B': (1, 1)},
    #         'BGGR': {'B': (0, 0), 'Gb': (0, 1), 'Gr': (1, 0), 'R': (1, 1)},
    #         'GRBG': {'Gr': (0, 0), 'R': (0, 1), 'B': (1, 0), 'Gb': (1, 1)},
    #         'GBRG': {'Gb': (0, 0), 'B': (0, 1), 'R': (1, 0), 'Gr': (1, 1)}
    #     }
        
    #     pattern = patterns[bayer_pattern]
        
    #     for color, (row_offset, col_offset) in pattern.items():
    #         channel_data = channels[color]
    #         ch_h = (h - row_offset + 1) // 2
    #         ch_w = (w - col_offset + 1) // 2
    #         reconstructed[row_offset::2, col_offset::2] = channel_data[:ch_h, :ch_w]
        
    #     return reconstructed