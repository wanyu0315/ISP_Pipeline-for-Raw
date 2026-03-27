# yuv_to_rgb.py
import numpy as np
import cv2

class YUVtoRGB:
    """
    YUV到RGB颜色空间转换模块
    将YUV格式转换回RGB格式，用于最终输出
    
    接收 np.float32 YUV图像 (支持 HDR 范围)，
    输出 np.float32 RGB图像 (支持 HDR 范围)。
    在最后的 `convert_for_display` 中进行统一的量化与截断。
    """
    
    def execute(self, yuv_image: np.ndarray, method: str = 'bt709') -> np.ndarray:
        """
        执行YUV到RGB的颜色空间转换。
        """
        print(f"Executing Color Space Conversion (YUV->RGB) using method: {method}")
        
        # ===================================================================
        # 🌟 [修改] 零损耗数据流：只校验类型，不因 max>1.0 而除以 65535
        # ===================================================================
        if not np.issubdtype(yuv_image.dtype, np.floating):
            print(f"  [警告] YUVtoRGB 期望接收 float32，当前为 {yuv_image.dtype}。正在转换...")
            yuv_image = yuv_image.astype(np.float32)
        
        if method == 'bt601':
            rgb_image = self._yuv_to_rgb_bt601(yuv_image)
        elif method == 'bt709':
            rgb_image = self._yuv_to_rgb_bt709(yuv_image)
        elif method == 'bt2020':
            rgb_image = self._yuv_to_rgb_bt2020(yuv_image)
        elif method == 'opencv_bt601':
            rgb_image = self._yuv_to_rgb_opencv(yuv_image)
        else:
            raise ValueError(f"Unknown YUV to RGB conversion method: {method}")
        
        # 返回 float32 的 RGB 图像 (保留 HDR 极值)
        return rgb_image
    
    def _yuv_to_rgb_bt601(self, yuv_float: np.ndarray) -> np.ndarray:
        """
        BT.601标准转换 (SDTV) - (float32 兼容)
        """
        # UV分量去偏移 (从 [0, 1] 恢复到 [-0.5, 0.5])
        yuv_centered = yuv_float.copy()
        yuv_centered[:, :, 1] -= 0.5  # U
        yuv_centered[:, :, 2] -= 0.5  # V
        
        # YUV到RGB的转换矩阵 (BT.601)
        transform_matrix = np.array([
            [1.0,  0.0,       1.402],
            [1.0, -0.344136, -0.714136],
            [1.0,  1.772,     0.0]
        ], dtype=np.float32)
        
        # 执行矩阵乘法
        rgb = np.dot(yuv_centered, transform_matrix.T)
        
        # [修改] 移除 np.clip，保留浮点极值
        return rgb
    
    def _yuv_to_rgb_bt709(self, yuv_float: np.ndarray) -> np.ndarray:
        """
        BT.709标准转换 (HDTV) - (float32 兼容)
        """
        yuv_centered = yuv_float.copy()
        yuv_centered[:, :, 1] -= 0.5
        yuv_centered[:, :, 2] -= 0.5
        
        # BT.709 YUV到RGB转换矩阵
        transform_matrix = np.array([
            [1.0,  0.0,       1.5748],
            [1.0, -0.187324, -0.468124],
            [1.0,  1.8556,    0.0]
        ], dtype=np.float32)
        
        rgb = np.dot(yuv_centered, transform_matrix.T)
        # [修改] 移除 np.clip
        return rgb
    
    def _yuv_to_rgb_bt2020(self, yuv_float: np.ndarray) -> np.ndarray:
        """
        BT.2020标准转换 (UHDTV/4K/8K) - (float32 兼容)
        """
        yuv_centered = yuv_float.copy()
        yuv_centered[:, :, 1] -= 0.5
        yuv_centered[:, :, 2] -= 0.5
        
        # BT.2020 YUV到RGB转换矩阵
        transform_matrix = np.array([
            [1.0,  0.0,       1.4746],
            [1.0, -0.164553, -0.571353],
            [1.0,  1.8814,    0.0]
        ], dtype=np.float32)
        
        rgb = np.dot(yuv_centered, transform_matrix.T)
        # [修改] 移除 np.clip
        return rgb
    
    def _yuv_to_rgb_opencv(self, yuv_float: np.ndarray) -> np.ndarray:
        """
        使用OpenCV的YUV到RGB转换 (BT.601)
        """
        # 注意: cv2.cvtColor 在处理浮点越界数值时可能会有内置的安全机制
        return cv2.cvtColor(yuv_float, cv2.COLOR_YUV2RGB)
    
    def convert_for_display(self, yuv_image: np.ndarray, 
                            method: str = 'bt709',
                            output_format: str = 'bgr',
                            bit_depth: int = 8) -> np.ndarray:
        """
        【重构版】终极出口大清算：转换YUV图像用于显示或保存
        
        此函数在 ISP 管道 *最后一步* 调用。它负责将一路狂奔的 HDR float32 图像，
        安全、精准地压回物理世界认识的 uint8/uint16 格式。
        """
        # 1. 获取包含 HDR 极值的 float32 RGB 图像
        rgb_float = self.execute(yuv_image, method=method)
        
        # ===================================================================
        # 🌟 零损耗数据流的终结：全管线唯一的一次全局截断！
        # 将小于 0 的负下冲砍回 0；将大于 1.0 的高光砍回 1.0。
        # ===================================================================
        rgb_float_clipped = np.clip(rgb_float, 0.0, 1.0)
        
        # 2. 极其安全的位深度转换：乘法放大 -> 四舍五入(极其关键!) -> astype
        if bit_depth == 8:
            output_image = np.clip(np.round(rgb_float_clipped * 255.0), 0, 255).astype(np.uint8)
        elif bit_depth == 16:
            output_image = np.clip(np.round(rgb_float_clipped * 65535.0), 0, 65535).astype(np.uint16)
        else:
            raise ValueError("输出位深度 (bit_depth) 必须是 8 或 16")
        
        # 3. 转换通道顺序 (为 cv2.imwrite 准备)
        if output_format == 'bgr':
            # 注意: 这里输入的是 uint 图像，cv2.cvtColor 完全没问题
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        elif output_format != 'rgb':
            raise ValueError("输出格式 (output_format) 必须是 'rgb' 或 'bgr'")
        
        return output_image
    
    def batch_convert(self, yuv_images: list, method: str = 'bt709') -> list:
        """
        批量转换YUV图像到RGB
        """
        rgb_images = []
        for i, yuv_img in enumerate(yuv_images):
            print(f"Converting image {i+1}/{len(yuv_images)}")
            rgb_img = self.execute(yuv_img, method=method)
            rgb_images.append(rgb_img)
        return rgb_images
    
    def get_conversion_matrix(self, method: str = 'bt709') -> np.ndarray:
        """获取YUV到RGB的转换矩阵"""
        matrices = {
            'bt601': np.array([
                [1.0,  0.0,       1.402],
                [1.0, -0.344136, -0.714136],
                [1.0,  1.772,     0.0]
            ], dtype=np.float32),
            'bt709': np.array([
                [1.0,  0.0,       1.5748],
                [1.0, -0.187324, -0.468124],
                [1.0,  1.8556,    0.0]
            ], dtype=np.float32),
            'bt2020': np.array([
                [1.0,  0.0,       1.4746],
                [1.0, -0.164553, -0.571353],
                [1.0,  1.8814,    0.0]
            ], dtype=np.float32)
        }
        if method not in matrices:
            raise ValueError(f"Unknown method: {method}")
        return matrices[method]
    
    def verify_conversion(self, original_rgb: np.ndarray, 
                          yuv_image: np.ndarray,
                          method: str = 'bt709',
                          tolerance: float = 0.01) -> dict:
        """验证YUV到RGB转换的准确性"""
        
        # --- 1. 归一化输入 ---
        if original_rgb.dtype == np.uint16:
            orig_float = original_rgb.astype(np.float32) / 65535.0
        elif original_rgb.dtype == np.uint8:
            orig_float = original_rgb.astype(np.float32) / 255.0
        else:
            orig_float = original_rgb 

        if yuv_image.dtype == np.uint16:
            yuv_float = yuv_image.astype(np.float32) / 65535.0
        elif yuv_image.dtype == np.uint8:
            yuv_float = yuv_image.astype(np.float32) / 255.0
        else:
            yuv_float = yuv_image 
            
        # --- 2. 执行转换 (注意：这里的执行是没有clip的) ---
        recon_float = self.execute(yuv_float, method=method)
        # 为了比对真实误差，手动裁掉超出 [0,1] 的 HDR 数值
        recon_float = np.clip(recon_float, 0, 1)
        
        # --- 3. 计算误差 ---
        diff = np.abs(orig_float - recon_float)
        max_error = np.max(diff)
        mean_error = np.mean(diff)
        mse = np.mean(diff ** 2)
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        
        within_tolerance = max_error <= tolerance
        
        results = {
            'max_error': max_error, 'mean_error': mean_error, 'mse': mse,
            'psnr': psnr, 'within_tolerance': within_tolerance, 'tolerance': tolerance
        }
        
        print("\n=== YUV to RGB Conversion Verification ===")
        print(f"Method: {method}")
        print(f"Max Error:  {max_error:.6f}")
        print(f"Mean Error: {mean_error:.6f}")
        print(f"PSNR:       {psnr:.2f} dB")
        print(f"Within Tolerance ({tolerance}): {'✓ Yes' if within_tolerance else '✗ No'}")
        
        return results

# # yuv_to_rgb.py
# import numpy as np
# import cv2


# class YUVtoRGB:
#     """
#     YUV到RGB颜色空间转换模块
#     将YUV格式转换回RGB格式，用于最终输出
    
#     所有方法均假设输入为 np.float32 [0, 1] YUV图像，
#     输出也为 np.float32 [0, 1] RGB图像。
#     """
    
#     def execute(self, yuv_image: np.ndarray, method: str = 'bt709') -> np.ndarray:
#         """
#         执行YUV到RGB的颜色空间转换。
        
#         Args:
#             yuv_image: 输入的YUV图像 (np.float32, 范围 [0, 1])
#             method: 转换方法，应与之前RGB->YUV使用的方法一致
#                 'bt601': BT.601标准 (SDTV)
#                 'bt709': BT.709标准 (HDTV) - 推荐
#                 'bt2020': BT.2020标准 (UHDTV)
#                 'opencv_bt601': 使用OpenCV的转换 (BT.601)
        
#         Returns:
#             RGB格式的图像 (np.float32, 范围 [0, 1])
#         """
#         print(f"Executing Color Space Conversion (YUV->RGB) using method: {method}")
        
#         # 确保输入是 float32
#         if yuv_image.dtype != np.float32:
#             print(f"警告: YUVtoRGB 模块接收到非 float32 图像 (dtype: {yuv_image.dtype})。")
#             if yuv_image.max() > 1.0:
#                  yuv_image = yuv_image.astype(np.float32) / 65535.0
#             else:
#                  yuv_image = yuv_image.astype(np.float32)
        
#         if method == 'bt601':
#             rgb_image = self._yuv_to_rgb_bt601(yuv_image)
#         elif method == 'bt709':
#             rgb_image = self._yuv_to_rgb_bt709(yuv_image)
#         elif method == 'bt2020':
#             rgb_image = self._yuv_to_rgb_bt2020(yuv_image)
#         elif method == 'opencv_bt601':
#             rgb_image = self._yuv_to_rgb_opencv(yuv_image)
#         else:
#             raise ValueError(f"Unknown YUV to RGB conversion method: {method}")
        
#         # 返回 float32 [0, 1] 的 RGB 图像
#         return rgb_image
    
#     def _yuv_to_rgb_bt601(self, yuv_float: np.ndarray) -> np.ndarray:
#         """
#         BT.601标准转换 (SDTV) - (float32 兼容)
#         """
#         # yuv_float 已经是 [0, 1]
        
#         # UV分量去偏移 (从 [0, 1] 恢复到 [-0.5, 0.5])
#         yuv_centered = yuv_float.copy()
#         yuv_centered[:, :, 1] -= 0.5  # U
#         yuv_centered[:, :, 2] -= 0.5  # V
        
#         # YUV到RGB的转换矩阵 (BT.601)
#         transform_matrix = np.array([
#             [1.0,  0.0,       1.402],
#             [1.0, -0.344136, -0.714136],
#             [1.0,  1.772,     0.0]
#         ], dtype=np.float32)
        
#         # 执行矩阵乘法
#         rgb = np.dot(yuv_centered, transform_matrix.T)
        
#         # 裁剪到有效范围
#         return np.clip(rgb, 0, 1)
    
#     def _yuv_to_rgb_bt709(self, yuv_float: np.ndarray) -> np.ndarray:
#         """
#         BT.709标准转换 (HDTV) - (float32 兼容)
#         """
#         yuv_centered = yuv_float.copy()
#         yuv_centered[:, :, 1] -= 0.5
#         yuv_centered[:, :, 2] -= 0.5
        
#         # BT.709 YUV到RGB转换矩阵
#         transform_matrix = np.array([
#             [1.0,  0.0,       1.5748],
#             [1.0, -0.187324, -0.468124],
#             [1.0,  1.8556,    0.0]
#         ], dtype=np.float32)
        
#         rgb = np.dot(yuv_centered, transform_matrix.T)
#         return np.clip(rgb, 0, 1)
    
#     def _yuv_to_rgb_bt2020(self, yuv_float: np.ndarray) -> np.ndarray:
#         """
#         BT.2020标准转换 (UHDTV/4K/8K) - (float32 兼容)
#         """
#         yuv_centered = yuv_float.copy()
#         yuv_centered[:, :, 1] -= 0.5
#         yuv_centered[:, :, 2] -= 0.5
        
#         # BT.2020 YUV到RGB转换矩阵
#         transform_matrix = np.array([
#             [1.0,  0.0,       1.4746],
#             [1.0, -0.164553, -0.571353],
#             [1.0,  1.8814,    0.0]
#         ], dtype=np.float32)
        
#         rgb = np.dot(yuv_centered, transform_matrix.T)
#         return np.clip(rgb, 0, 1)
    
#     def _yuv_to_rgb_opencv(self, yuv_float: np.ndarray) -> np.ndarray:
#         """
#         【修正版】使用OpenCV的YUV到RGB转换 (BT.601)
        
#         cv2.cvtColor 完美支持 float32 [0, 1] 输入
#         """
#         # 移除了所有 uint8 转换，避免精度损失
#         return cv2.cvtColor(yuv_float, cv2.COLOR_YUV2RGB)
    
#     def convert_for_display(self, yuv_image: np.ndarray, 
#                             method: str = 'bt709',
#                             output_format: str = 'bgr',
#                             bit_depth: int = 8) -> np.ndarray:
#         """
#         【重构版】转换YUV图像用于显示或保存
        
#         此函数在 ISP 管道 *之后* 调用，将 float32 [0, 1] 图像
#         转换为最终的 uint8/uint16 BGR/RGB 格式。
        
#         Args:
#             yuv_image: 输入YUV图像 (np.float32, 范围 [0, 1])
#             method: YUV到RGB转换方法
#             output_format: 'rgb' 或 'bgr' (OpenCV 默认)
#             bit_depth: 8 或 16
        
#         Returns:
#             转换后的图像 (uint8 或 uint16)
#         """
#         # 1. 执行ISP管道的最后一步：获取 float32 [0, 1] RGB 图像
#         #    （假设 yuv_image 已经是 float32 [0, 1]）
#         rgb_float = self.execute(yuv_image, method=method)
        
#         # 2. 转换位深度
#         if bit_depth == 8:
#             output_image = (rgb_float * 255.0).astype(np.uint8)
#         elif bit_depth == 16:
#             output_image = (rgb_float * 65535.0).astype(np.uint16)
#         else:
#             raise ValueError("输出位深度 (bit_depth) 必须是 8 或 16")
        
#         # 3. 转换通道顺序 (为 cv2.imwrite 准备)
#         if output_format == 'bgr':
#             output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
#         elif output_format != 'rgb':
#             raise ValueError("输出格式 (output_format) 必须是 'rgb' 或 'bgr'")
        
#         return output_image
    
#     def batch_convert(self, yuv_images: list, method: str = 'bt709') -> list:
#         """
#         批量转换YUV图像到RGB
#         (假设 yuv_images 列表中的都是 float32 [0, 1])
#         """
#         rgb_images = []
#         for i, yuv_img in enumerate(yuv_images):
#             print(f"Converting image {i+1}/{len(yuv_images)}")
#             rgb_img = self.execute(yuv_img, method=method)
#             rgb_images.append(rgb_img)
#         return rgb_images
    
#     def get_conversion_matrix(self, method: str = 'bt709') -> np.ndarray:
#         """
#         获取YUV到RGB的转换矩阵
#         """
#         matrices = {
#             'bt601': np.array([
#                 [1.0,  0.0,       1.402],
#                 [1.0, -0.344136, -0.714136],
#                 [1.0,  1.772,     0.0]
#             ], dtype=np.float32),
#             'bt709': np.array([
#                 [1.0,  0.0,       1.5748],
#                 [1.0, -0.187324, -0.468124],
#                 [1.0,  1.8556,    0.0]
#             ], dtype=np.float32),
#             'bt2020': np.array([
#                 [1.0,  0.0,       1.4746],
#                 [1.0, -0.164553, -0.571353],
#                 [1.0,  1.8814,    0.0]
#             ], dtype=np.float32)
#         }
        
#         if method not in matrices:
#             raise ValueError(f"Unknown method: {method}")
        
#         return matrices[method]
    
#     def verify_conversion(self, original_rgb: np.ndarray, 
#                           yuv_image: np.ndarray,
#                           method: str = 'bt709',
#                           tolerance: float = 0.01) -> dict:
#         """
#         【重构版】验证YUV到RGB转换的准确性
        
#         此函数为测试工具，它会自行处理输入的归一化。
#         """
        
#         # --- 1. 归一化输入 ---
#         if original_rgb.dtype == np.uint16:
#             orig_float = original_rgb.astype(np.float32) / 65535.0
#         elif original_rgb.dtype == np.uint8:
#             orig_float = original_rgb.astype(np.float32) / 255.0
#         else:
#             orig_float = original_rgb # 假设已经是 float [0, 1]

#         if yuv_image.dtype == np.uint16:
#             yuv_float = yuv_image.astype(np.float32) / 65535.0
#         elif yuv_image.dtype == np.uint8:
#             yuv_float = yuv_image.astype(np.float32) / 255.0
#         else:
#             yuv_float = yuv_image # 假设已经是 float [0, 1]
            
#         # --- 2. 执行转换 ---
#         recon_float = self.execute(yuv_float, method=method)
        
#         # --- 3. 计算误差 ---
#         diff = np.abs(orig_float - recon_float)
#         max_error = np.max(diff)
#         mean_error = np.mean(diff)
#         mse = np.mean(diff ** 2)
#         psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        
#         within_tolerance = max_error <= tolerance
        
#         results = {
#             'max_error': max_error, 'mean_error': mean_error, 'mse': mse,
#             'psnr': psnr, 'within_tolerance': within_tolerance, 'tolerance': tolerance
#         }
        
#         print("\n=== YUV to RGB Conversion Verification ===")
#         print(f"Method: {method}")
#         print(f"Max Error:  {max_error:.6f}")
#         print(f"Mean Error: {mean_error:.6f}")
#         print(f"PSNR:       {psnr:.2f} dB")
#         print(f"Within Tolerance ({tolerance}): {'✓ Yes' if within_tolerance else '✗ No'}")
        
#         return results