# color_space_conversion.py
import numpy as np
import cv2

class ColorSpaceConversion:
    """
    颜色空间转换模块：RGB -> YUV
    输出将保持为 float32 的高动态范围 (HDR) 数据，以便后续 YUV 模块无损处理。
    """
    
    def execute(self, rgb_image: np.ndarray, method: str = 'bt709') -> np.ndarray:
        """
        执行RGB到YUV的颜色空间转换。
        
        Args:
            rgb_image: 输入的RGB图像 (严格期望 float32)
            method: 转换方法 ('bt601', 'bt709', 'bt2020', 'opencv_bt601')
        
        Returns:
            YUV格式的图像 (dtype=float32)
        """
        print(f"Executing Color Space Conversion (RGB->YUV) using method: {method}")
        
        # ===================================================================
        # 🌟 零损耗数据流：信任上游数据
        # 直接使用输入数据，仅做防御性的 float32 强转，不做任何数值缩放
        # ===================================================================
        if not np.issubdtype(rgb_image.dtype, np.floating):
            print("  [警告] ColorSpaceConversion 期望输入 float32，正在强制转换...")
            img_float = rgb_image.astype(np.float32)
        else:
            img_float = rgb_image
        
        # 2. 执行转换
        if method == 'bt601':
            yuv_image = self._rgb_to_yuv_bt601(img_float)
        elif method == 'bt709':
            yuv_image = self._rgb_to_yuv_bt709(img_float)
        elif method == 'bt2020':
            yuv_image = self._rgb_to_yuv_bt2020(img_float)
        elif method == 'opencv_bt601':
            yuv_image = self._rgb_to_yuv_opencv_float(img_float)
        else:
            raise ValueError(f"Unknown color space conversion method: {method}")
        
        # 3. 返回高精度 HDR 图像
        return yuv_image
    
    def _rgb_to_yuv_bt601(self, rgb_float: np.ndarray) -> np.ndarray:
        """BT.601标准转换矩阵 (SDTV)"""
        transform_matrix = np.array([
            [ 0.299,     0.587,     0.114],
            [-0.168736, -0.331264,  0.5],
            [ 0.5,      -0.418688, -0.081312]
        ], dtype=np.float32)
        yuv = np.dot(rgb_float, transform_matrix.T)
        yuv[:, :, 1:] += 0.5 # 偏移 U, V
        
        # ===================================================================
        # 🌟 零损耗数据流：移除 np.clip，允许 Y 大于 1.0，允许 U/V 溢出边界
        # ===================================================================
        return yuv
    
    def _rgb_to_yuv_bt709(self, rgb_float: np.ndarray) -> np.ndarray:
        """BT.709标准转换矩阵 (HDTV)"""
        transform_matrix = np.array([
            [ 0.2126,     0.7152,     0.0722],
            [-0.114572, -0.385428,   0.5],
            [ 0.5,       -0.454153, -0.045847]
        ], dtype=np.float32)
        yuv = np.dot(rgb_float, transform_matrix.T)
        yuv[:, :, 1:] += 0.5
        
        # 移除 np.clip
        return yuv
    
    def _rgb_to_yuv_bt2020(self, rgb_float: np.ndarray) -> np.ndarray:
        """BT.2020标准转换矩阵 (UHDTV)"""
        transform_matrix = np.array([
            [ 0.2627,     0.6780,     0.0593],
            [-0.139630, -0.360370,   0.5],
            [ 0.5,       -0.459786, -0.040214]
        ], dtype=np.float32)
        yuv = np.dot(rgb_float, transform_matrix.T)
        yuv[:, :, 1:] += 0.5
        
        # 移除 np.clip
        return yuv
    
    def _rgb_to_yuv_opencv_float(self, rgb_float: np.ndarray) -> np.ndarray:
        """
        使用OpenCV的YUV转换，保持float32精度
        注意：OpenCV 的 COLOR_RGB2YUV 使用的是 BT.601 标准。
        """
        # cv2.cvtColor 可以直接处理浮点数组，但在处理 >1.0 的超亮像素时
        # OpenCV 底层的色度空间映射可能存在自己内置的截断逻辑（视具体版本而定）。
        # 如果需要绝对严谨的 HDR rPPG 数据，强烈建议优先使用上面自定义的矩阵乘法 (bt709/bt601)。
        yuv_float = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2YUV)
        
        return yuv_float

# # color_space_conversion.py
# import numpy as np
# import cv2

# class ColorSpaceConversion:
#     """
#     颜色空间转换模块：RGB -> YUV
#     输出将保持为 float32 [0, 1]，以便后续YUV模块处理。
#     """
    
#     def execute(self, rgb_image: np.ndarray, method: str = 'bt709') -> np.ndarray:
#         """
#         执行RGB到YUV的颜色空间转换。
        
#         Args:
#             rgb_image: 输入的RGB图像 (来自GammaCorrection, uint16)
#             method: 转换方法 ('bt601', 'bt709', 'bt2020', 'opencv_bt601')
        
#         Returns:
#             YUV格式的图像 (dtype=float32, 范围=[0, 1])
#         """
#         print(f"Executing Color Space Conversion (RGB->YUV) using method: {method}")
        
#         # 1. 归一化到 [0, 1]
#         if rgb_image.dtype == np.uint16:
#             max_val = 65535.0
#         elif rgb_image.dtype == np.uint8:
#             max_val = 255.0
#         else:
#             # 假设已经是 float，直接使用
#             img_float = rgb_image.astype(np.float32)
#             # 确保范围
#             if img_float.max() > 1.0:
#                  img_float = np.clip(img_float / 65535.0, 0, 1) # 假设来自16bit
            
#         if rgb_image.dtype != np.float32:
#              img_float = rgb_image.astype(np.float32) / max_val
        
#         # 2. 执行转换
#         if method == 'bt601':
#             yuv_image = self._rgb_to_yuv_bt601(img_float)
#         elif method == 'bt709':
#             yuv_image = self._rgb_to_yuv_bt709(img_float)
#         elif method == 'bt2020':
#             yuv_image = self._rgb_to_yuv_bt2020(img_float)
#         elif method == 'opencv_bt601':
#             # 使用修正后的 OpenCV 方法
#             yuv_image = self._rgb_to_yuv_opencv_float(img_float)
#         else:
#             raise ValueError(f"Unknown color space conversion method: {method}")
        
#         # 3. 【重要】返回 float32 [0, 1] 图像，不要转回 uint16，让后续的 YUV 模块在 float32 上工作
#         return yuv_image
    
#     def _rgb_to_yuv_bt601(self, rgb_float: np.ndarray) -> np.ndarray:
#         """BT.601标准转换矩阵 (SDTV)"""
#         transform_matrix = np.array([
#             [ 0.299,     0.587,     0.114],
#             [-0.168736, -0.331264,  0.5],
#             [ 0.5,      -0.418688, -0.081312]
#         ], dtype=np.float32)
#         yuv = np.dot(rgb_float, transform_matrix.T)
#         yuv[:, :, 1:] += 0.5 # 偏移 U, V
#         return np.clip(yuv, 0, 1)
    
#     def _rgb_to_yuv_bt709(self, rgb_float: np.ndarray) -> np.ndarray:
#         """BT.709标准转换矩阵 (HDTV)"""
#         transform_matrix = np.array([
#             [ 0.2126,     0.7152,     0.0722],
#             [-0.114572, -0.385428,   0.5],
#             [ 0.5,       -0.454153, -0.045847]
#         ], dtype=np.float32)
#         yuv = np.dot(rgb_float, transform_matrix.T)
#         yuv[:, :, 1:] += 0.5
#         return np.clip(yuv, 0, 1)
    
#     def _rgb_to_yuv_bt2020(self, rgb_float: np.ndarray) -> np.ndarray:
#         """BT.2020标准转换矩阵 (UHDTV)"""
#         transform_matrix = np.array([
#             [ 0.2627,     0.6780,     0.0593],
#             [-0.139630, -0.360370,   0.5],
#             [ 0.5,       -0.459786, -0.040214]
#         ], dtype=np.float32)
#         yuv = np.dot(rgb_float, transform_matrix.T)
#         yuv[:, :, 1:] += 0.5
#         return np.clip(yuv, 0, 1)
    
#     def _rgb_to_yuv_opencv_float(self, rgb_float: np.ndarray) -> np.ndarray:
#         """
#         【修正版】使用OpenCV的YUV转换，保持float32精度
        
#         注意：OpenCV 的 COLOR_RGB2YUV 使用的是 BT.601 标准。
#         """
#         # cv2.cvtColor 可以直接处理 [0, 1] 范围的 float32 数组
#         # 它会正确地保持浮点精度，不会进行8-bit量化
#         yuv_float = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2YUV)
        
#         # yuv_float 已经是 [0, 1] 范围的 float32，直接返回
#         return yuv_float