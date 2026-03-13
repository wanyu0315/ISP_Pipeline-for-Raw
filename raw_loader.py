# raw_loader.py
import numpy as np
import os

class RawLoader:
    """
    ISP管道的第一个模块：负责加载无头 (headerless) RAW 文件。
    它将文件路径字符串转换为 NumPy 数组。
    """
    def __init__(self, width: int, height: int, dtype: np.dtype):
        """
        初始化加载器所需的元数据。
        
        Args:
            width: 图像宽度
            height: 图像高度
            dtype: 数据类型 (例如 np.uint16)
        """
        self.width = width
        self.height = height
        self.dtype = dtype
        self.frame_size = self.width * self.height * np.dtype(self.dtype).itemsize
        print(f"✅ RawLoader 模块已初始化: W={self.width}, H={self.height}, DType={self.dtype}")

    def execute(self, raw_file_path: str, **kwargs) -> np.ndarray:
        """
        执行加载操作。
        
        Args:
            raw_file_path: 输入的 .raw 文件路径 (字符串)
        
        Returns:
            2D Bayer 图像数组 (np.ndarray) - [修改] 已转换为 float32 且归一化至 [0.0, 1.0]
        """
        # 检查文件大小是否匹配
        file_size = os.path.getsize(raw_file_path)
        if file_size != self.frame_size:
            print(f"警告: 文件 '{raw_file_path}' 大小 ({file_size}) 与预期 ({self.frame_size}) 不符。")
            # 可以在这里抛出异常或尝试继续
        
        # 从文件加载
        bayer_image = np.fromfile(raw_file_path, dtype=self.dtype)
        
        # Reshape 为 2D
        try:
            bayer_image = bayer_image.reshape((self.height, self.width))
        except ValueError as e:
            print(f"错误: 无法将图像 reshape 为 ({self.height}, {self.width})。数据大小: {bayer_image.size}")
            raise e
            
        # ===================================================================
        # 🌟 [修改] 零损耗数据流：阶段一 - 入口提权
        # 将原始 uint16 数据立即转化为 float32，并归一化到 [0.0, 1.0]
        # 后续所有 ISP 模块都将在此高精度空间内进行数学运算
        # ===================================================================
        max_val = float(np.iinfo(self.dtype).max)
        bayer_image_float = bayer_image.astype(np.float32) / max_val
            
        print("--- RawLoader: 文件加载完成 (已提权至 Float32 [0.0, 1.0]) ---")
        return bayer_image_float