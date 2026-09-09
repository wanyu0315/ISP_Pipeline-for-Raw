# raw_loader.py
import numpy as np
import os


class CorruptRawFrameError(RuntimeError):
    """Raised when a RAW frame exceeds the allowed total of near-zero rows."""


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if starts.size == 0:
        return 0, -1

    lengths = ends - starts
    longest_index = int(np.argmax(lengths))
    return int(lengths[longest_index]), int(starts[longest_index])


def detect_zero_row_band(
        bayer_image: np.ndarray,
        max_zero_rows: int = 10) -> dict:
    """Count near-zero RAW rows, including separated rows; keep band statistics for diagnostics."""
    if bayer_image.ndim != 2:
        raise ValueError(f"RAW 坏帧检测期望二维 Bayer 图像，实际形状: {bayer_image.shape}")
    if isinstance(max_zero_rows, (bool, np.bool_)) or not isinstance(max_zero_rows, (int, np.integer)) or max_zero_rows < 0:
        raise ValueError("max_zero_rows 必须为非负整数")

    if np.issubdtype(bayer_image.dtype, np.integer):
        zero_tolerance = 1.0
    else:
        finite_values = bayer_image[np.isfinite(bayer_image)]
        max_value = float(np.max(finite_values)) if finite_values.size else 0.0
        zero_tolerance = 1.0 / 65535.0 if max_value <= 1.5 else 1.0

    row_max = np.max(bayer_image, axis=1)
    zero_rows = row_max <= zero_tolerance
    longest_rows, start_row = _longest_true_run(zero_rows)
    total_zero_rows = int(np.count_nonzero(zero_rows))
    required_rows = int(max_zero_rows) + 1

    remaining_pixels = bayer_image[~zero_rows]
    remaining_median = (
        float(np.median(remaining_pixels))
        if remaining_pixels.size else 0.0
    )
    is_corrupt = (
        total_zero_rows >= required_rows
        and remaining_median > zero_tolerance
    )

    return {
        "is_corrupt": bool(is_corrupt),
        "start_row": int(start_row),
        "end_row": int(start_row + longest_rows) if start_row >= 0 else -1,
        "band_rows": int(longest_rows),
        "total_zero_rows": total_zero_rows,
        "max_zero_rows": int(max_zero_rows),
        "required_rows": int(required_rows),
        "zero_tolerance": float(zero_tolerance),
        "remaining_median": remaining_median,
    }


class RawLoader:
    """
    ISP管道的第一个模块：负责加载无头 (headerless) RAW 文件或 .npy RAW 帧。
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
        self.last_probe_telemetry = {}
        print(f"✅ RawLoader 模块已初始化: W={self.width}, H={self.height}, DType={self.dtype}")

    def execute(self, raw_file_path: str, **kwargs) -> np.ndarray:
        """
        执行加载操作。
        
        Args:
            raw_file_path: 输入的 .raw 或 .npy 文件路径 (字符串)
        
        Returns:
            2D Bayer 图像数组 (np.ndarray) - [修改] 已转换为 float32 且归一化至 [0.0, 1.0]
        """
        ext = os.path.splitext(raw_file_path)[1].lower()
        actual_file_size = os.path.getsize(raw_file_path)

        if ext == ".npy":
            bayer_image = np.load(raw_file_path)
            expected_shape = (self.height, self.width)
            if bayer_image.shape != expected_shape:
                raise ValueError(
                    f"错误: .npy 图像形状 {bayer_image.shape} 与预期 {expected_shape} 不符: "
                    f"{raw_file_path}"
                )
        else:
            # 检查 headerless RAW 文件大小是否匹配
            file_size = actual_file_size
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

        integrity = detect_zero_row_band(bayer_image)
        if integrity["is_corrupt"]:
            raise CorruptRawFrameError(
                "RAW 近零行累计超限: "
                f"total_zero_rows={integrity['total_zero_rows']}, "
                f"max_zero_rows={integrity['max_zero_rows']}, "
                f"longest_band_rows={integrity['band_rows']}"
            )
            
        # ===================================================================
        # 🌟 [修改] 零损耗数据流：阶段一 - 入口提权
        # 将原始 uint16 数据立即转化为 float32，并归一化到 [0.0, 1.0]
        # 后续所有 ISP 模块都将在此高精度空间内进行数学运算
        # ===================================================================
        source_dtype = str(bayer_image.dtype)
        source_min = float(np.nanmin(bayer_image))
        source_max = float(np.nanmax(bayer_image))
        if np.issubdtype(bayer_image.dtype, np.integer):
            max_val = float(np.iinfo(bayer_image.dtype).max)
            bayer_image_float = bayer_image.astype(np.float32) / max_val
            quantization_step = 1.0 / max_val
        else:
            max_val = 1.0
            bayer_image_float = bayer_image.astype(np.float32)
            quantization_step = 0.0

        self.last_probe_telemetry = {
            "source_path": os.path.abspath(raw_file_path),
            "source_extension": ext or ".raw",
            "source_dtype": source_dtype,
            "source_shape": list(bayer_image.shape),
            "expected_shape": [int(self.height), int(self.width)],
            "file_size_bytes": int(actual_file_size),
            "expected_file_size_bytes": int(self.frame_size),
            "file_size_matches": bool(ext == ".npy" or actual_file_size == self.frame_size),
            "normalization_scale": float(max_val),
            "normalization_quantization_step": float(quantization_step),
            "source_min": source_min,
            "source_max": source_max,
            "source_zero_fraction": float(np.mean(bayer_image == 0)),
            "source_saturation_fraction": float(np.mean(bayer_image >= max_val)),
            "zero_row_band_check": integrity,
        }
            
        print("--- RawLoader: 文件加载完成 (已提权至 Float32 [0.0, 1.0]) ---")
        return bayer_image_float
