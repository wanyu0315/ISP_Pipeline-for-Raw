# demosaic.py (最终混合版本 - 已适配流水线)

import numpy as np
import colour_demosaicing as cdm
import cv2
import tifffile
import rawpy
import io

class Demosaic:
    """
    去马赛克处理模块
    【重构】: 不再加载文件。接收一个 2D Bayer 数组作为输入。
    """
    def __init__(self, bayer_pattern: str, dtype: np.dtype = np.uint16):
        """
        初始化模块，必须提供RAW图像的元数据。
        
        【重构】: 从 main.py 接收 dtype
        """
        self.dtype = dtype
        self.bayer_pattern = bayer_pattern.upper()
        print(f"✅ Demosaic 模块已初始化: Pattern={self.bayer_pattern}, DType={self.dtype}")

    
    def _demosaic_with_rawpy(self, bayer_array: np.ndarray, algorithm: str) -> np.ndarray:
        """使用rawpy后端进行去马赛克"""
        print(f"  - 使用 rawpy 后端 (算法: {algorithm})")
        # 1. 定义CFA Pattern的数字映射和黑白电平
        # 0=Red, 1=Green, 2=Blue

        # 【修复】: 使用传入的 self.dtype
        BLACK_LEVEL = 1024 if self.dtype == np.uint16 else 16
        WHITE_LEVEL = np.iinfo(self.dtype).max # 使用数据类型的最大值作为白电平
        black_level_4_channels = (BLACK_LEVEL,) * 4

        pattern_map = {
            'RGGB': (0, 1, 1, 2),
            'GRBG': (1, 0, 2, 1),
            'GBRG': (1, 2, 0, 1),
            'BGGR': (2, 1, 1, 0),
        }
        cfa_pattern = pattern_map.get(self.bayer_pattern)
        if cfa_pattern is None:
            raise ValueError(f"不支持的Bayer Pattern: {self.bayer_pattern}")

        # 2. 内存里构造一个带 CFA 标签的TIFF
        with io.BytesIO() as tiff_buffer:
            with tifffile.TiffWriter(tiff_buffer, bigtiff=False) as tif:
                tif.write(
                    bayer_array,
                    photometric='cfa',
                    extratags=[
                        (33421, 'H', 2, (2, 2)), 
                        (33422, 'B', 4, cfa_pattern),
                        (37380, 'H', 4, black_level_4_channels),
                        (37384, 'H', 1, WHITE_LEVEL),
                    ]
                )
            
            # 3. 让rawpy从内存缓冲区中读取这个TIFF文件
            tiff_buffer.seek(0)
            with rawpy.imread(tiff_buffer) as raw:
                # 4. 使用rawpy的postprocess方法和指定算法
                algo_map = {
                    'AHD': rawpy.DemosaicAlgorithm.AHD,
                    'LMMSE': rawpy.DemosaicAlgorithm.LMMSE,
                    'EA': rawpy.DemosaicAlgorithm.EA,
                }
                rgb_image = raw.postprocess(
                    demosaic_algorithm=algo_map[algorithm.upper()],
                    use_camera_wb=False,
                    no_auto_bright=True,
                    # 【修复】: 使用传入的 self.dtype
                    output_bps=16 if self.dtype == np.uint16 else 8
                )
        return rgb_image


    def _demosaic_with_colour(self, bayer_array: np.ndarray, algorithm: str) -> np.ndarray:
        """使用colour-demosaicing后端"""
        print(f"  - 使用 colour-demosaicing 后端 (算法: {algorithm})")
        
        # 【修复】: 使用传入的 self.dtype
        max_val = np.iinfo(self.dtype).max
        bayer_float = bayer_array.astype(np.float64) / max_val

        if algorithm.lower() == 'bilinear':
            rgb_float = cdm.demosaicing_CFA_Bayer_bilinear(bayer_float, pattern=self.bayer_pattern)
        elif algorithm.lower() == 'malvar2004':
            rgb_float = cdm.demosaicing_CFA_Bayer_Malvar2004(bayer_float, pattern=self.bayer_pattern)
        elif algorithm.lower() == 'menon2007':
            rgb_float = cdm.demosaicing_CFA_Bayer_Menon2007(bayer_float, pattern=self.bayer_pattern)
        else:
            raise ValueError("内部错误：不应由此函数处理的算法。")
            
        return np.clip(rgb_float * max_val, 0, max_val).astype(self.dtype)
        

    def _demosaic_with_cv2(self, bayer_array: np.ndarray, algorithm: str) -> np.ndarray:
        """使用OpenCV后端"""
        print(f"  - 使用 OpenCV 后端 (算法: {algorithm})")
        pattern_map = {
            'RGGB': cv2.COLOR_BAYER_RG2RGB,
            'GRBG': cv2.COLOR_BAYER_GR2RGB,
            'GBRG': cv2.COLOR_BAYER_GB2RGB,
            'BGGR': cv2.COLOR_BAYER_BG2RGB,
        }
        
        cv_pattern = pattern_map.get(self.bayer_pattern)
        if cv_pattern is None:
             raise ValueError(f"不支持的Bayer Pattern: {self.bayer_pattern}")

        if algorithm.upper() == 'CV_VNG':
             # VNG 算法需要 3 通道输出
             return cv2.cvtColor(bayer_array, cv_pattern.replace("RGB", "RGB_VNG"))
        
        # 默认 'CV'
        return cv2.cvtColor(bayer_array, cv_pattern)


    def execute(self, bayer_array: np.ndarray, algorithm: str = 'AHD') -> np.ndarray:
        """
        【重构】执行去马赛克操作，并根据算法自动选择后端。

        Args:
            bayer_array (np.ndarray): 2D Bayer 数组 (来自 RawDenoise)
            algorithm (str):
                 - rawpy后端: 'AHD', 'LMMSE', 'EA'
                 - colour-demosaicing后端: 'Bilinear', 'Malvar2004', 'Menon2007'
                 - OpenCV后端: 'CV', 'CV_VNG'
        """
        # 【修改】: 不再从文件读取，而是打印传入的数组信息
        print(f"Executing Demosaic on array (shape: {bayer_array.shape}) with algorithm: {algorithm}")
        
        # 确保传入的 bayer_array 具有正确的 dtype
        if bayer_array.dtype != self.dtype:
            print(f"警告: Demosaic 模块期望 dtype={self.dtype}，但收到了 {bayer_array.dtype}。将尝试转换。")
            bayer_array = bayer_array.astype(self.dtype)

        # 定义算法归属
        RAWPY_ALGOS = ['AHD', 'LMMSE', 'EA']
        COLOUR_ALGOS = ['BILINEAR', 'MALVAR2004', 'MENON2007']
        CV2_ALGOS = ['CV','CV_VNG']

        # bayer_array 已经是传入的参数了

        # 2. 根据算法选择合适的处理函数
        algo_upper = algorithm.upper()
        if algo_upper in RAWPY_ALGOS:
            return self._demosaic_with_rawpy(bayer_array, algo_upper)
        elif algo_upper in COLOUR_ALGOS:
            return self._demosaic_with_colour(bayer_array, algo_upper)
        elif algo_upper in CV2_ALGOS:
            return self._demosaic_with_cv2(bayer_array, algo_upper)
        else:
            print(f"警告: 不支持的去马赛克算法: {algorithm}。将回退到 'AHD'。")
            return self._demosaic_with_rawpy(bayer_array, 'AHD')