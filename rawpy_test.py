import numpy as np
import tifffile
import io
from typing import Tuple, Optional
# import pydng

class RAWToDNGConverter:
    """将裸RAW数据转换为DNG格式的转换器"""
    
    def __init__(self):
        # DNG/TIFF标签定义
        self.TIFF_TAGS = {
            # 基本TIFF标签
            'ImageWidth': 256,
            'ImageLength': 257,
            'BitsPerSample': 258,
            'Compression': 259,
            'PhotometricInterpretation': 262,
            'StripOffsets': 273,
            'SamplesPerPixel': 277,
            'RowsPerStrip': 278,
            'StripByteCounts': 279,
            'PlanarConfiguration': 284,
            'Software': 305,
            
            # DNG特定标签
            'DNGVersion': 50706,
            'DNGBackwardVersion': 50707,
            'UniqueCameraModel': 50708,
            'ColorMatrix1': 50721,
            'CFARepeatPatternDim': 33421,
            'CFAPattern': 33422,
            'CFAPlaneColor': 50710,
            'CFALayout': 50711,
            'LinearizationTable': 50712,
            'BlackLevel': 50714,
            'WhiteLevel': 50717,
            'DefaultScale': 50718,
            'DefaultCropOrigin': 50719,
            'DefaultCropSize': 50720,
            'CalibrationIlluminant1': 50778,
            'AnalogBalance': 50727,
        }
        
        # Bayer模式映射
        self.BAYER_PATTERNS = {
            'RGGB': (0, 1, 1, 2),
            'GRBG': (1, 0, 2, 1), 
            'GBRG': (1, 2, 0, 1),
            'BGGR': (2, 1, 1, 0),
        }

    def create_dng_basic(self, 
                        bayer_array: np.ndarray, 
                        bayer_pattern: str = 'RGGB',
                        output_path: Optional[str] = None) -> bytes:
        """创建基本的DNG文件"""
        
        height, width = bayer_array.shape
        
        if bayer_pattern not in self.BAYER_PATTERNS:
            raise ValueError(f"不支持的Bayer Pattern: {bayer_pattern}")
        
        cfa_pattern = self.BAYER_PATTERNS[bayer_pattern]
        
        # 确保是16位数据
        if bayer_array.dtype != np.uint16:
            bayer_array = bayer_array.astype(np.uint16)
        
        # 创建DNG标签
        dng_tags = [
            # DNG版本信息
            (self.TIFF_TAGS['DNGVersion'], 'B', 4, (1, 4, 0, 0)),
            (self.TIFF_TAGS['DNGBackwardVersion'], 'B', 4, (1, 4, 0, 0)),
            (self.TIFF_TAGS['UniqueCameraModel'], 's', None, 'Custom Camera\0'),
            
            # CFA信息
            (self.TIFF_TAGS['CFARepeatPatternDim'], 'H', 2, (2, 2)),
            (self.TIFF_TAGS['CFAPattern'], 'B', 4, cfa_pattern),
            (self.TIFF_TAGS['CFAPlaneColor'], 'B', 3, (0, 1, 2)),  # RGB
            (self.TIFF_TAGS['CFALayout'], 'H', 1, 1),  # 矩形排列
            
            # 颜色信息
            (self.TIFF_TAGS['ColorMatrix1'], 'I', 9, (
                int(1.0 * 10000), 0, 0,
                0, int(1.0 * 10000), 0, 
                0, 0, int(1.0 * 10000)
            )),
            (self.TIFF_TAGS['CalibrationIlluminant1'], 'H', 1, 21),  # D65
            
            # 数据范围
            (self.TIFF_TAGS['BlackLevel'], 'H', 1, 0),
            (self.TIFF_TAGS['WhiteLevel'], 'H', 1, 65535),
            
            # 裁剪信息
            (self.TIFF_TAGS['DefaultCropOrigin'], 'H', 2, (0, 0)),
            (self.TIFF_TAGS['DefaultCropSize'], 'H', 2, (width, height)),
            (self.TIFF_TAGS['DefaultScale'], 'H', 2, (10000, 10000)),
            
            # 软件信息
            (self.TIFF_TAGS['Software'], 's', None, 'Python RAW to DNG Converter\0'),
        ]
        
        # 写入TIFF/DNG文件
        if output_path:
            with tifffile.TiffWriter(output_path, bigtiff=False) as tif:
                tif.write(
                    bayer_array,
                    photometric='cfa',
                    compression='none',
                    extratags=dng_tags
                )
            with open(output_path, 'rb') as f:
                return f.read()
        else:
            # 返回内存中的DNG数据
            with io.BytesIO() as buffer:
                with tifffile.TiffWriter(buffer, bigtiff=False) as tif:
                    tif.write(
                        bayer_array,
                        photometric='cfa',
                        compression='none',
                        extratags=dng_tags
                    )
                return buffer.getvalue()

    def create_dng_advanced(self, 
                           bayer_array: np.ndarray,
                           bayer_pattern: str = 'RGGB',
                           camera_model: str = 'Custom Camera',
                           iso: int = 100,
                           exposure_time: float = 1.0/60,
                           aperture: float = 2.8,
                           focal_length: float = 50.0,
                           output_path: Optional[str] = None) -> bytes:
        """创建包含更多元数据的高级DNG文件"""
        
        height, width = bayer_array.shape
        cfa_pattern = self.BAYER_PATTERNS[bayer_pattern]
        
        # 确保是16位数据
        if bayer_array.dtype != np.uint16:
            bayer_array = bayer_array.astype(np.uint16)
        
        # 创建扩展的DNG标签
        advanced_tags = [
            # DNG版本信息
            (self.TIFF_TAGS['DNGVersion'], 'B', 4, (1, 4, 0, 0)),
            (self.TIFF_TAGS['DNGBackwardVersion'], 'B', 4, (1, 4, 0, 0)),
            (self.TIFF_TAGS['UniqueCameraModel'], 's', None, f'{camera_model}\0'),
            
            # CFA信息
            (self.TIFF_TAGS['CFARepeatPatternDim'], 'H', 2, (2, 2)),
            (self.TIFF_TAGS['CFAPattern'], 'B', 4, cfa_pattern),
            (self.TIFF_TAGS['CFAPlaneColor'], 'B', 3, (0, 1, 2)),
            (self.TIFF_TAGS['CFALayout'], 'H', 1, 1),
            
            # 颜色矩阵 (sRGB近似)
            (self.TIFF_TAGS['ColorMatrix1'], 'I', 9, (
                int(0.4124 * 10000), int(0.3576 * 10000), int(0.1805 * 10000),
                int(0.2126 * 10000), int(0.7152 * 10000), int(0.0722 * 10000),
                int(0.0193 * 10000), int(0.1192 * 10000), int(0.9505 * 10000)
            )),
            (self.TIFF_TAGS['CalibrationIlluminant1'], 'H', 1, 21),  # D65
            
            # 数据范围
            (self.TIFF_TAGS['BlackLevel'], 'H', 4, (64, 64, 64, 64)),  # 每个通道的黑电平
            (self.TIFF_TAGS['WhiteLevel'], 'H', 1, 65535),
            
            # 裁剪和缩放
            (self.TIFF_TAGS['DefaultCropOrigin'], 'H', 2, (0, 0)),
            (self.TIFF_TAGS['DefaultCropSize'], 'H', 2, (width, height)),
            (self.TIFF_TAGS['DefaultScale'], 'H', 2, (10000, 10000)),
            
            # EXIF信息
            (271, 's', None, f'{camera_model}\0'),  # Make
            (272, 's', None, f'{camera_model}\0'),  # Model
            (34665, 'L', 1, 0),  # ExifIFD占位符
            
            # 软件信息
            (self.TIFF_TAGS['Software'], 's', None, 'Python Advanced RAW to DNG Converter\0'),
        ]
        
        # 写入文件
        if output_path:
            with tifffile.TiffWriter(output_path, bigtiff=False) as tif:
                tif.write(
                    bayer_array,
                    photometric='cfa',
                    compression='none',
                    extratags=advanced_tags
                )
            with open(output_path, 'rb') as f:
                return f.read()
        else:
            with io.BytesIO() as buffer:
                with tifffile.TiffWriter(buffer, bigtiff=False) as tif:
                    tif.write(
                        bayer_array,
                        photometric='cfa', 
                        compression='none',
                        extratags=advanced_tags
                    )
                return buffer.getvalue()

    # def create_dng_with_pydng(self, 
    #                          bayer_array: np.ndarray,
    #                          bayer_pattern: str = 'RGGB',
    #                          output_path: str = 'output.dng') -> str:
    #     """使用PiDNG库创建DNG文件（需要安装PiDNG）"""
    #     try:
    #         from PiDNG.core import RPICAM2DNG
            
    #         # 创建DNG转换器
    #         dng_converter = RPICAM2DNG()
            
    #         # 设置Bayer模式
    #         pattern_map = {
    #             'RGGB': 'RGGB',
    #             'GRBG': 'GRBG', 
    #             'GBRG': 'GBRG',
    #             'BGGR': 'BGGR',
    #         }
            
    #         # 将numpy数组转换为字节
    #         raw_bytes = bayer_array.tobytes()
            
    #         # 转换为DNG
    #         dng_converter.convert(
    #             raw_bytes, 
    #             width=bayer_array.shape[1],
    #             height=bayer_array.shape[0],
    #             output_path=output_path,
    #             bayer_pattern=pattern_map[bayer_pattern]
    #         )
            
    #         return output_path
            
    #     except ImportError:
    #         raise ImportError("需要安装PiDNG库: pip install pydng")

def demo_raw_to_dng_conversion():
    """演示RAW到DNG转换的完整流程"""
    
    # 输入16位Bayer数据
    bayer_data = "raw_images_sequence\raw_frame_Color_1758105206261.74291992187500.raw"

    
    # 创建转换器
    converter = RAWToDNGConverter()
    
    # 方法1：基本DNG转换
    print("\n=== 基本DNG转换 ===")
    basic_dng = converter.create_dng_basic(
        bayer_data, 
        bayer_pattern='GRBG',
        output_path='basic_output.dng'
    )
    print(f"基本DNG文件大小: {len(basic_dng)} bytes")
    print("已保存: basic_output.dng")
    
    # 方法2：高级DNG转换
    print("\n=== 高级DNG转换 ===")
    advanced_dng = converter.create_dng_advanced(
        bayer_data,
        bayer_pattern='GRBG',
        camera_model='Test Camera v1.0',
        iso=200,
        exposure_time=1.0/125,
        output_path='advanced_output.dng'
    )
    print(f"高级DNG文件大小: {len(advanced_dng)} bytes")
    print("已保存: advanced_output.dng")
    
    # # 方法3：内存中创建DNG
    # print("\n=== 内存DNG创建 ===")
    # memory_dng = converter.create_dng_basic(bayer_data, 'RGGB')
    # print(f"内存DNG大小: {len(memory_dng)} bytes")
    
    # 验证DNG文件
    print("\n=== 验证DNG文件 ===")
    try:
        import rawpy
        
        # 测试基本DNG
        with rawpy.imread('basic_output.dng') as raw:
            print(f"基本DNG验证成功:")
            print(f"  - 图像尺寸: {raw.raw_image.shape}")
            print(f"  - 数据类型: {raw.raw_image.dtype}")
            print(f"  - 数据范围: {raw.raw_image.min()} - {raw.raw_image.max()}")
            
            # 尝试处理
            rgb = raw.postprocess(
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                use_camera_wb=False,
                no_auto_bright=True
            )
            print(f"  - 处理后RGB: {rgb.shape}, {rgb.dtype}")
            
        # 测试高级DNG
        with rawpy.imread('advanced_output.dng') as raw:
            print(f"高级DNG验证成功:")
            print(f"  - 图像尺寸: {raw.raw_image.shape}")
            
    except ImportError:
        print("需要安装rawpy进行验证: pip install rawpy")
    except Exception as e:
        print(f"DNG验证失败: {e}")

# 使用示例
def convert_your_raw_file(raw_file_path: str, 
                         width: int, 
                         height: int,
                         bayer_pattern: str = 'GRBG',
                         bit_depth: int = 16) -> str:
    """转换您的实际RAW文件到DNG"""
    
    # 读取RAW文件
    if bit_depth == 16:
        dtype = np.uint16
    else:
        dtype = np.uint8
    
    with open(raw_file_path, 'rb') as f:
        raw_data = np.frombuffer(f.read(), dtype=dtype)
    
    # 重塑为2D数组
    bayer_array = raw_data.reshape((height, width))
    
    # 创建转换器并转换
    converter = RAWToDNGConverter()
    output_path = raw_file_path.replace('.raw', '.dng')
    
    converter.create_dng_advanced(
        bayer_array,
        bayer_pattern=bayer_pattern,
        camera_model='Custom RAW Camera',
        output_path=output_path
    )
    
    print(f"转换完成: {raw_file_path} -> {output_path}")
    return output_path

if __name__ == "__main__":
    # 运行演示
    # demo_raw_to_dng_conversion()

    # 转换您的文件示例
    output_dng = convert_your_raw_file(
        r"raw_images_sequence\raw_frame_Color_1758105206261.74291992187500.raw",
        width=1280, 
        height=800, 
        bayer_pattern='GRBG'
    )