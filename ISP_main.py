# main_batch_raw.py

import imageio
import numpy as np
import json
import os
import glob
from tqdm import tqdm
import cv2
import subprocess # 导入subprocess模块

# 导入我们的ISP管道和模块
from isp_pipeline import ISPPipeline
from raw_loader import RawLoader
from black_level_correction import BlackLevelCorrection
from defect_pixel_correction import DefectPixelCorrection
from raw_denoise import RawDenoise   
from demosaic import Demosaic  
from white_balance import WhiteBalanceRaw
from gamma_correction import GammaCorrection
from color_correction_matrix import ColorCorrectionMatrix  
from gamma_correction import GammaCorrection
from color_space_conversion import ColorSpaceConversion
from denoise_in_yuv import Denoise
from sharpening import Sharpen
from contrast_and_saturation import ContrastSaturation
from yuv_to_rgb import YUVtoRGB

def main_batch():
    # --- 1. 定义传感器/图像的元数据 ---
    # !! 必须为无头RAW文件提供元数据 !!
    IMAGE_WIDTH = 1280
    IMAGE_HEIGHT = 800
    IMAGE_DTYPE = np.uint16  # 或 np.uint8, 取决于您的RAW数据位深
    BAYER_PATTERN = 'GBRG'   # 根据传感器规格设置

    # --- 2. 定义输入和输出文件夹 ---
    input_folder = 'ISPpipline/raw_data/raw_data_1' # 存放RAW序列的文件夹
    output_folder = '/home/lizize/pyVHR_for_ISP/ISPpipline/isp_output_frame/raw_denoise/bin_and_space/video_1_frame_rawdenoise_isp(6)'   # 存放处理后PNG帧的文件夹
    
    output_video_path = 'Data_for_pyVHR/isp_output_Video/rawdenoise/Video_1_bin_and_space/output_video_1_isp(6)_8bit.mkv'

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- 3. 获取所有RAW文件 ---
    #   文件扩展名为.raw，以匹配无头RAW文件的场景
    raw_files = sorted(glob.glob(os.path.join(input_folder, '*.raw')))
    
    if not raw_files:
        print(f"在文件夹 '{input_folder}' 中没有找到 .raw 文件。")
        return

    print(f"找到 {len(raw_files)} 个 .raw 文件进行处理。")

    # --- 4. 实例化并组装ISP管道 ---
    # <--- 在创建Demosaic实例时，必须传入元数据
    loader_module = RawLoader(
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            dtype=IMAGE_DTYPE
        )
    demosaic_module = Demosaic(
        bayer_pattern=BAYER_PATTERN,
        dtype=IMAGE_DTYPE
    )
    
    my_isp = ISPPipeline(modules=[
        # raw域处理
        loader_module,
        BlackLevelCorrection(),           #  黑电平
        DefectPixelCorrection(),          #  坏点校正
        RawDenoise(),
        WhiteBalanceRaw(),
        # RGB域处理
        demosaic_module,
        GammaCorrection(),

        #YUV域处理
        ColorSpaceConversion(),     
        Denoise(),               
        Sharpen(),                                                  
        ContrastSaturation(),

        #YUV——RGB处理
        YUVtoRGB()                                   
    ])

    # 加载坏点位置图
    defect_map = np.load('Data_preprocessing/defect_report/bad_points_report_long/defect_map.npy')

    # --- 5. 定义处理参数 (所有帧使用相同参数) ---
    processing_params = {
        # 黑电平参数
        'blacklevelcorrection': {
            'black_level': 0  # 从黑电平标定得到
        },
        # 坏点参数
        'defectpixelcorrection': {
            'method': 'median',       # 推荐：中值滤波
            'defect_map': defect_map, # 使用生成的坏点图
            'auto_detect': False      # 不需要自动检测
        },
        # raw域参数、
        'rawdenoise': {
            'bayer_pattern': BAYER_PATTERN,
            # 定义级联步骤列表 'steps'
            'steps': [
                # Step 1: 像素合并 (Binning) - 物理提质，降分辨率
                {
                    'algorithm': 'binning',
                    'mode': 'average'
                },
                
                # # Step 2: 时域降噪 (Temporal) - 保护 rPPG
                # {
                #     'algorithm': 'temporal',
                #     'alpha': 0.6,           # 历史权重 (0.6 表示 60% 来自当前帧，40% 历史)
                #     'motion_thresh': 0.05   # 运动阈值
                # },
                
                # Step 3: 空域降噪 (Spatial)
                {
                    'algorithm': 'gaussian', 
                    'sigma': 2.5
                    # 'd': 5,             # 直径
                    # 'sigma_color': 25,  # 颜色权重 (0-255标准)
                    # 'sigma_space': 50   # 空间权重
                }
            ]
        },
        'whitebalance': {
            'algorithm': 'gray_world_green',
            'bayer_pattern': BAYER_PATTERN
        },
                         
        # RGB域参数
        'demosaic': {'algorithm': 'CV'},
        'gammacorrection': {'gamma': 2.2},

        # YUV域参数
        'colorspaceconversion': {
            'method': 'bt709'  # HDTV标准
        },
        'denoise': {
            'algorithm': 'gaussian',
            'sigma': 0.1,
            #'process_chroma': False
        },
        'sharpen': {
            'algorithm': 'unsharp_mask',  # 专业级锐化
            'radius': 1.0,  # 半径越大，锐化影响范围越大 → 边缘变得更粗更强
            'amount': 1.2,  # 控制增强“细节差值”的比例
            'threshold': 3  # 当像素差值低于 threshold → 不做锐化
        },
        'contrastsaturation': {
            'contrast_method': 'linear',      # 自适应直方图均衡
            'saturation_method': 'vibrance',  # 智能饱和度
            'contrast_factor': 1.2,
            'saturation_factor': 1.3,
            'clip_limit': 2.0,           # 对比度限制，对比度clahe算法中的参数
            'tile_grid_size': (8, 8),   # 网格限制，对比度clahe算法中的参数
            'skin_protection': 0.5     # 肤色保护强度 (0-1)，饱和度vibrance算法中的参数
        },

        # YUV转RGB
        'yuvtorgb': {
            'method': 'bt709'  # 必须与RGB->YUV的方法一致！
        }
    }

    # --- 6. 循环处理所有文件，并保存为序列表文件名 ---

    skip_processing = False
    # 检查输出文件夹是否存在
    if os.path.isdir(output_folder):
        # 如果存在，检查里面是否已有处理好的png文件
        # 使用 glob 查找符合命名规则的文件，更精确
        existing_frames = glob.glob(os.path.join(output_folder, 'frame_*.png'))
        if existing_frames:
            print(f" 输出文件夹 '{output_folder}' 已存在且包含 {len(existing_frames)} 帧，将跳过ISP处理步骤。")
            skip_processing = True
            # 为后续视频合成步骤准备好 padding 和 total_files 变量
            total_files = len(existing_frames)
            padding = len(str(total_files)) # 根据文件数计算padding
        else:
            print(f" 输出文件夹 '{output_folder}' 已存在但为空，将开始处理RAW文件。")
    else:
        print(f" 输出文件夹 '{output_folder}' 不存在，将创建并开始处理RAW文件。")
        os.makedirs(output_folder, exist_ok=True) # 创建文件夹

    # --- 如果不需要跳过，则执行处理循环 ---
    if not skip_processing:
        print("\n 开始执行ISP处理流程...")
        # 在循环开始前，获取文件总数以确定命名格式的宽度
        try:
            total_files = len(raw_files)
            # 计算补零的位数，例如 total_files=800 -> padding=3; total_files=1234 -> padding=4
            padding = len(str(total_files)) 
        except (NameError, TypeError):
            print("错误：'raw_files' 列表不存在或为空。请确保在此代码块之前已定义 'raw_files'。")
            raw_files = [] 
            padding = 4 # 设置一个默认值

        # 定义一个“全黑”行的阈值。一行像素的平均值低于此值（满分255）几乎可以肯定是损坏的，而不是一个非常暗的场景。
        BLACK_ROW_THRESHOLD = 1.0  # 1.0 out of 255
        
        # 我们假设一个正常的帧不应该有任何“全黑”的行。
        MIN_CORRUPT_ROWS_TO_REJECT = 1

        # ⭐️ 用于存储上一帧
        last_good_frame_bgr = None

        # 初始化帧计数器
        frame_counter = 0

        # 使用 pbar.write 来打印警告，避免破坏进度条
        pbar = tqdm(raw_files, desc="Processing RAW sequence")
        for raw_file_path in pbar:

            frame_to_save = None
            
            #  将初始化移动到 'try' 块的顶部
            is_frame_corrupt = False 
            corrupt_row_count = 0

            try:
                # 1. 运行管道
                final_image = my_isp.process(raw_file_path, params=processing_params)
                       
                # 2. 转换位深
                if  final_image.dtype == np.float32:
                    # --- 决定输出位深 ---
                    # 8-bit:
                    frame_to_save = (final_image * 255.0).astype(np.uint8)
                    # 16-bit: 
                    # frame_to_save = (final_image * 65535.0).astype(np.uint16)
                else:
                    frame_to_save = final_image

                # 3. 转换颜色通道 (从 RGB -> BGR), 为了满足 cv2.imwrite 的 BGR 要求
                frame_bgr = cv2.cvtColor(frame_to_save, cv2.COLOR_RGB2BGR)

                # 4. 坏帧检测
                try:
                    # 4a. 计算行均值
                    row_means = np.mean(frame_bgr, axis=(1, 2))
                    
                    # 4b. 仅在 4a 成功后才计算
                    corrupt_row_count = np.sum(row_means < BLACK_ROW_THRESHOLD)
                    
                except Exception as e:
                    pbar.write(f"  [!] 警告: 帧 {os.path.basename(raw_file_path)} 无法计算行均值: {e}。")
                    is_frame_corrupt = True # 标记为损坏

                if corrupt_row_count >= MIN_CORRUPT_ROWS_TO_REJECT:
                    is_frame_corrupt = True
                
                # 5.  决策：保存、替换还是跳过
                if is_frame_corrupt:
                    # 这是一个损坏的帧
                    pbar.write(f"  [!] 警告: 帧 {os.path.basename(raw_file_path)} 似乎已损坏。")
                    
                    if last_good_frame_bgr is not None:
                        # 替换为上一帧
                        frame_to_save = last_good_frame_bgr
                        pbar.write(f"      ...已替换为上一帧。")
                    else:
                        # 这是第一帧，且已损坏，我们别无选择，只能跳过
                        pbar.write(f"      ...这是第一帧且已损坏，无法替换，已跳过！")
                        continue # 跳过循环，不保存也不递增计数器
                
                else:
                    # 这是一个好帧
                    frame_to_save = frame_bgr
                    last_good_frame_bgr = frame_bgr.copy() # 更新“上一好帧”

            except Exception as e:
                # 捕获ISP处理中的错误 (例如 `my_isp.process` 失败)
                pbar.write(f"  [!] 错误: 处理文件 {raw_file_path} 时出错: {e}。")
                if last_good_frame_bgr is not None:
                    # ISP处理失败，也用上一帧替换
                    frame_to_save = last_good_frame_bgr
                    pbar.write(f"      ...ISP处理失败，已替换为上一帧。")
                else:
                    # 第一帧的ISP处理就失败了
                    pbar.write(f"      ...第一帧处理失败，无法替换，已跳过！")
                    continue
            
            # 6. 保存 (无论是好帧还是替换帧)
            if frame_to_save is not None:
                new_file_name = f"frame_{frame_counter:0{padding}d}.png"
                output_path = os.path.join(output_folder, new_file_name)
                
                cv2.imwrite(output_path, frame_to_save)

                # 7. 增加计数器
                frame_counter += 1
            
        pbar.close() 
        print(f"\n✅ 所有帧处理完毕，已保存至 '{output_folder}' 文件夹，并已重命名为序列格式。")
        print(f"  共处理 {frame_counter} / {len(raw_files)} 帧 (已替换或跳过损坏帧)。") 
        
        if frame_counter > 0:
            padding = len(str(frame_counter - 1)) 
        
    else:
        print("\n🚀 直接进入视频合成步骤。")

    """
        --- 额外说明 ---
        注意使用CV的去马赛克算法时，输出图像的颜色通道顺序是BGR而不是RGB。
        因此在后续进行合成视频时，需注意这一点，如果不是使用OpenCV进行视频写入，
        可能需要转换颜色通道顺序。
        当imageio读取您的PNG文件时，它并不知道这个文件是OpenCV以BGR顺序创建的。它只是按顺序读取了三个通道的数据，
        并把它们加载到一个NumPy数组中。
        frame = imageio.imread(frame_path) 这行代码返回的frame变量，其内存中的通道顺序实际上还是 B-G-R。
    """

    # --- 7. 将处理后的帧合成为视频 (FFmpeg—16bit无损方案) ---
    
    print("正在将处理后的帧合成为无损视频 (使用 FFmpeg)...")

    # 检查帧是否存在
    processed_frames_pattern = os.path.join(output_folder, '*.png')
    frames_exist = glob.glob(processed_frames_pattern)

    if not frames_exist:
        print("错误:在输出文件夹中找不到任何处理后的帧。")
        return

   #output_video_path = 'Data_for_pyVHR/isp_output_Video/Video_1/output_video_1_isp(2)_framewell_8bit.mkv'
    framerate = 30.0

    #   显式定义编码参数，以便保存到JSON
    video_encoder = 'ffv1'
    video_pix_fmt = 'bgr24'

    if not 'padding' in locals():
         if frames_exist:
             total_files = len(frames_exist)
             padding = len(str(total_files))
         else:
             padding = 4 # 备用

    first_frame = os.path.basename(frames_exist[0])

# 尝试检测序列模式
    if 'frame_' in first_frame and first_frame.endswith('.png'):
    # 动态构建序列模式
    # 使用f-string将变量padding插入到字符串中
        sequence_pattern = os.path.join(output_folder, f'frame_%0{padding}d.png').replace('\\', '/')
        command = [
        'ffmpeg',
        '-y',
        '-framerate', str(framerate),  # 输入帧率
        '-start_number', '0',  # 如果帧从frame_000.png开始
        '-i', sequence_pattern,
        '-c:v', video_encoder,  # 编码器（ffv1，libx264等）
        '-level', '3',
        '-pix_fmt', video_pix_fmt,  # 像素格式(bgr48le、bgr24、yuv420p等),注意需要和上面处理后的视频帧通道格式对应，OpenCV是BGR格式
        '-slices', '24',  # 多线程编码,提升性能
        '-slicecrc', '1',  # 错误检测
        '-r', str(framerate),  # 明确指定输出帧率
        '-vsync', 'cfr',  # 恒定帧率
        output_video_path
    ]
    try:
        print(f"执行FFmpeg命令: {' '.join(command)}")
        
        # Windows推荐的执行方式
        result = subprocess.run(
            command,  # 直接传递列表,不使用shell=True更安全
            check=True,
            capture_output=True,
            text=True
        )
        
        print(f"无损视频已成功创建: {output_video_path}")

        # ⭐️ [新增] 保存ISP和编码参数到JSON文件
        print(f"正在保存参数到JSON文件...")
        
        # 1. 准备要保存的数据
        metadata_to_save = {
            'isp_processing_params': processing_params,
            'video_encoding_params': {
                'encoder': video_encoder,
                'pixel_format': video_pix_fmt,
                'framerate': framerate,
                'output_video_file': os.path.basename(output_video_path),
                'input_sequence_pattern': os.path.basename(sequence_pattern)
            }
        }
        
        # 2. 定义JSON输出路径 (例如: output_video.mkv -> output_video.json)
        json_output_path = os.path.splitext(output_video_path)[0] + '.json'
        
        # 3. 写入文件
        try:
            # 使用 utf-8 编码确保中文（如果未来有的话）和特殊字符正确保存
            with open(json_output_path, 'w', encoding='utf-8') as f:
                # indent=4 使JSON文件格式化，更易读
                json.dump(metadata_to_save, f, indent=4)
            print(f"✓ 参数JSON文件已成功保存: {json_output_path}")
        except Exception as e:
            print(f"✗ 保存JSON参数文件失败: {e}")

        print(f"\n视频信息:")
        print(f"- 帧数: {len(frames_exist)}")
        print(f"- 帧率: {framerate} fps")
        print(f"- 时长: {len(frames_exist)/framerate:.2f} 秒")
        
    except subprocess.CalledProcessError as e:
        print("FFmpeg 执行失败!")
        print(f"返回码: {e.returncode}")
        if e.stdout:
            print(f"标准输出:\n{e.stdout}")
        if e.stderr:
            print(f"错误输出:\n{e.stderr}")
    except FileNotFoundError:
        print("错误: 找不到FFmpeg。请确保FFmpeg已安装并添加到系统PATH中。")


    # # --- 7. (可选) 将处理后的帧合成为视频 (OpenCV-MKV无损方案) ---
    # print("正在将处理后的帧合成为无损视频 (FFV1)...")
    # processed_frames = sorted(glob.glob(os.path.join(output_folder, '*.png')))
    
    # if not processed_frames:
    #     # ... (错误处理)
    #     return
        
    # first_frame = cv2.imread(processed_frames[0], cv2.IMREAD_UNCHANGED)
    # height, width, _ = first_frame.shape

    # #  指定输出文件为 .avi 或 .mkv，它们对FFV1支持更好
    # output_video_path = 'output_video_lossless.mkv'
    
    # #  使用 FFV1 的 FourCC 代码
    # fourcc = cv2.VideoWriter_fourcc(*'FFV1') 
    # writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    # if not writer.isOpened():
    #     print("无法打开VideoWriter，请检查OpenCV配置。")
    #     return

    # for frame_path in tqdm(processed_frames, desc="Creating Lossless Video"):
    #     frame_16bit_bgr = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    #     frame_8bit_bgr = (frame_16bit_bgr / 257.0).astype(np.uint8)
    #     writer.write(frame_8bit_bgr)
            
    # writer.release()
    # print(f"无损视频 '{output_video_path}' 创建成功！")


    # # --- 7. (可选) 将处理后的帧合成为视频 (OpenCV-MP4格式) ---
    # print("正在将处理后的帧合成为视频 (使用OpenCV)...")
    # processed_frames = sorted(glob.glob(os.path.join(output_folder, '*.png')))
    
    # if not processed_frames:
    #     print("没有找到已处理的帧，无法创建视频。")
    #     return
        
    # #  从第一张图片获取视频的尺寸
    # first_frame = cv2.imread(processed_frames[0], cv2.IMREAD_UNCHANGED)
    # if first_frame is None:
    #     print(f"无法读取第一帧图像: {processed_frames[0]}")
    #     return
    # height, width, _ = first_frame.shape

    # #  定义视频编码器和创建 VideoWriter 对象
    # # 'mp4v' 是一个常用的MP4编码器
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # writer = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (width, height))

    # if not writer.isOpened():
    #     print("无法打开VideoWriter，请检查OpenCV配置。")
    #     return

    # for frame_path in tqdm(processed_frames, desc="Creating video with OpenCV"):
    #     #  使用OpenCV读取16位PNG图像 (它会读取为BGR顺序)
    #     frame_16bit_bgr = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        
    #     if frame_16bit_bgr is None:
    #         print(f"警告：跳过无法读取的帧 {frame_path}")
    #         continue

    #     #  将16位数据转换为8位
    #     frame_8bit_bgr = (frame_16bit_bgr / 257.0).astype(np.uint8)
        
    #     #  将8位帧写入视频
    #     writer.write(frame_8bit_bgr)
            
    # # 释放writer对象，这是完成视频写入的关键步骤！
    # writer.release()
    # print("视频 'output_video.mp4' 创建成功！")

if __name__ == "__main__":
    main_batch()