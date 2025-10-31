# main_batch_raw.py (已修改以适配混合版 Demosaic)

import imageio
import numpy as np
import os
import glob
from tqdm import tqdm
import cv2
import subprocess # 导入subprocess模块

# 导入我们的ISP管道和模块
from isp_pipeline import ISPPipeline
from demosaic import Demosaic  # 导入我们最终的混合版本
from white_balance import WhiteBalance
from gamma_correction import GammaCorrection

def main_batch():
    # --- 1. 定义传感器/图像的元数据 ---
    # !! 必须为无头RAW文件提供元数据 !!
    IMAGE_WIDTH = 1280
    IMAGE_HEIGHT = 800
    IMAGE_DTYPE = np.uint16  # 或 np.uint8, 取决于您的RAW数据位深
    BAYER_PATTERN = 'GRBG'   # 根据传感器规格设置

    # --- 2. 定义输入和输出文件夹 ---
    input_folder = 'ISPpipline/integrated_test_raw16_2raw' # 存放RAW序列的文件夹
    output_folder = 'ISPpipline/isp_processed_frames_raw16_2raw/'   # 存放处理后PNG帧的文件夹
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- 3. 获取所有RAW文件 ---
    # <--- 关键改动: 文件扩展名从.dng改为.raw，以匹配无头RAW文件的场景
    raw_files = sorted(glob.glob(os.path.join(input_folder, '*.raw')))
    
    if not raw_files:
        print(f"在文件夹 '{input_folder}' 中没有找到 .raw 文件。")
        return

    print(f"找到 {len(raw_files)} 个 .raw 文件进行处理。")

    # --- 4. 实例化并组装ISP管道 ---
    # <--- 关键改动: 在创建Demosaic实例时，必须传入元数据
    demosaic_module = Demosaic(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        bayer_pattern=BAYER_PATTERN,
        dtype=IMAGE_DTYPE
    )
    
    my_isp = ISPPipeline(modules=[
        demosaic_module,
        WhiteBalance(),
        GammaCorrection()
    ])

    # --- 5. 定义处理参数 (所有帧使用相同参数) ---
    processing_params = {
        'demosaic': {'algorithm': 'CV'},
        'whitebalance': {'algorithm': 'gray_world'},
        'gammacorrection': {'gamma': 2.2}
    }

    # --- 6. 循环处理所有文件，并保存为序列表文件名 ---

    skip_processing = False
    # 检查输出文件夹是否存在
    if os.path.isdir(output_folder):
        # 如果存在，检查里面是否已有处理好的png文件
        # 使用 glob 查找符合命名规则的文件，更精确
        existing_frames = glob.glob(os.path.join(output_folder, 'frame_*.png'))
        if existing_frames:
            print(f"✅ 输出文件夹 '{output_folder}' 已存在且包含 {len(existing_frames)} 帧，将跳过ISP处理步骤。")
            skip_processing = True
            # 为后续视频合成步骤准备好 padding 和 total_files 变量
            total_files = len(existing_frames)
            padding = len(str(total_files)) # 根据文件数计算padding
        else:
            print(f"ℹ️ 输出文件夹 '{output_folder}' 已存在但为空，将开始处理RAW文件。")
    else:
        print(f"ℹ️ 输出文件夹 '{output_folder}' 不存在，将创建并开始处理RAW文件。")
        os.makedirs(output_folder, exist_ok=True) # 创建文件夹

    # --- 如果不需要跳过，则执行处理循环 ---
    if not skip_processing:
        print("\n🚀 开始执行ISP处理流程...")
        # 在循环开始前，获取文件总数以确定命名格式的宽度
        try:
            total_files = len(raw_files)
            # 计算补零的位数，例如 total_files=800 -> padding=3; total_files=1234 -> padding=4
            padding = len(str(total_files)) 
        except (NameError, TypeError):
            print("错误：'raw_files' 列表不存在或为空。请确保在此代码块之前已定义 'raw_files'。")
            raw_files = [] 
            padding = 4 # 设置一个默认值

        # 初始化帧计数器
        frame_counter = 0
        for raw_file_path in tqdm(raw_files, desc="Processing RAW sequence"):
            try:
                # 运行管道
                final_image = my_isp.process(raw_file_path, params=processing_params)
                
                # 使用计数器生成新的序列化文件名
                new_file_name = f"frame_{frame_counter:0{padding}d}.png"
                output_path = os.path.join(output_folder, new_file_name)
                
                # 保存为16位PNG
                cv2.imwrite(output_path, final_image)

                frame_counter += 1  # 增加计数器

            except Exception as e:
                print(f"处理文件 {raw_file_path} 时出错: {e}")
                continue

        print(f"\n✅ 所有帧处理完毕，已保存至 '{output_folder}' 文件夹，并已重命名为序列格式。") 

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

    output_video_path = 'output_video_8bit_yuv_lossless__raw16_2raw.mkv'
    framerate = 30.0

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
        '-c:v', 'ffv1',  # 编码器（ffv1，libx264）
        '-level', '3',
        '-pix_fmt', 'yuv420p',  # 像素格式(bgr48le、bgr24、yuv420p),注意需要和上面处理后的视频帧通道格式对应，OpenCV是BGR格式
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