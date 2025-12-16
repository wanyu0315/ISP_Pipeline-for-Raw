import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_image_channels(image_path):
    """
    加载图像，详细分析其三个通道，并生成可视化结果。

    参数:
    image_path (str): 图像文件的路径。
    """
    # --- 0. 加载图像 ---
    if not os.path.exists(image_path):
        print(f"错误: 找不到文件 '{image_path}'")
        return

    # 使用OpenCV读取图像，它默认以BGR顺序加载
    img = cv2.imread(image_path)

    if img is None:
        print(f"错误: OpenCV无法读取图像文件 '{image_path}'。文件可能已损坏或格式不支持。")
        return

    print(f"成功加载图像: {os.path.basename(image_path)}\n")

    # --- 1. 图像基本属性 ---
    print("--- 1. 图像基本属性 ---")
    height, width, channels = img.shape
    print(f"尺寸: {width}x{height}")
    print(f"通道数: {channels}")

    if channels != 3:
        print("结论: 这是一个单通道（或非三通道）图像，无法进行颜色通道分析。")
        return
    else:
        print("结论: 这是一个三通道图像。\n")

    # --- 2. 各通道数值分析 ---
    print("--- 2. 各通道数值分析 ---")
    # OpenCV中通道顺序是 B, G, R
    b_channel, g_channel, r_channel = cv2.split(img)
    
    # 定义通道信息
    channel_data = {
        "蓝色通道 (B)": b_channel,
        "绿色通道 (G)": g_channel,
        "红色通道 (R)": r_channel
    }

    # 计算并打印每个通道的统计数据
    stats = {}
    for name, channel in channel_data.items():
        min_val = np.min(channel)
        max_val = np.max(channel)
        avg_val = np.mean(channel)
        stats[name] = {'avg': avg_val, 'max': max_val}
        print(f"{name} -> 最小值: {min_val:<4} | 最大值: {max_val:<4} | 平均值: {avg_val:.2f}")

    # 动态生成分析结论
    # 找出平均值最高的通道
    dominant_channel = max(stats, key=lambda k: stats[k]['avg'])
    dominant_channel_name = dominant_channel.split(" ")[0] # "蓝色", "绿色", or "红色"
    
    analysis_text = f"分析: {dominant_channel_name}通道的平均值({stats[dominant_channel]['avg']:.2f})显著高于其他通道，这表明图像的整体色调可能偏{dominant_channel_name}。"
    print(f"\n{analysis_text}\n")

    # --- 2.5. 随机像素点抽样 ---
    print("--- 2.5. 随机像素点抽样 ---")
    num_samples = 10 # 随机选取10个点
    
    # 随机生成坐标 (y坐标, x坐标)
    random_y_coords = np.random.randint(0, height, num_samples)
    random_x_coords = np.random.randint(0, width, num_samples)

    print(f"随机选取 {num_samples} 个像素点进行B, G, R值分析：")
    for i in range(num_samples):
        y = random_y_coords[i]
        x = random_x_coords[i]
        
        # img[y, x] 直接返回 [B, G, R] 值
        pixel_value = img[y, x]
        
        # 使用:<3进行左对齐，使输出更整洁
        print(f"  坐标 (x={x:<4}, y={y:<4}) -> B: {pixel_value[0]:<3} | G: {pixel_value[1]:<3} | R: {pixel_value[2]:<3}")
    
    print("") # 添加一个换行


    # --- 3. 可视化显示 ---
    print("--- 3. 可视化显示 ---")
    print("正在生成可视化图像...")
    
    # 创建一个2x2的图表布局
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Matplotlib期望RGB格式，而OpenCV是BGR，所以需要转换
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 0,0: 显示原始图像
    axs[0, 0].imshow(img_rgb)
    axs[0, 0].set_title('原始图像 (RGB)')
    axs[0, 0].axis('off')

    # 0,1: 显示红色通道
    axs[0, 1].imshow(r_channel, cmap='Reds')
    axs[0, 1].set_title('红色通道 (R)')
    axs[0, 1].axis('off')

    # 1,0: 显示绿色通道
    axs[1, 0].imshow(g_channel, cmap='Greens')
    axs[1, 0].set_title('绿色通道 (G)')
    axs[1, 0].axis('off')

    # 1,1: 显示蓝色通道
    axs[1, 1].imshow(b_channel, cmap='Blues')
    axs[1, 1].set_title('蓝色通道 (B)')
    axs[1, 1].axis('off')
    
    # 调整布局并保存
    plt.tight_layout()
    output_filename = 'channel_analysis.png'
    plt.savefig(output_filename)
    print(f"可视化结果已保存为 {output_filename}")
    # plt.show() # 如果需要，可以取消注释以直接显示图像


# --- 进行分析 ---
# 请确保路径正确，并使用'r'前缀来处理反斜杠
analyze_image_channels(r"D:\Project File\VSCode\ISP_processing_for_rppg\Check_raw\demosaic_raw16.png")