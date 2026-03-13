# Float32 零损耗高动态范围图像信号处理器 (Zero-Loss HDR ISP Pipeline)

**专为远程光电容积脉搏波描记法 (rPPG) 与高精度计算摄影设计的科研级 ISP 管线**

------

## Ⅰ. 项目概述 (Overview)

本项目实现了一条高度模块化、纯 Python 驱动的图像信号处理 (Image Signal Processor, ISP) 管线。与传统的工业界硬件 ISP 或常规计算机视觉图像处理流程不同，本架构的核心设计哲学是**“全浮点、零损耗、延迟截断” (Full-Float32, Zero-Loss, Late-Clipping)**。

在微弱生理信号提取（如 rPPG）任务中，由血容量变化引起的肤色波动（AC分量）通常仅占整体环境光强（DC分量）的千分之一甚至万分之一。传统的 8-bit 或常规 16-bit 整数 ISP 管线会在色彩矩阵转换 (CCM)、锐化下冲 (Overshoot/Undershoot) 和降噪滤波等环节频繁执行强制的边界截断 (`clip(0, 1)`) 和向下取整操作。这些操作会在物理和数学层面上直接抹杀微弱的交流信号，并引发非线性的色相扭曲。

本 ISP 管线通过在入口处将 RAW 数据提升至 32 位浮点空间，并**允许数据流中合法存在代表超高光的 $>1.0$ 数值以及代表锐化下冲或极度饱和的 $<0.0$ 数值**，彻底确保了整个计算图（Computational Graph）中的能量守恒与光学线性，直至最终导出阶段才进行唯一一次全局量化清算。

------

## Ⅱ. 核心设计哲学 (Core Philosophy)

1. **绝对浮点漫游 (Absolute Float32 Roaming)**

   从 `RawLoader` 开始，所有像素阵列均被强制转换为 `numpy.float32`。任何中间模块不得擅自将其降级为 `uint8` 或 `uint16`（除非调用 OpenCV 极个别底层 C++ 强绑定 8-bit 的算法，如 NLM/CLAHE，且会触发严厉的终端警告）。

2. **禁止早产截断 (No Premature Clipping)**

   在图像进入最终的 `YUVtoRGB.convert_for_display()` 出口之前，坚决禁止 `np.clip(..., 0, 1)` 操作。保留高光端与阴影端的极值，为后续的时域滤波和曝光补偿留出完美的线性计算空间。

3. **四舍五入保护 (Rounding Protection)**

   在最终落地为整数格式保存时，强制使用 `np.clip(np.round(img * scale), ...)`，以消除 `astype(int)` 向下取整造成的系统性负向截断误差。

------

## Ⅲ. 管线模块架构 (Pipeline Architecture)

数据流严格按照现代相机物理光学捕获的逆过程进行处理。所有模块均通过高度解耦的类实现，统一暴露 `execute(self, data, **kwargs)` 接口。

### 1. RAW 域处理 (RAW Domain Processing)

- **`RawLoader` (RAW 数据加载器)**: 解析底层物理拜耳 (Bayer) 阵列，执行初始位深提权（提升至 Float32 空间）。
- **`BlackLevelCorrection` (黑电平校正)**: 扣减传感器暗电流底座 (Pedestal/OB)，还原绝对光子计数。
- **`DefectPixelCorrection` (坏点校正)**: 引入预标定的 `defect_map.npy`，采用中值替换策略修复传感器物理死像素，防止时域静态坏点在 rPPG 提取中化为高频噪声。
- **`WhiteBalanceRaw` (RAW 白平衡)**: 基于灰度世界 (Gray World) 等算法对 R 和 B 通道施加线性增益，配平光源色温。

### 2. RGB 域处理 (RGB Domain Processing)

- **`Demosaic` (去马赛克)**: 将 1 通道 Bayer 阵列插值为 3 通道 RGB 图像。集成 `colour-demosaicing` (纯净双线性插值) 与 `rawpy` (AHD等高级算法)，通过 16-bit 高精度桥接技术实现无损流转。
- **`ColorCorrectionMatrix` (色彩校正矩阵)**: 将相机传感器色彩空间映射至标准 sRGB。内置针对 rPPG 研发的特殊矩阵（如 `rppg_green_isolation`）。**此模块坚决不执行截断，允许负数与高光溢出。**
- **`GammaCorrection` (伽马校正)**: 提供纯线性 (Gamma=1.0) 透传与标准 sRGB 曲线映射。内建底线防御机制 `np.clip(img, 0.0, None)`，有效防止 CCM 产生的负数导致指数运算崩溃 (NaN)。

### 3. YUV 域处理 (YUV Domain Processing)

- **`ColorSpaceConversion` (RGB 转 YUV)**: 使用高精度浮点矩阵转换。解开 HDR 紧箍咒，允许 Y 分量 $>1.0$ 以及 U/V 分量超出常规色差边界。
- **`Denoise` (空域降噪)**: 亮度/色度分离降噪。原生支持浮点级高斯 (Gaussian)、双边 (Bilateral)、小波 (Wavelet) 与各向异性扩散 (Anisotropic Diffusion)。
- **`Sharpen` (锐化增强)**: 仅对亮度 (Y) 通道作用。**绝对保留边缘振荡产生的上下冲 (Overshoot/Undershoot)**，防止非对称截断引起的高光区域色相扭曲。
- **`ContrastSaturation` (对比度与饱和度)**: 线性缩放与 S 曲线调节。

### 4. 终极出口清算 (Exit Cleansing)

- **`YUVtoRGB`**: 负责将 YUV 浮点还原为 RGB。其核心方法 `convert_for_display` 是整条管线的“海关”。它在此时才执行全管线唯一的一次全局 `clip(0.0, 1.0)`，并根据配置的 `OUTPUT_BIT_DEPTH` (8 或 16) 执行最严谨的放大量化与四舍五入，最终交付给底层系统保存。

------

## Ⅳ. 高级探针系统：Pipeline Probe

在极其复杂的非线性管线中排查数值溢出或物理模型失效是一项艰巨的任务。为此，系统内建了科研级观测工具 `PipelineProbe`。

### 功能特性

通过将 `PipelineProbe` 实例化并插入到 `ISPPipeline` 的模块列表中的任意节点，可以实现：

1. **零损耗透传 (Zero-Loss Passthrough)**：作为透明中间件，不仅不改变数据本身的任何形态，连内存引用都保持一致。
2. **X光级数值监控**：在终端实时打印当前流经张量的 `Shape`、`Min`、`Max`、`Mean` 以及致命的 `NaN/Inf` 警报。
3. **Numpy 内存快照 (`.npy` Export)**：将该节点的纯净 Float32 张量落盘，供 MATLAB 或 Python 后续精确绘制特定像素的时域/空域演化曲线，极大助力 rPPG 信号在复杂网络中的追踪分析。
4. **安全预览图生成**：自动对截获的数据进行旁路降维并保存为可视化 PNG 文件，便于人类直观检查模块的宏观效果。

------

## Ⅴ. 视频合成与输出规范

批量处理脚本 (`main_batch_raw.py`) 提供了一套完整的时域重组方案：

- **坏帧自适应剔除**：基于输出位深动态调整黑帧判定阈值，自动丢弃或替换受损的 RAW 帧。
- **无损视频编码 (FFV1)**：摒弃 H.264 等高损耗帧间压缩编码器。强制使用 `FFV1` 无损编码器，GOP 设定为 1（全 I 帧），彻底关闭时域运动补偿预测，以绝对忠实于原始的 rPPG 逐帧时序波动。
- **16-bit 深色域支持**：原生支持 `bgr48le` (16-bit) 和 `bgr0` (8-bit) 像素格式的自适应切换，将 ISP 极力保护的极值变化封装进 MKV 容器。

------

## Ⅵ. 运行与使用方法 (Usage)

### 环境依赖

Bash

```
pip install numpy opencv-python imageio scipy tqdm tifffile rawpy colour-demosaicing PyWavelets
```

*注：视频合成依赖于系统环境中正确安装的 FFmpeg 工具。*

### 快速启动

在 `main_batch_raw.py` 中配置核心参数后即可执行：

Python

```
# 1. 设定输出位深度 (科研强烈建议 16-bit)
OUTPUT_BIT_DEPTH = 16 

# 2. 修改输入输出路径
ROOT_INPUT_DIR = 'your/raw/data/path'
ROOT_OUTPUT_VIDEO_DIR = 'your/output/video/path'

# 3. 运行批处理脚本
python main_batch_raw.py
```

### 探针 (Probe) 部署示例
在组装 `my_isp` 时，可任意穿插探针以监控数据：

Python

```
from pipeline_probe import PipelineProbe

my_isp = ISPPipeline(modules=[
    loader_module,
    BlackLevelCorrection(),
    PipelineProbe(probe_name="1_after_blc", save_npy=True), # 监控黑电平扣除后数据
    WhiteBalanceRaw(),
    demosaic_module,
    ColorCorrectionMatrix(),
    PipelineProbe(probe_name="2_after_ccm", save_npy=True), # 监控 CCM 后的 HDR 极值
    # ... 后续模块
])
```

