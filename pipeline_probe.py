import numpy as np
import cv2
import os

class PipelineProbe:
    """
    ISP 管线探针模块 (Pipeline Probe)
    
    作用：无损截获数据流，打印统计信息，并保存原始矩阵和预览图，
    随后将数据原封不动地返回给流水线的下一环。
    """
    def __init__(self, probe_name: str, save_dir: str = "isp_debug_probes", 
                 save_npy: bool = True, save_preview: bool = True):
        """
        初始化探针。
        
        Args:
            probe_name: 探针的名称（比如 'after_demosaic', 'before_gamma'）
            save_dir: 调试数据的保存根目录
            save_npy: 是否保存未经任何处理的纯正 Float32 矩阵 (用于硬核数值分析)
            save_preview: 是否保存一张用于人眼查看的 PNG 预览图
        """
        self.probe_name = probe_name
        self.save_dir = os.path.join(save_dir, probe_name)
        self.save_npy = save_npy
        self.save_preview = save_preview
        
        # 确保输出目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 内部计数器，用于在处理视频序列时给帧编号
        self.frame_count = 0

    def execute(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        执行探针拦截。
        注意：它会 100% 原样返回 image_data，实现零干扰透传。
        """
        self.frame_count += 1
        file_prefix = f"frame_{self.frame_count:04d}"
        
        print(f"\n🔍 [探针报告] 正在拦截位置: {self.probe_name} (帧 {self.frame_count})")
        
        # ==========================================
        # 1. 打印 X光 级别的数值统计 (核心分析功能)
        # ==========================================
        dtype = image_data.dtype
        shape = image_data.shape
        data_min = np.min(image_data)
        data_max = np.max(image_data)
        data_mean = np.mean(image_data)
        
        # 检查是否有致命的 NaN (Not a Number) 或 Inf (无穷大)
        has_nan = np.isnan(image_data).any()
        has_inf = np.isinf(image_data).any()
        
        print(f"   ▶ 形状: {shape} | 类型: {dtype}")
        print(f"   ▶ 数值范围: Min={data_min:.4f}, Max={data_max:.4f}, Mean={data_mean:.4f}")
        if has_nan or has_inf:
            print(f"   ⚠️ [致命警告] 数据中包含 NaN: {has_nan}, Inf: {has_inf}！")

        # ==========================================
        # 2. 保存纯粹的数学矩阵 (.npy 格式)
        # ==========================================
        # 这是 rPPG 分析最宝贵的资产，你可以用 Python 或 MATLAB 再次读取它
        if self.save_npy:
            npy_path = os.path.join(self.save_dir, f"{file_prefix}_raw.npy")
            np.save(npy_path, image_data)
            print(f"   💾 原始矩阵已保存至: {npy_path}")

        # ==========================================
        # 3. 妥协的预览图保存 (.png 格式)
        # ==========================================
        # 注意：人眼看不了 Float32 里的负数和 HDR 高光，所以这里必须强行 Clip。
        # 但我们只改变要保存的这部分副本，【绝不改变】透传下去的本体！
        if self.save_preview:
            preview_path = os.path.join(self.save_dir, f"{file_prefix}_preview.png")
            
            # 复制一份用于造预览图
            vis_img = image_data.copy()
            
            # 如果是单通道 (Bayer RAW)，归一化后保存为灰度图
            if len(shape) == 2:
                # RAW 数据可能在 0-65535，也可能已经归一化，自适应处理
                if dtype == np.uint16 or data_max > 2.0:
                    vis_img = vis_img.astype(np.float32) / 65535.0
                vis_uint8 = np.clip(np.round(vis_img * 255.0), 0, 255).astype(np.uint8)
                cv2.imwrite(preview_path, vis_uint8)
                
            # 如果是三通道 (RGB / YUV)
            elif len(shape) == 3 and shape[2] == 3:
                # 简单粗暴地切除 HDR，仅作预览
                vis_clipped = np.clip(vis_img, 0.0, 1.0)
                vis_uint8 = np.clip(np.round(vis_clipped * 255.0), 0, 255).astype(np.uint8)
                
                # ⚠️ 简单的探针无法知道当前色彩空间是 RGB 还是 YUV。
                # 统一当作 RGB 转 BGR 处理，如果你探针插在 YUV 域，预览图颜色会很诡异（偏绿偏紫），这是正常的！
                vis_bgr = cv2.cvtColor(vis_uint8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(preview_path, vis_bgr)
            
            print(f"   🖼️ 预览图像已保存至: {preview_path}")

        print("   ✅ 拦截完成，数据已无损放行往下游传递。\n")
        
        # ==========================================
        # 4. 绝对透传，不改变一丝一毫的属性
        # ==========================================
        return image_data