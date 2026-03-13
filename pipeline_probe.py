import numpy as np
import cv2
import os
import csv

class PipelineProbe:
    """
    高级 ISP 管线探针模块 (集成自动人脸 ROI 追踪与区间录制)
    
    除了无损透传和保存图像，本探针支持：
    1. 追踪特定 ROI 区域的通道均值 (直接模拟 rPPG 提取)
    2. 计算帧间差分 (AC Delta)，监控信号是否被抹平
    3. 将所有时序数据写入 CSV，便于科研绘图分析
    4. 灵活的区间录制 (防止长视频导致输出文件爆炸)
    """
    
    def __init__(self, probe_name: str, save_dir: str = "isp_debug_probes",
                 save_npy: bool = False, save_preview: bool = True,
                 auto_detect_roi: bool = True, fallback_roi: tuple = None,
                 start_frame: int = 1000, max_frames: int = 50,
                 max_csv_frames: int = 1000, max_preview_frames: int = 1000):
        """
        初始化探针参数
        Args:
            probe_name: 探针名称
            save_dir: 根目录
            save_npy: 是否保存全尺寸浮点矩阵 (极占硬盘，科研需要精确数值时才开启)
            save_preview: 是否保存带有 ROI 标记的预览图
            auto_detect_roi: 是否自动检测人脸并提取皮肤区域作为 ROI
            fallback_roi: 如果自动检测失败，使用的备用 ROI (y1, y2, x1, x2)
            start_frame: 从视频的第几帧开始录制探测数据
            max_frames: CSV 和图像的默认最大录制帧数 (为 None 则录制到视频结束)
            max_csv_frames: CSV 最大录制帧数，覆盖 max_frames (为 None 则使用 max_frames)
            max_preview_frames: 图像最大保存帧数，覆盖 max_frames (为 None 则使用 max_frames)
        """
        self.probe_name = probe_name
        self.save_dir = os.path.join(save_dir, probe_name)
        self.save_npy = save_npy
        self.save_preview = save_preview
        
        self.auto_detect_roi = auto_detect_roi
        self.fallback_roi = fallback_roi
        self.current_roi = fallback_roi  
        
        # 🌟 录制区间控制
        self.start_frame = start_frame
        self.max_frames = max_frames
        self.max_csv_frames = max_csv_frames if max_csv_frames is not None else max_frames
        self.max_preview_frames = max_preview_frames if max_preview_frames is not None else max_frames
        
        # 初始化人脸检测器 (OpenCV 自带的轻量级检测器)
        if self.auto_detect_roi:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                print("   [!] 警告: 无法加载 OpenCV 人脸检测模型，将降级为无 ROI 模式。")
                self.auto_detect_roi = False
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.global_frame_count = 0   # 实际流过的总帧数
        self.recorded_csv_count = 0   # 实际记录到 CSV 的帧数
        self.recorded_preview_count = 0  # 实际保存图像的帧数
        self.prev_frame = None  
        
        # 初始化 CSV 文件
        self.csv_path = os.path.join(self.save_dir, f"{probe_name}_timeseries.csv")
        self._init_csv()

    def _init_csv(self):
        """初始化 CSV 表头"""
        headers = ['Frame_ID', 'Global_Mean', 'Global_AC_Delta']
        if self.auto_detect_roi or self.fallback_roi:
            headers.extend(['ROI_Mean_C0', 'ROI_Mean_C1', 'ROI_Mean_C2', 'ROI_AC_Delta'])
            
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _detect_face_roi(self, image_data: np.ndarray) -> tuple:
        """从任意色彩空间的 float32 图像中安全提取人脸 ROI"""
        # 1. 安全降级为 8-bit 灰度图供检测使用 (绝对不改变传入的原数据)
        safe_img = np.nan_to_num(image_data, nan=0.0, posinf=1.0, neginf=0.0)
        safe_img = np.clip(safe_img, 0.0, 1.0)
        
        if len(safe_img.shape) == 3:
            gray_float = np.mean(safe_img, axis=2)
        else:
            gray_float = safe_img
            
        gray_uint8 = (gray_float * 255.0).astype(np.uint8)
        
        # 2. 执行人脸检测
        faces = self.face_cascade.detectMultiScale(gray_uint8, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        if len(faces) > 0:
            # 取面积最大的人脸
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]
            
            # 3. 大面积 rPPG 区域
            y1 = y + int(h * 0.05)
            y2 = y + int(h * 0.60) 
            x1 = x + int(w * 0.15)
            x2 = x + w - int(w * 0.15)

            # 确保 ROI 有效
            if y2 > y1 and x2 > x1:
                return (y1, y2, x1, x2)
                
        return None

    def execute(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        self.global_frame_count += 1
        
        # =========================================================
        # 1. 拦截放行判定
        # =========================================================
        # 如果还没到开始帧，直接放行
        if self.global_frame_count < self.start_frame:
            return image_data
            
        # 检查是否两种数据都已经录满
        csv_full = (self.max_csv_frames is not None) and (self.recorded_csv_count >= self.max_csv_frames)
        preview_full = (self.max_preview_frames is not None) and (self.recorded_preview_count >= self.max_preview_frames)
        
        # 如果 CSV 和图片都已经达到上限，探针进入彻底休眠，直接放行
        if csv_full and preview_full:
            return image_data

        file_prefix = f"frame_{self.global_frame_count:04d}"
        
        print(f"\n🔍 [探针] 拦截位置: {self.probe_name} (物理帧 {self.global_frame_count})")
        print(f"   📊 CSV录制进度: {self.recorded_csv_count}/{self.max_csv_frames} | 🖼️ 图片保存进度: {self.recorded_preview_count}/{self.max_preview_frames}")
        
        shape = image_data.shape
        is_3_channel = len(shape) == 3 and shape[2] == 3

        # =========================================================
        # 2. 动态 ROI 检测 (公用步骤)
        # =========================================================
        if self.auto_detect_roi:
            detected_roi = self._detect_face_roi(image_data)
            if detected_roi is not None:
                self.current_roi = detected_roi

        # =========================================================
        # 3. CSV 时序数据提取与写入 (独立控制)
        # =========================================================
        if not csv_full:
            csv_row = [self.global_frame_count]
            
            # 全局数据
            csv_row.append(f"{np.mean(image_data):.6f}")
            if self.prev_frame is not None:
                csv_row.append(f"{np.mean(np.abs(image_data - self.prev_frame)):.6f}")
            else:
                csv_row.append("0.000000")

            # ROI 数据
            if self.current_roi:
                y1, y2, x1, x2 = self.current_roi
                roi_data = image_data[y1:y2, x1:x2]
                
                if is_3_channel:
                    roi_means = np.mean(roi_data, axis=(0, 1))
                    csv_row.extend([f"{roi_means[0]:.6f}", f"{roi_means[1]:.6f}", f"{roi_means[2]:.6f}"])
                else:
                    csv_row.extend([f"{np.mean(roi_data):.6f}", "N/A", "N/A"])
                    
                if self.prev_frame is not None:
                    prev_roi = self.prev_frame[y1:y2, x1:x2]
                    csv_row.append(f"{np.mean(np.abs(roi_data - prev_roi)):.6f}")
                else:
                    csv_row.append("0.000000")
            else:
                # 如果没检测到 ROI，用 0 或 N/A 占位，保证 CSV 结构不乱
                csv_row.extend(["N/A", "N/A", "N/A", "0.000000"])

            # 写入 CSV 并更新计数器
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_row)
            
            self.recorded_csv_count += 1
            self.prev_frame = image_data.copy()

        # =========================================================
        # 4. 图像预览与 Numpy 保存 (独立控制)
        # =========================================================
        if not preview_full:
            # NPY 保存 (和预览图共享配额限制，防止炸硬盘)
            if self.save_npy:
                np.save(os.path.join(self.save_dir, f"{file_prefix}_raw.npy"), image_data)

            # PNG 保存
            if self.save_preview:
                preview_path = os.path.join(self.save_dir, f"{file_prefix}_preview.png")
                vis_img = image_data.copy()
                
                if len(shape) == 2:
                    if vis_img.dtype == np.uint16 or np.max(vis_img) > 2.0:
                        vis_img = vis_img.astype(np.float32) / 65535.0
                    vis_uint8 = np.clip(np.round(vis_img * 255.0), 0, 255).astype(np.uint8)
                    vis_bgr = cv2.cvtColor(vis_uint8, cv2.COLOR_GRAY2BGR) 
                else:
                    vis_clipped = np.clip(vis_img, 0.0, 1.0)
                    vis_uint8 = np.clip(np.round(vis_clipped * 255.0), 0, 255).astype(np.uint8)
                    vis_bgr = cv2.cvtColor(vis_uint8, cv2.COLOR_RGB2BGR)
                
                if self.current_roi:
                    y1, y2, x1, x2 = self.current_roi
                    cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                cv2.imwrite(preview_path, vis_bgr)
            
            self.recorded_preview_count += 1
            
        print(f"   ✅ 数据放行. (ROI: {'追踪中' if self.current_roi else '未找到'})")
        
        return image_data
