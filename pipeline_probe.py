import numpy as np
import cv2
import os
import csv
import mediapipe as mp

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
                 start_frame: int = 1000, max_frames: int = 20,
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
        
        # 初始化人脸检测器：MediaPipe Face Mesh（主）+ Haar Cascade（兜底）
        if self.auto_detect_roi:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            # 皮肤掩码：面部轮廓地标索引
            self._FACE_OVAL  = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,
                                 379,378,400,377,152,148,176,149,150,136,172,58,132,93,
                                 234,127,162,21,54,103,67,109]
            self._EXCL_ZONES = [
                [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],   # 左眼
                [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398], # 右眼
                [70,63,105,66,107,55,65,52,53,46],                                 # 左眉
                [300,293,334,296,336,285,295,282,283,276],                         # 右眉
                [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95], # 嘴唇
            ]
            self._skin_mask = None   # 缓存当前帧皮肤掩码
            if self.face_cascade.empty():
                print("   [!] 警告: 无法加载 OpenCV Haar Cascade，兜底检测不可用。")
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.global_frame_count = 0   # 实际流过的总帧数
        self.recorded_csv_count = 0   # 实际记录到 CSV 的帧数
        self.recorded_preview_count = 0  # 实际保存图像的帧数
        self.prev_frame = None
        self._prev_skin_mask = None  # 上帧皮肤掩码，用于 AC Delta 对齐
        
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

    def _build_skin_mask(self, image_data: np.ndarray) -> np.ndarray:
        """
        用 MediaPipe Face Mesh 构建动态皮肤掩码。
        返回 uint8 掩码（255=皮肤，0=非皮肤），检测失败返回 None。
        兼容 RAW（单通道）、RGB、YUV 三种域的 float32 输入。
        """
        # 安全归一化为 uint8 RGB（兼容所有域）
        img = np.nan_to_num(image_data, nan=0.0, posinf=1.0, neginf=0.0)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        lo, hi = img.min(), img.max()
        img_norm = (img - lo) / (hi - lo + 1e-6)
        rgb_uint8 = (img_norm * 255.0).clip(0, 255).astype(np.uint8)

        H, W = image_data.shape[:2]
        result = self._mp_face_mesh.process(rgb_uint8)

        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark

            def _pts(indices):
                return np.array([[int(lms[i].x * W), int(lms[i].y * H)]
                                  for i in indices], dtype=np.int32)

            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [_pts(self._FACE_OVAL)], 255)
            for zone in self._EXCL_ZONES:
                hull = cv2.convexHull(_pts(zone))
                cv2.fillPoly(mask, [hull], 0)
            return mask if np.any(mask) else None

        # Haar Cascade 兜底：返回矩形掩码
        gray = rgb_uint8[:, :, 0]
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            y1 = y + int(h * 0.05);  y2 = y + int(h * 0.60)
            x1 = x + int(w * 0.15);  x2 = x + w - int(w * 0.15)
            if y2 > y1 and x2 > x1:
                mask = np.zeros((H, W), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                return mask
        return None

    def _detect_face_roi(self, image_data: np.ndarray) -> tuple:
        """保留兼容接口，返回皮肤掩码的 bounding box（供预览框绘制用）"""
        mask = self._build_skin_mask(image_data)
        if mask is None:
            return None
        self._skin_mask = mask
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return None
        return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))

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
        # 2. 动态皮肤掩码检测 (公用步骤)
        # =========================================================
        self._skin_mask = None  # 每帧重置
        if self.auto_detect_roi:
            new_mask = self._build_skin_mask(image_data)
            if new_mask is not None:
                # mask 面积变化 > 15% 才更新，抑制帧间抖动
                if self._prev_skin_mask is not None:
                    old_count = np.sum(self._prev_skin_mask > 0)
                    new_count = np.sum(new_mask > 0)
                    if old_count > 0 and abs(new_count - old_count) / old_count < 0.15:
                        self._skin_mask = self._prev_skin_mask  # 沿用上帧 mask
                    else:
                        self._skin_mask = new_mask
                else:
                    self._skin_mask = new_mask
            elif self._prev_skin_mask is not None:
                # 检测失败时沿用上帧 mask，避免退化到 Haar Cascade 产生尖刺
                self._skin_mask = self._prev_skin_mask

            if self._skin_mask is not None:
                ys, xs = np.where(self._skin_mask > 0)
                if len(ys) > 0:
                    self.current_roi = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))

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

            # ROI 数据：优先用皮肤掩码均值，无掩码则退化为矩形框均值
            if self._skin_mask is not None:
                mask = self._skin_mask
                if is_3_channel:
                    skin_pixels = image_data[mask > 0]  # shape: (N, 3)
                    roi_means = np.mean(skin_pixels, axis=0)
                    csv_row.extend([f"{roi_means[0]:.6f}", f"{roi_means[1]:.6f}", f"{roi_means[2]:.6f}"])
                else:
                    skin_pixels = image_data[mask > 0]
                    csv_row.extend([f"{np.mean(skin_pixels):.6f}", "N/A", "N/A"])

                # Bug 2 修复：AC Delta 用前后帧 mask 交集区域计算
                if self.prev_frame is not None and self._prev_skin_mask is not None:
                    common = (mask > 0) & (self._prev_skin_mask > 0)
                    if np.any(common):
                        csv_row.append(f"{np.mean(np.abs(image_data[common] - self.prev_frame[common])):.6f}")
                    else:
                        csv_row.append("0.000000")
                else:
                    csv_row.append("0.000000")

            elif self.current_roi:
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
                csv_row.extend(["N/A", "N/A", "N/A", "0.000000"])

            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_row)

            self.recorded_csv_count += 1

        # Bug 1 修复：prev_frame 每帧都更新，不受 csv_full 限制
        self.prev_frame = image_data.copy()
        self._prev_skin_mask = self._skin_mask

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
                
                if self._skin_mask is not None:
                    # 绿色半透明覆盖皮肤区域，非皮肤区域变暗
                    overlay = vis_bgr.copy()
                    overlay[self._skin_mask == 0] = (overlay[self._skin_mask == 0] * 0.3).astype(np.uint8)
                    green_layer = np.zeros_like(vis_bgr)
                    green_layer[self._skin_mask > 0] = (0, 80, 0)
                    vis_bgr = cv2.addWeighted(overlay, 1.0, green_layer, 0.4, 0)
                    skin_count = int(np.sum(self._skin_mask > 0))
                    cv2.putText(vis_bgr, f"Skin px: {skin_count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                elif self.current_roi:
                    y1, y2, x1, x2 = self.current_roi
                    cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                cv2.imwrite(preview_path, vis_bgr)
            
            self.recorded_preview_count += 1
            
        print(f"   ✅ 数据放行. (ROI: {'追踪中' if self.current_roi else '未找到'})")
        
        return image_data
