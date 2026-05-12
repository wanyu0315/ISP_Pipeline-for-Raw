import csv
import hashlib
import json
import os

import cv2
import mediapipe as mp
import numpy as np


class ProbeSkinContext:
    """
    跨 probe 共享的皮肤提取上下文。

    同一物理帧内只生成一次 reference RGB，只做一次 landmarks + skin mask 检测，
    之后由所有 probe 复用同一套 skin/valid mask，保证跨域可比。
    """

    RGB_LOW_TH = np.int32(20)
    RGB_HIGH_TH = np.int32(220)
    MAX_STALE_FRAMES = 3

    _BAYER_TO_RGB_CODE = {
        "RGGB": cv2.COLOR_BAYER_RGGB2RGB,
        "BGGR": cv2.COLOR_BAYER_BGGR2RGB,
        "GRBG": cv2.COLOR_BAYER_GRBG2RGB,
        "GBRG": cv2.COLOR_BAYER_GBRG2RGB,
    }

    # --- 与 pyVHR MagicLandmarks 对齐的排除区域 landmark 索引 ---
    # 旧版（更小的排除区域，保留以备回滚）：
    # _LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    # _RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    # _MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    #           61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    _LEFT_EYE = [
        157, 144, 145, 22, 23, 25, 154, 31, 160, 33, 46, 52,
        53, 55, 56, 189, 190, 63, 65, 66, 70, 221, 222, 223,
        225, 226, 228, 229, 230, 231, 232, 105, 233, 107, 243, 124,
    ]
    _RIGHT_EYE = [
        384, 385, 386, 259, 388, 261, 265, 398, 276, 282, 283, 285,
        413, 293, 296, 300, 441, 442, 445, 446, 449, 451, 334, 463,
        336, 464, 467, 339, 341, 342, 353, 381, 373, 249, 253, 255,
    ]
    _MOUTH = [
        391, 393, 11, 269, 270, 271, 287, 164, 165, 37, 167, 40,
        43, 181, 313, 314, 186, 57, 315, 61, 321, 73, 76, 335,
        83, 85, 90, 106,
    ]

    def __init__(self, rgb_low_th: int = 55, rgb_high_th: int = 200, max_stale_frames: int = 3):
        self.rgb_low_th = np.int32(rgb_low_th)
        self.rgb_high_th = np.int32(rgb_high_th)
        self.max_stale_frames = int(max_stale_frames)

        self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.frame_id = None
        self.reference_rgb = None
        self.skin_mask = None
        self.valid_skin_mask = None
        self.bbox = None
        self.mask_source = "none"
        self.skin_pixel_count = 0
        self.valid_pixel_count = 0
        self.frame_shape = None

        self._last_good_skin_mask = None
        self._last_good_valid_mask = None
        self._last_good_bbox = None
        self._last_good_shape = None
        self._stale_frame_count = 0

    @staticmethod
    def _mask_bbox(mask: np.ndarray):
        if mask is None or not np.any(mask > 0):
            return None
        ys, xs = np.where(mask > 0)
        return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))

    @staticmethod
    def _normalize_image_to_uint8_rgb(image_data: np.ndarray) -> np.ndarray:
        img = np.nan_to_num(image_data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)

        if np.issubdtype(image_data.dtype, np.integer):
            max_val = float(np.iinfo(image_data.dtype).max)
            if max_val <= 255.0:
                return np.clip(img, 0, 255).astype(np.uint8)
            img = img / max_val
            return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
        # 对 float32 统一使用固定的显示映射语义：
        # 仅将线性 RGB 的有效检测范围视为 [0, 1]，超范围值直接截断。
        # 不能按帧 min/max 归一化，否则 pyVHR 的 55/200 阈值会失去固定物理意义。
        img = np.clip(img, 0.0, 1.0)
        return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)

    def _raw_to_reference_rgb(self, image_data: np.ndarray, raw_bayer_pattern: str) -> np.ndarray:
        if raw_bayer_pattern is None:
            raise ValueError("Reference RAW preview requires raw_bayer_pattern.")
        code = self._BAYER_TO_RGB_CODE.get(raw_bayer_pattern.upper())
        if code is None:
            raise ValueError(f"Unsupported raw_bayer_pattern={raw_bayer_pattern}.")

        raw = np.nan_to_num(image_data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.clip(raw, 0.0, 1.0)
        raw16 = np.clip(np.round(raw * 65535.0), 0, 65535).astype(np.uint16)

        rgb16 = cv2.cvtColor(raw16, code)
        return np.clip(np.round(rgb16.astype(np.float32) / 65535.0 * 255.0), 0, 255).astype(np.uint8)

    @staticmethod
    def _yuv_to_rgb(yuv_image: np.ndarray, method: str) -> np.ndarray:
        yuv = np.nan_to_num(yuv_image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).copy()
        yuv[:, :, 1] -= 0.5
        yuv[:, :, 2] -= 0.5
        matrices = {
            "bt601": np.array([
                [1.0, 0.0, 1.402],
                [1.0, -0.344136, -0.714136],
                [1.0, 1.772, 0.0],
            ], dtype=np.float32),
            "bt709": np.array([
                [1.0, 0.0, 1.5748],
                [1.0, -0.187324, -0.468124],
                [1.0, 1.8556, 0.0],
            ], dtype=np.float32),
            "bt2020": np.array([
                [1.0, 0.0, 1.4746],
                [1.0, -0.164553, -0.571353],
                [1.0, 1.8814, 0.0],
            ], dtype=np.float32),
        }
        transform = matrices.get((method or "bt709").lower())
        if transform is None:
            raise ValueError(f"Unsupported yuv_color_method={method}. Expected one of {sorted(matrices)}.")
        return np.dot(yuv, transform.T)

    def build_reference_rgb(
        self,
        image_data: np.ndarray,
        frame_domain: str,
        raw_bayer_pattern: str = None,
        yuv_color_method: str = "bt709",
    ) -> np.ndarray:
        domain = (frame_domain or "auto").lower()
        if domain == "raw" or image_data.ndim == 2:
            return self._raw_to_reference_rgb(image_data, raw_bayer_pattern)
        if domain == "yuv":
            rgb = self._yuv_to_rgb(image_data, yuv_color_method)
            return self._normalize_image_to_uint8_rgb(rgb)
        return self._normalize_image_to_uint8_rgb(image_data)

    @staticmethod
    def _extract_points(landmarks, indices, width: int, height: int) -> np.ndarray:
        points = []
        for idx in indices:
            if idx >= len(landmarks):
                continue
            x = int(round(landmarks[idx].x * (width - 1)))
            y = int(round(landmarks[idx].y * (height - 1)))
            if 0 <= x < width and 0 <= y < height:
                points.append([x, y])
        return np.asarray(points, dtype=np.int32)

    @staticmethod
    def _fill_hull(points: np.ndarray, shape) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        if points is None or len(points) < 3:
            return mask
        hull = cv2.convexHull(points.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def _build_masks(self, rgb_uint8: np.ndarray):
        height, width = rgb_uint8.shape[:2]
        result = self._mp_face_mesh.process(rgb_uint8)
        if not result.multi_face_landmarks:
            return None, None, None

        lms = result.multi_face_landmarks[0].landmark
        all_points = self._extract_points(lms, range(len(lms)), width, height)
        face_mask = self._fill_hull(all_points, (height, width))
        if not np.any(face_mask > 0):
            return None, None, None

        left_eye_mask = self._fill_hull(self._extract_points(lms, self._LEFT_EYE, width, height), (height, width))
        right_eye_mask = self._fill_hull(self._extract_points(lms, self._RIGHT_EYE, width, height), (height, width))
        mouth_mask = self._fill_hull(self._extract_points(lms, self._MOUTH, width, height), (height, width))

        skin_mask = face_mask.copy()
        skin_mask[left_eye_mask > 0] = 0
        skin_mask[right_eye_mask > 0] = 0
        skin_mask[mouth_mask > 0] = 0
        if not np.any(skin_mask > 0):
            return None, None, None

        too_dark = np.all(rgb_uint8 <= self.rgb_low_th, axis=2)
        too_bright = np.all(rgb_uint8 >= self.rgb_high_th, axis=2)
        valid_mask = np.where((skin_mask > 0) & (~too_dark) & (~too_bright), 255, 0).astype(np.uint8)
        if not np.any(valid_mask > 0):
            return None, None, None

        bbox = self._mask_bbox(valid_mask)
        return skin_mask, valid_mask, bbox

    def prepare(
        self,
        frame_id: int,
        image_data: np.ndarray,
        frame_domain: str,
        raw_bayer_pattern: str = None,
        yuv_color_method: str = "bt709",
    ):
        shape = tuple(image_data.shape[:2])
        if self.frame_id == frame_id and self.frame_shape == shape:
            return

        self.frame_id = frame_id
        self.frame_shape = shape
        self.reference_rgb = self.build_reference_rgb(
            image_data,
            frame_domain,
            raw_bayer_pattern,
            yuv_color_method,
        )

        skin_mask, valid_skin_mask, bbox = self._build_masks(self.reference_rgb)
        if valid_skin_mask is not None:
            self.skin_mask = skin_mask
            self.valid_skin_mask = valid_skin_mask
            self.bbox = bbox
            self.mask_source = "reference_rgb"
            self.skin_pixel_count = int(np.sum(skin_mask > 0))
            self.valid_pixel_count = int(np.sum(valid_skin_mask > 0))

            self._last_good_skin_mask = skin_mask.copy()
            self._last_good_valid_mask = valid_skin_mask.copy()
            self._last_good_bbox = bbox
            self._last_good_shape = shape
            self._stale_frame_count = 0
            return

        if (
            self._last_good_valid_mask is not None
            and self._last_good_shape == shape
            and self._stale_frame_count < self.max_stale_frames
        ):
            self.skin_mask = self._last_good_skin_mask.copy()
            self.valid_skin_mask = self._last_good_valid_mask.copy()
            self.bbox = self._last_good_bbox
            self.mask_source = "stale_reference"
            self.skin_pixel_count = int(np.sum(self.skin_mask > 0))
            self.valid_pixel_count = int(np.sum(self.valid_skin_mask > 0))
            self._stale_frame_count += 1
            return

        self.skin_mask = None
        self.valid_skin_mask = None
        self.bbox = None
        self.mask_source = "none"
        self.skin_pixel_count = 0
        self.valid_pixel_count = 0


class RawInputReferenceStore:
    """
    持久化 RAW input reference：保存 RAW input ROI 时序与逐帧 valid mask。

    该缓存独立于普通探针输出目录，用于后续生成 RAW proxy-GT Reference。
    """

    TIMESERIES_HEADERS = [
        "Frame_ID", "Global_Mean", "ROI_Mean_C0", "ROI_Mean_C1", "ROI_Mean_C2",
        "RAW_R_Mean", "RAW_G1_Mean", "RAW_G2_Mean", "RAW_G_Mean", "RAW_B_Mean",
        "RAW_R_Count", "RAW_G1_Count", "RAW_G2_Count", "RAW_B_Count",
        "ROI_Valid_Pixel_Count", "ROI_Skin_Pixel_Count", "ROI_Mask_Source",
        "Mask_Hash", "Mask_File", "BBox_Y1", "BBox_Y2", "BBox_X1", "BBox_X2",
    ]
    MANIFEST_HEADERS = [
        "Frame_ID", "Mask_Index", "Mask_File", "Mask_Hash",
        "ROI_Valid_Pixel_Count", "ROI_Skin_Pixel_Count", "ROI_Mask_Source",
        "BBox_Y1", "BBox_Y2", "BBox_X1", "BBox_X2", "Frame_H", "Frame_W",
    ]

    def __init__(self, config: dict):
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.active = False
        self.skip_reason = None
        self.recorded_count = 0

        self.max_frames = config.get("max_frames")
        self.max_frames = None if self.max_frames is None else int(self.max_frames)
        self.save_masks = bool(config.get("save_masks", True))
        self.overwrite = bool(config.get("overwrite", False))

        self.video_name = str(config.get("video_name", "unknown_video"))
        self.save_root = config.get("save_dir", "probe_analysis/raw_input_reference")
        self.metadata = dict(config.get("metadata", {}))
        self.metadata.update({
            "video_name": self.video_name,
            "cache_type": "raw_input_reference",
            "save_masks": self.save_masks,
            "max_frames": self.max_frames,
        })

        if not self.enabled:
            self.skip_reason = "disabled"
            return

        profile_name = config.get("profile_name")
        if not profile_name:
            profile_name = self._build_profile_name(self.metadata)

        self.cache_dir = os.path.join(self.save_root, self._safe_name(self.video_name), profile_name)
        self.masks_dir = os.path.join(self.cache_dir, "masks")
        self.timeseries_path = os.path.join(self.cache_dir, "raw_input_timeseries.csv")
        self.manifest_path = os.path.join(self.cache_dir, "mask_manifest.csv")
        self.metadata_path = os.path.join(self.cache_dir, "reference_meta.json")

        if (os.path.exists(self.timeseries_path) or os.path.exists(self.manifest_path)) and not self.overwrite:
            self.skip_reason = f"existing_cache:{self.cache_dir}"
            print(f"   ℹ️ RAW input reference 已存在，跳过写入: {self.cache_dir}")
            return

        os.makedirs(self.masks_dir, exist_ok=True)
        self._init_csv(self.timeseries_path, self.TIMESERIES_HEADERS)
        self._init_csv(self.manifest_path, self.MANIFEST_HEADERS)
        self._write_metadata()
        self.active = True
        print(f"   💾 RAW input reference 将写入: {self.cache_dir}")

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))

    @classmethod
    def _build_profile_name(cls, metadata: dict) -> str:
        key = {
            "start_frame": metadata.get("start_frame"),
            "max_frames": metadata.get("max_frames"),
            "raw_bayer_pattern": metadata.get("raw_bayer_pattern"),
            "auto_detect_roi": metadata.get("auto_detect_roi"),
            "fallback_roi": metadata.get("fallback_roi"),
            "rgb_low_th": metadata.get("rgb_low_th"),
            "rgb_high_th": metadata.get("rgb_high_th"),
            "max_mask_stale_frames": metadata.get("max_mask_stale_frames"),
            "sensor_width": metadata.get("sensor_width"),
            "sensor_height": metadata.get("sensor_height"),
        }
        digest = hashlib.sha1(json.dumps(key, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:10]
        start_frame = int(metadata.get("start_frame") or 0)
        max_frames = metadata.get("max_frames")
        frame_label = "all" if max_frames is None else str(int(max_frames))
        pattern = cls._safe_name(metadata.get("raw_bayer_pattern") or "raw")
        return f"start_{start_frame:06d}_frames_{frame_label}_bayer_{pattern}_roi_{digest}"

    @staticmethod
    def _init_csv(path: str, headers: list):
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _write_metadata(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        self.metadata["cache_dir"] = self.cache_dir
        self.metadata["timeseries_file"] = os.path.basename(self.timeseries_path)
        self.metadata["mask_manifest_file"] = os.path.basename(self.manifest_path)
        with open(self.metadata_path, mode="w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def is_full(self) -> bool:
        if not self.active:
            return True
        return self.max_frames is not None and self.recorded_count >= self.max_frames

    @staticmethod
    def _fmt_value(value, default: str = "N/A") -> str:
        if value is None:
            return default
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return default
        return f"{float(value):.6f}"

    @staticmethod
    def _bbox_values(bbox):
        if bbox is None:
            return ["N/A", "N/A", "N/A", "N/A"]
        y1, y2, x1, x2 = bbox
        return [str(int(y1)), str(int(y2)), str(int(x1)), str(int(x2))]

    def record(
        self,
        frame_id: int,
        image_data: np.ndarray,
        effective_mask: np.ndarray,
        raw_stats: dict,
        skin_pixel_count: int,
        valid_pixel_count: int,
        mask_source: str,
        bbox,
    ):
        if self.is_full():
            return

        frame_h, frame_w = image_data.shape[:2]
        mask_file = "N/A"
        mask_hash = "N/A"
        if effective_mask is not None and np.any(effective_mask > 0):
            mask_u8 = (effective_mask > 0).astype(np.uint8)
            mask_hash = hashlib.sha256(mask_u8.tobytes()).hexdigest()
            if self.save_masks:
                mask_file = os.path.join("masks", f"mask_{int(frame_id):06d}_{mask_hash[:12]}.npz")
                np.savez_compressed(
                    os.path.join(self.cache_dir, mask_file),
                    valid_skin_mask=mask_u8,
                    frame_id=np.array([int(frame_id)], dtype=np.int64),
                    frame_shape=np.array([frame_h, frame_w], dtype=np.int64),
                    mask_hash=np.array([mask_hash]),
                )

        if raw_stats is None:
            raw_stats = {
                "RAW_R_Mean": np.nan,
                "RAW_G1_Mean": np.nan,
                "RAW_G2_Mean": np.nan,
                "RAW_G_Mean": np.nan,
                "RAW_B_Mean": np.nan,
                "RAW_R_Count": 0,
                "RAW_G1_Count": 0,
                "RAW_G2_Count": 0,
                "RAW_B_Count": 0,
            }

        roi_mean = np.nan
        if effective_mask is not None and np.any(effective_mask > 0):
            roi_mean = float(np.mean(image_data[effective_mask > 0]))
        bbox_values = self._bbox_values(bbox)

        ts_row = [
            int(frame_id),
            self._fmt_value(np.mean(image_data)),
            self._fmt_value(roi_mean),
            "N/A",
            "N/A",
            self._fmt_value(raw_stats.get("RAW_R_Mean")),
            self._fmt_value(raw_stats.get("RAW_G1_Mean")),
            self._fmt_value(raw_stats.get("RAW_G2_Mean")),
            self._fmt_value(raw_stats.get("RAW_G_Mean")),
            self._fmt_value(raw_stats.get("RAW_B_Mean")),
            int(raw_stats.get("RAW_R_Count", 0)),
            int(raw_stats.get("RAW_G1_Count", 0)),
            int(raw_stats.get("RAW_G2_Count", 0)),
            int(raw_stats.get("RAW_B_Count", 0)),
            int(valid_pixel_count),
            int(skin_pixel_count),
            mask_source,
            mask_hash,
            mask_file,
            *bbox_values,
        ]
        with open(self.timeseries_path, mode="a", newline="") as f:
            csv.writer(f).writerow(ts_row)

        manifest_row = [
            int(frame_id),
            self.recorded_count,
            mask_file,
            mask_hash,
            int(valid_pixel_count),
            int(skin_pixel_count),
            mask_source,
            *bbox_values,
            int(frame_h),
            int(frame_w),
        ]
        with open(self.manifest_path, mode="a", newline="") as f:
            csv.writer(f).writerow(manifest_row)

        self.recorded_count += 1


class PipelineProbe:
    """
    ISP 管线探针模块。

    当前版本对齐 pyVHR 的 holistic skin mask 思路：
    1. 基于 reference RGB 做 landmarks + ConvexHull skin mask
    2. 对 skin mask 应用 pyVHR 的亮度阈值过滤
    3. 统一以 valid_skin_mask 对各域进行均值统计
    4. 基本与pyVHR的SkinExtractionConvexHull是相同的人脸mask方法
    """

    _BAYER_OFFSETS = {
        "RGGB": {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)},
        "BGGR": {"B": (0, 0), "G1": (0, 1), "G2": (1, 0), "R": (1, 1)},
        "GRBG": {"G1": (0, 0), "R": (0, 1), "B": (1, 0), "G2": (1, 1)},
        "GBRG": {"G1": (0, 0), "B": (0, 1), "R": (1, 0), "G2": (1, 1)},
    }

    def __init__(
        self,
        probe_name: str,
        save_dir: str = "isp_probes",
        save_npy: bool = False,
        save_preview: bool = True,
        auto_detect_roi: bool = True,
        fallback_roi: tuple = None,
        start_frame: int = 1000,
        max_frames: int = 20,
        max_csv_frames: int = 1000,
        max_preview_frames: int = 1000,
        raw_bayer_pattern: str = None,
        frame_domain: str = "auto",
        yuv_color_method: str = "bt709",
        skin_context: ProbeSkinContext = None,
        raw_reference_config: dict = None,
        standard_output_enabled: bool = True,
    ):
        self.probe_name = probe_name
        self.save_dir = os.path.join(save_dir, probe_name)
        self.standard_output_enabled = bool(standard_output_enabled)
        self.save_npy = save_npy
        self.save_preview = save_preview

        self.auto_detect_roi = auto_detect_roi
        self.fallback_roi = fallback_roi
        self.current_roi = fallback_roi if not auto_detect_roi else None

        self.start_frame = start_frame
        self.max_frames = max_frames
        self.max_csv_frames = max_csv_frames if max_csv_frames is not None else max_frames
        self.max_preview_frames = max_preview_frames if max_preview_frames is not None else max_frames
        self.raw_bayer_pattern = raw_bayer_pattern.upper() if raw_bayer_pattern else None
        if self.raw_bayer_pattern is not None and self.raw_bayer_pattern not in self._BAYER_OFFSETS:
            raise ValueError(
                f"Unsupported raw_bayer_pattern={raw_bayer_pattern}. "
                f"Expected one of {sorted(self._BAYER_OFFSETS)}."
            )
        self.frame_domain = (frame_domain or "auto").lower()
        self.yuv_color_method = (yuv_color_method or "bt709").lower()
        self.is_raw_bayer_mode = self.raw_bayer_pattern is not None
        self.skin_context = skin_context if skin_context is not None else (
            ProbeSkinContext() if self.auto_detect_roi else None
        )

        if self.standard_output_enabled:
            os.makedirs(self.save_dir, exist_ok=True)

        self.global_frame_count = 0
        self.recorded_csv_count = 0
        self.recorded_preview_count = 0
        self.prev_frame = None
        self._prev_effective_mask = None

        self._skin_mask = None
        self._valid_skin_mask = None
        self._mask_source = "none"
        self._skin_pixel_count = 0
        self._valid_pixel_count = 0

        self.csv_path = os.path.join(self.save_dir, f"{probe_name}_timeseries.csv")
        self.meta_path = os.path.join(self.save_dir, "probe_meta.json")
        self._meta_written_with_shape = False
        self.raw_reference_store = RawInputReferenceStore(raw_reference_config)
        if self.standard_output_enabled:
            self._init_csv()
            self._write_probe_meta()

    @classmethod
    def _get_bayer_offsets(cls, pattern: str):
        if pattern is None:
            return None
        return cls._BAYER_OFFSETS[pattern]

    def _is_raw_bayer_frame(self, image_data: np.ndarray) -> bool:
        return self.is_raw_bayer_mode and image_data.ndim == 2

    def _empty_raw_stats(self):
        return {
            "RAW_R_Mean": np.nan,
            "RAW_G1_Mean": np.nan,
            "RAW_G2_Mean": np.nan,
            "RAW_G_Mean": np.nan,
            "RAW_B_Mean": np.nan,
            "RAW_R_Count": 0,
            "RAW_G1_Count": 0,
            "RAW_G2_Count": 0,
            "RAW_B_Count": 0,
        }

    @staticmethod
    def _fmt_value(value, default: str = "N/A") -> str:
        if value is None:
            return default
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return default
        return f"{float(value):.6f}"

    @staticmethod
    def _mask_bbox(mask: np.ndarray):
        if mask is None or not np.any(mask > 0):
            return None
        ys, xs = np.where(mask > 0)
        return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))

    def _collect_bayer_pixels(self, image_data: np.ndarray, mask: np.ndarray):
        stats = {"R": np.array([], dtype=image_data.dtype), "G1": np.array([], dtype=image_data.dtype),
                 "G2": np.array([], dtype=image_data.dtype), "B": np.array([], dtype=image_data.dtype)}
        if not self._is_raw_bayer_frame(image_data) or mask is None:
            return stats

        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return stats

        offsets = self._get_bayer_offsets(self.raw_bayer_pattern)
        for channel, (off_y, off_x) in offsets.items():
            phase_mask = ((ys % 2) == off_y) & ((xs % 2) == off_x)
            if np.any(phase_mask):
                stats[channel] = image_data[ys[phase_mask], xs[phase_mask]]
        return stats

    def _compute_raw_bayer_stats(self, image_data: np.ndarray, mask: np.ndarray):
        out = self._empty_raw_stats()
        pixels = self._collect_bayer_pixels(image_data, mask)
        for channel in ["R", "G1", "G2", "B"]:
            vals = pixels[channel]
            out[f"RAW_{channel}_Count"] = int(vals.size)
            if vals.size > 0:
                out[f"RAW_{channel}_Mean"] = float(np.mean(vals))

        g_pixels = []
        if pixels["G1"].size > 0:
            g_pixels.append(pixels["G1"])
        if pixels["G2"].size > 0:
            g_pixels.append(pixels["G2"])
        if g_pixels:
            out["RAW_G_Mean"] = float(np.mean(np.concatenate(g_pixels)))
        return out

    def _compute_raw_bayer_delta(self, image_data: np.ndarray, prev_image: np.ndarray, common_mask: np.ndarray):
        deltas = {
            "RAW_R_AC_Delta": np.nan,
            "RAW_G1_AC_Delta": np.nan,
            "RAW_G2_AC_Delta": np.nan,
            "RAW_G_AC_Delta": np.nan,
            "RAW_B_AC_Delta": np.nan,
        }
        if not self._is_raw_bayer_frame(image_data) or prev_image is None or common_mask is None:
            return deltas

        curr_pixels = self._collect_bayer_pixels(image_data, common_mask)
        prev_pixels = self._collect_bayer_pixels(prev_image, common_mask)
        g_diffs = []
        for channel in ["R", "G1", "G2", "B"]:
            curr = curr_pixels[channel]
            prev = prev_pixels[channel]
            if curr.size > 0 and prev.size == curr.size:
                diff = np.abs(curr.astype(np.float32) - prev.astype(np.float32))
                deltas[f"RAW_{channel}_AC_Delta"] = float(np.mean(diff))
                if channel in ("G1", "G2"):
                    g_diffs.append(diff)
        if g_diffs:
            deltas["RAW_G_AC_Delta"] = float(np.mean(np.concatenate(g_diffs)))
        return deltas

    def _init_csv(self):
        headers = ["Frame_ID", "Global_Mean", "Global_AC_Delta"]
        if self.auto_detect_roi or self.fallback_roi:
            headers.extend(["ROI_Mean_C0", "ROI_Mean_C1", "ROI_Mean_C2", "ROI_AC_Delta"])
        if self.is_raw_bayer_mode:
            headers.extend([
                "RAW_R_Mean", "RAW_G1_Mean", "RAW_G2_Mean", "RAW_G_Mean", "RAW_B_Mean",
                "RAW_R_Count", "RAW_G1_Count", "RAW_G2_Count", "RAW_B_Count",
                "RAW_R_AC_Delta", "RAW_G1_AC_Delta", "RAW_G2_AC_Delta", "RAW_G_AC_Delta", "RAW_B_AC_Delta",
            ])
        headers.extend(["ROI_Valid_Pixel_Count", "ROI_Skin_Pixel_Count", "ROI_Mask_Source"])

        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _write_probe_meta(self, image_shape=None):
        meta = {
            "probe_name": self.probe_name,
            "domain": "raw" if self.is_raw_bayer_mode else self.frame_domain,
            "frame_domain": self.frame_domain,
            "yuv_color_method": self.yuv_color_method if self.frame_domain == "yuv" else None,
            "raw_bayer_pattern": self.raw_bayer_pattern,
            "raw_mode": "bayer_aware" if self.is_raw_bayer_mode else "legacy",
            "auto_detect_roi": self.auto_detect_roi,
            "fallback_roi": list(self.fallback_roi) if self.fallback_roi is not None else None,
            "start_frame": self.start_frame,
            "max_frames": self.max_frames,
            "max_csv_frames": self.max_csv_frames,
            "max_preview_frames": self.max_preview_frames,
            "save_npy": self.save_npy,
            "save_preview": self.save_preview,
            "roi_mode": "skin_mask" if self.auto_detect_roi else ("fallback_roi" if self.fallback_roi else "none"),
            "image_shape": list(image_shape) if image_shape is not None else None,
            "skin_backend": "pyvhr_convex_hull" if self.auto_detect_roi else "none",
            "skin_source_mode": "reference_rgb" if self.auto_detect_roi else "none",
            "rgb_low_th": int(self.skin_context.rgb_low_th) if self.skin_context is not None else None,
            "rgb_high_th": int(self.skin_context.rgb_high_th) if self.skin_context is not None else None,
            "max_mask_stale_frames": self.skin_context.max_stale_frames if self.skin_context is not None else None,
        }
        with open(self.meta_path, mode="w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        self._meta_written_with_shape = image_shape is not None

    def _prepare_masks(self, image_data: np.ndarray):
        self._skin_mask = None
        self._valid_skin_mask = None
        self._mask_source = "none"
        self._skin_pixel_count = 0
        self._valid_pixel_count = 0
        self.current_roi = None if self.auto_detect_roi else self.fallback_roi

        if self.auto_detect_roi and self.skin_context is not None:
            self.skin_context.prepare(
                self.global_frame_count,
                image_data,
                frame_domain=self.frame_domain,
                raw_bayer_pattern=self.raw_bayer_pattern,
                yuv_color_method=self.yuv_color_method,
            )
            if self.skin_context.frame_shape == tuple(image_data.shape[:2]):
                self._skin_mask = self.skin_context.skin_mask
                self._valid_skin_mask = self.skin_context.valid_skin_mask
                self._mask_source = self.skin_context.mask_source
                self._skin_pixel_count = self.skin_context.skin_pixel_count
                self._valid_pixel_count = self.skin_context.valid_pixel_count
                self.current_roi = self.skin_context.bbox
        elif self.fallback_roi:
            self.current_roi = self.fallback_roi

    @staticmethod
    def _image_to_preview_bgr(image_data: np.ndarray, frame_domain: str, yuv_color_method: str = "bt709") -> np.ndarray:
        if image_data.ndim == 2:
            gray8 = ProbeSkinContext._normalize_image_to_uint8_rgb(image_data)
            return cv2.cvtColor(gray8, cv2.COLOR_RGB2BGR)

        domain = (frame_domain or "auto").lower()
        if domain == "yuv":
            rgb_float = ProbeSkinContext._yuv_to_rgb(image_data, yuv_color_method)
            rgb8 = ProbeSkinContext._normalize_image_to_uint8_rgb(rgb_float)
        else:
            rgb8 = ProbeSkinContext._normalize_image_to_uint8_rgb(image_data)
        return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)

    def execute(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        self.global_frame_count += 1
        if self.global_frame_count < self.start_frame:
            return image_data

        csv_full = (
            (not self.standard_output_enabled)
            or ((self.max_csv_frames is not None) and (self.recorded_csv_count >= self.max_csv_frames))
        )
        preview_full = (
            (not self.standard_output_enabled)
            or ((self.max_preview_frames is not None) and (self.recorded_preview_count >= self.max_preview_frames))
        )
        reference_full = self.raw_reference_store.is_full()
        if csv_full and preview_full and reference_full:
            return image_data

        file_prefix = f"frame_{self.global_frame_count:04d}"
        print(f"\n🔍 [探针] 拦截位置: {self.probe_name} (物理帧 {self.global_frame_count})")
        print(f"   📊 CSV录制进度: {self.recorded_csv_count}/{self.max_csv_frames} | 🖼️ 图片保存进度: {self.recorded_preview_count}/{self.max_preview_frames}")

        shape = image_data.shape
        is_3_channel = len(shape) == 3 and shape[2] == 3
        if self.standard_output_enabled and not self._meta_written_with_shape:
            self._write_probe_meta(shape)

        self._prepare_masks(image_data)

        effective_mask = None
        if self._valid_skin_mask is not None:
            effective_mask = self._valid_skin_mask
        elif not self.auto_detect_roi and self.current_roi:
            y1, y2, x1, x2 = self.current_roi
            rect_mask = np.zeros(image_data.shape[:2], dtype=np.uint8)
            rect_mask[y1:y2, x1:x2] = 255
            effective_mask = rect_mask

        if not csv_full:
            csv_row = [self.global_frame_count]
            raw_stats = None
            raw_deltas = None

            csv_row.append(f"{np.mean(image_data):.6f}")
            if self.prev_frame is not None:
                csv_row.append(f"{np.mean(np.abs(image_data - self.prev_frame)):.6f}")
            else:
                csv_row.append("0.000000")

            if effective_mask is not None and np.any(effective_mask > 0):
                if is_3_channel:
                    skin_pixels = image_data[effective_mask > 0]
                    roi_means = np.mean(skin_pixels, axis=0)
                    csv_row.extend([f"{roi_means[0]:.6f}", f"{roi_means[1]:.6f}", f"{roi_means[2]:.6f}"])
                else:
                    skin_pixels = image_data[effective_mask > 0]
                    csv_row.extend([f"{np.mean(skin_pixels):.6f}", "N/A", "N/A"])
                    if self._is_raw_bayer_frame(image_data):
                        raw_stats = self._compute_raw_bayer_stats(image_data, effective_mask)

                if self.prev_frame is not None and self._prev_effective_mask is not None:
                    common = (effective_mask > 0) & (self._prev_effective_mask > 0)
                    if np.any(common):
                        csv_row.append(f"{np.mean(np.abs(image_data[common] - self.prev_frame[common])):.6f}")
                        if self._is_raw_bayer_frame(image_data):
                            raw_deltas = self._compute_raw_bayer_delta(image_data, self.prev_frame, common)
                    else:
                        csv_row.append("0.000000")
                else:
                    csv_row.append("0.000000")
            else:
                csv_row.extend(["N/A", "N/A", "N/A", "0.000000"])

            if self.is_raw_bayer_mode:
                if raw_stats is None:
                    raw_stats = self._empty_raw_stats()
                if raw_deltas is None:
                    raw_deltas = self._compute_raw_bayer_delta(image_data, None, None)
                csv_row.extend([
                    self._fmt_value(raw_stats["RAW_R_Mean"]),
                    self._fmt_value(raw_stats["RAW_G1_Mean"]),
                    self._fmt_value(raw_stats["RAW_G2_Mean"]),
                    self._fmt_value(raw_stats["RAW_G_Mean"]),
                    self._fmt_value(raw_stats["RAW_B_Mean"]),
                    str(raw_stats["RAW_R_Count"]),
                    str(raw_stats["RAW_G1_Count"]),
                    str(raw_stats["RAW_G2_Count"]),
                    str(raw_stats["RAW_B_Count"]),
                    self._fmt_value(raw_deltas["RAW_R_AC_Delta"]),
                    self._fmt_value(raw_deltas["RAW_G1_AC_Delta"]),
                    self._fmt_value(raw_deltas["RAW_G2_AC_Delta"]),
                    self._fmt_value(raw_deltas["RAW_G_AC_Delta"]),
                    self._fmt_value(raw_deltas["RAW_B_AC_Delta"]),
                ])

            csv_row.extend([
                str(int(self._valid_pixel_count)),
                str(int(self._skin_pixel_count)),
                self._mask_source,
            ])

            with open(self.csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(csv_row)

            self.recorded_csv_count += 1
        elif self._is_raw_bayer_frame(image_data) and effective_mask is not None and np.any(effective_mask > 0):
            raw_stats = self._compute_raw_bayer_stats(image_data, effective_mask)
        else:
            raw_stats = None

        if not reference_full and self._is_raw_bayer_frame(image_data):
            if raw_stats is None and effective_mask is not None and np.any(effective_mask > 0):
                raw_stats = self._compute_raw_bayer_stats(image_data, effective_mask)
            self.raw_reference_store.record(
                frame_id=self.global_frame_count,
                image_data=image_data,
                effective_mask=effective_mask,
                raw_stats=raw_stats,
                skin_pixel_count=self._skin_pixel_count,
                valid_pixel_count=self._valid_pixel_count,
                mask_source=self._mask_source,
                bbox=self.current_roi,
            )

        self.prev_frame = image_data.copy()
        self._prev_effective_mask = None if effective_mask is None else (effective_mask > 0).copy()

        if self.standard_output_enabled and not preview_full:
            if self.save_npy:
                np.save(os.path.join(self.save_dir, f"{file_prefix}_raw.npy"), image_data)

            if self.save_preview:
                preview_path = os.path.join(self.save_dir, f"{file_prefix}_preview.png")
                vis_bgr = self._image_to_preview_bgr(
                    image_data,
                    frame_domain=self.frame_domain,
                    yuv_color_method=self.yuv_color_method,
                )

                skin_mask = self._skin_mask if self._skin_mask is not None and np.any(self._skin_mask > 0) else None
                valid_mask = self._valid_skin_mask if self._valid_skin_mask is not None and np.any(self._valid_skin_mask > 0) else None
                if skin_mask is not None:
                    overlay = vis_bgr.copy()
                    overlay[skin_mask == 0] = (overlay[skin_mask == 0] * 0.25).astype(np.uint8)

                    # 整体皮肤区域使用柔和绿色覆盖。
                    skin_layer = np.zeros_like(vis_bgr)
                    skin_layer[skin_mask > 0] = (0, 72, 0)
                    vis_bgr = cv2.addWeighted(overlay, 1.0, skin_layer, 0.38, 0)

                    # 被 valid mask 过滤掉的皮肤像素单独提亮为浅青色，避免看起来像“嘴下黑洞”。
                    if valid_mask is not None:
                        invalid_within_skin = (skin_mask > 0) & (valid_mask == 0)
                        if np.any(invalid_within_skin):
                            invalid_layer = np.zeros_like(vis_bgr)
                            invalid_layer[invalid_within_skin] = (160, 200, 0)
                            vis_bgr = cv2.addWeighted(vis_bgr, 1.0, invalid_layer, 0.28, 0)

                        valid_outline = cv2.Canny(valid_mask.astype(np.uint8), 50, 150)
                        vis_bgr[valid_outline > 0] = (255, 255, 255)
                elif self.current_roi:
                    y1, y2, x1, x2 = self.current_roi
                    cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

                text1 = f"Skin px: {self._skin_pixel_count} | Valid px: {self._valid_pixel_count}"
                text2 = f"Mask: {self._mask_source}"
                cv2.putText(vis_bgr, text1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(vis_bgr, text2, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.imwrite(preview_path, vis_bgr)

            self.recorded_preview_count += 1

        print(f"   ✅ 数据放行. (ROI: {'有效' if effective_mask is not None else '无有效皮肤ROI'})")
        return image_data
