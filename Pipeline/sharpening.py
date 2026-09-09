# sharpen.py
import numpy as np
import cv2
from scipy.ndimage import convolve, gaussian_filter, uniform_filter

from Pipeline.probe_telemetry_metrics import operatorActivity


class Sharpen:
    """
    锐化模块：提供多种锐化算法。
    
    所有方法均假设输入为 np.float32 YUV图像 (包含 HDR 范围)，
    输出也为 np.float32 YUV图像。
    锐化操作将只在 Y (亮度) 通道上执行。
    """

    def __init__(self):
        self.last_probe_telemetry = {}
        self._active_algorithm = None
        self._rppg_stable_detail = None
        self._rppg_previous_y = None
        self._rppg_state_signature = None
        self._last_rppg_diagnostics = {}
        self._last_sharpen_diagnostics = {}
    
    def execute(self, yuv_image: np.ndarray, algorithm: str = 'unsharp_mask', **kwargs) -> np.ndarray:
        """
        执行图像锐化。通常只对Y通道（亮度）进行锐化。
        
        Args:
            yuv_image: 输入的YUV图像 (np.float32)
            algorithm: 锐化算法选择
                'laplacian': 拉普拉斯锐化
                'unsharp_mask': 非锐化掩蔽 (USM)
                'high_pass': 高通滤波锐化
                'sobel': Sobel边缘增强
                'gaussian_unsharp': 高斯USM (更平滑)
                'adaptive': 自适应锐化
                'wiener': 维纳滤波锐化
                'rppg_temporal_consistency_sharpen': rPPG 时间一致性门控锐化
                'coring_edge_aware_sharpen': Coring 边缘感知锐化
                'guided_filter_sharpen': 导向滤波锐化
                'halo_suppressed_sharpen': 光晕抑制锐化
                'bilateral_detail_sharpen': 双边细节锐化
            **kwargs: 算法特定参数
        
        Returns:
            锐化后的YUV图像 (np.float32)
        """
        print(f"Executing Sharpening using algorithm: {algorithm}")
        
        # ===================================================================
        # 🌟 [修改] 零损耗数据流：信任上游传递的浮点 HDR 数据
        # 不再因为 max() > 1.0 而错误地除以 65535.0
        # ===================================================================
        if not np.issubdtype(yuv_image.dtype, np.floating):
            print(f"  [警告] Sharpen 模块期望接收 float32，当前为 {yuv_image.dtype}。正在转换...")
            yuv_image = yuv_image.astype(np.float32)

        # 只对Y通道进行锐化，保持UV通道不变
        y_channel, u_channel, v_channel = cv2.split(yuv_image)

        if algorithm != self._active_algorithm:
            self.reset_temporal_state()
            self._active_algorithm = algorithm
        
        if algorithm == 'laplacian':
            effective_params = {"strength": 1.0}
            effective_params.update(self._pick_kwargs(kwargs, {'strength'}))
            y_sharpened = self._laplacian_sharpen(
                y_channel, **self._pick_kwargs(kwargs, {'strength'})
            )
        elif algorithm == 'unsharp_mask':
            effective_params = {"radius": 1.0, "amount": 1.0, "threshold": 0.0}
            effective_params.update(self._pick_kwargs(kwargs, {'radius', 'amount', 'threshold'}))
            y_sharpened = self._unsharp_mask(
                y_channel, **self._pick_kwargs(kwargs, {'radius', 'amount', 'threshold'})
            )
        elif algorithm == 'high_pass':
            effective_params = {"radius": 5, "strength": 1.5}
            effective_params.update(self._pick_kwargs(kwargs, {'radius', 'strength'}))
            effective_params["radius"] = max(1, int(round(effective_params["radius"])))
            y_sharpened = self._high_pass_sharpen(
                y_channel, **self._pick_kwargs(kwargs, {'radius', 'strength'})
            )
        elif algorithm == 'sobel':
            effective_params = {"strength": 1.0}
            effective_params.update(self._pick_kwargs(kwargs, {'strength'}))
            y_sharpened = self._sobel_sharpen(
                y_channel, **self._pick_kwargs(kwargs, {'strength'})
            )
        elif algorithm == 'gaussian_unsharp':
            effective_params = {"sigma": 1.0, "amount": 1.5}
            effective_params.update(self._pick_kwargs(kwargs, {'sigma', 'amount'}))
            y_sharpened = self._gaussian_unsharp(
                y_channel, **self._pick_kwargs(kwargs, {'sigma', 'amount'})
            )
        elif algorithm == 'adaptive':
            effective_params = {"strength": 1.0, "window_size": 5}
            effective_params.update(self._pick_kwargs(kwargs, {'strength', 'window_size'}))
            y_sharpened = self._adaptive_sharpen(
                y_channel, **self._pick_kwargs(kwargs, {'strength', 'window_size'})
            )
        elif algorithm == 'wiener':
            effective_params = {"noise_variance": 0.001, "strength": 1.0}
            effective_params.update(self._pick_kwargs(kwargs, {'noise_variance', 'strength'}))
            y_sharpened = self._wiener_sharpen(
                y_channel, **self._pick_kwargs(kwargs, {'noise_variance', 'strength'})
            )
        elif algorithm == 'rppg_temporal_consistency_sharpen':
            allowed_keys = {
                'radius', 'amount', 'temporal_alpha', 'detail_threshold',
                'edge_low', 'edge_high', 'motion_threshold',
            }
            effective_params = {
                "radius": 1.0,
                "amount": 0.75,
                "temporal_alpha": 0.2,
                "detail_threshold": 0.01,
                "edge_low": 0.01,
                "edge_high": 0.03,
                "motion_threshold": 0.05,
            }
            effective_params.update(self._pick_kwargs(kwargs, allowed_keys))
            effective_params = self._validate_rppg_params(effective_params)
            y_sharpened = self._rppg_temporal_consistency_sharpen(
                y_channel, **effective_params
            )
        elif algorithm == 'coring_edge_aware_sharpen':
            effective_params = {
                "radius": 1.0, "amount": 0.75, "coring_threshold": 0.01,
                "edge_threshold": 0.03,
            }
            effective_params.update(self._pick_kwargs(
                kwargs, {'radius', 'amount', 'coring_threshold', 'edge_threshold'}
            ))
            effective_params = self._validate_coring_params(effective_params)
            y_sharpened = self._coring_edge_aware_sharpen(y_channel, **effective_params)
        elif algorithm == 'guided_filter_sharpen':
            effective_params = {"radius": 2, "eps": 1e-3, "amount": 0.75}
            effective_params.update(self._pick_kwargs(kwargs, {'radius', 'eps', 'amount'}))
            effective_params = self._validate_guided_params(effective_params)
            y_sharpened = self._guided_filter_sharpen(y_channel, **effective_params)
        elif algorithm == 'halo_suppressed_sharpen':
            effective_params = {
                "radius": 1.0, "amount": 0.75, "halo_threshold": 0.02,
                "max_overshoot": 0.04,
            }
            effective_params.update(self._pick_kwargs(
                kwargs, {'radius', 'amount', 'halo_threshold', 'max_overshoot'}
            ))
            effective_params = self._validate_halo_params(effective_params)
            y_sharpened = self._halo_suppressed_sharpen(y_channel, **effective_params)
        elif algorithm == 'bilateral_detail_sharpen':
            effective_params = {
                "diameter": 5, "sigma_color": 0.08, "sigma_space": 2.0,
                "amount": 0.75,
            }
            effective_params.update(self._pick_kwargs(
                kwargs, {'diameter', 'sigma_color', 'sigma_space', 'amount'}
            ))
            effective_params = self._validate_bilateral_params(effective_params)
            y_sharpened = self._bilateral_detail_sharpen(y_channel, **effective_params)
        else:
            raise ValueError(f"Unknown sharpening algorithm: {algorithm}")

        if y_sharpened.dtype != y_channel.dtype:
            y_sharpened = y_sharpened.astype(y_channel.dtype, copy=False)
        
        # 合并通道
        output = cv2.merge([y_sharpened, u_channel, v_channel])
        activity = operatorActivity(yuv_image, output, primaryChannel=0)
        sample_step = int(activity.get("spatial_sample_step", 1))
        y_before = y_channel[::sample_step, ::sample_step].astype(np.float64, copy=False)
        y_after = y_sharpened[::sample_step, ::sample_step].astype(np.float64, copy=False)
        residual = y_after - y_before
        uv_error = np.abs(
            output[::sample_step, ::sample_step, 1:].astype(np.float64)
            - yuv_image[::sample_step, ::sample_step, 1:].astype(np.float64)
        )
        self.last_probe_telemetry = {
            "requested_algorithm": str(algorithm),
            "effective_algorithm": str(algorithm),
            "effective_params": effective_params,
            "identity": bool(np.array_equal(y_sharpened, y_channel)),
            "operator_activity": activity,
            "sharpening_active_fraction_sampled": float(np.mean(np.abs(residual) > 1e-12)),
            "positive_luma_residual_fraction_sampled": float(np.mean(residual > 1e-12)),
            "negative_luma_residual_fraction_sampled": float(np.mean(residual < -1e-12)),
            "configured_threshold": (
                float(effective_params["threshold"])
                if algorithm == 'unsharp_mask' else None
            ),
            "created_overshoot_fraction_sampled": float(np.mean((y_before <= 1.0) & (y_after > 1.0))),
            "created_undershoot_fraction_sampled": float(np.mean((y_before >= 0.0) & (y_after < 0.0))),
            "overshoot_above_one_mean_sampled": float(np.mean(np.maximum(y_after - 1.0, 0.0))),
            "undershoot_below_zero_mean_sampled": float(np.mean(np.maximum(-y_after, 0.0))),
            "uv_unchanged_abs_max_sampled": float(np.max(uv_error)) if uv_error.size else 0.0,
        }
        if algorithm == 'rppg_temporal_consistency_sharpen':
            self.last_probe_telemetry["rppg_temporal_consistency"] = dict(
                self._last_rppg_diagnostics
            )
        if algorithm in {
            'coring_edge_aware_sharpen', 'guided_filter_sharpen',
            'halo_suppressed_sharpen', 'bilateral_detail_sharpen',
        }:
            self.last_probe_telemetry["sharpen_diagnostics"] = dict(
                getattr(self, "_last_sharpen_diagnostics", {})
            )
        return output

    @staticmethod
    def _pick_kwargs(kwargs: dict, allowed_keys: set) -> dict:
        """Keep algorithm scans robust when baseline config contains params for another algorithm."""
        return {key: kwargs[key] for key in allowed_keys if key in kwargs}

    def reset_temporal_state(self) -> None:
        """Reset video-dependent state before a new sequence or algorithm."""
        self._rppg_stable_detail = None
        self._rppg_previous_y = None
        self._rppg_state_signature = None
        self._last_rppg_diagnostics = {}

    @staticmethod
    def _validate_rppg_params(params: dict) -> dict:
        validated = {key: float(value) for key, value in params.items()}
        if not all(np.isfinite(value) for value in validated.values()):
            raise ValueError("rPPG sharpening parameters must be finite")
        if validated["radius"] <= 0:
            raise ValueError("radius must be > 0")
        if validated["amount"] < 0:
            raise ValueError("amount must be >= 0")
        if not 0 < validated["temporal_alpha"] <= 1:
            raise ValueError("temporal_alpha must be in (0, 1]")
        if validated["detail_threshold"] <= 0:
            raise ValueError("detail_threshold must be > 0")
        if validated["edge_low"] < 0:
            raise ValueError("edge_low must be >= 0")
        if validated["edge_high"] <= validated["edge_low"]:
            raise ValueError("edge_high must be greater than edge_low")
        if validated["motion_threshold"] <= 0:
            raise ValueError("motion_threshold must be > 0")
        return validated

    @staticmethod
    def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
        return normalized * normalized * (3.0 - 2.0 * normalized)

    @staticmethod
    def _validate_coring_params(params: dict) -> dict:
        validated = {key: float(value) for key, value in params.items()}
        if not all(np.isfinite(value) for value in validated.values()):
            raise ValueError("coring sharpening parameters must be finite")
        if validated["radius"] <= 0 or validated["amount"] < 0:
            raise ValueError("radius must be > 0 and amount must be >= 0")
        if validated["coring_threshold"] < 0 or validated["edge_threshold"] < 0:
            raise ValueError("coring and edge thresholds must be >= 0")
        return validated

    @staticmethod
    def _validate_guided_params(params: dict) -> dict:
        validated = {key: float(value) for key, value in params.items()}
        if not all(np.isfinite(value) for value in validated.values()):
            raise ValueError("guided sharpening parameters must be finite")
        if validated["radius"] < 1 or validated["eps"] <= 0 or validated["amount"] < 0:
            raise ValueError("radius must be >= 1, eps must be > 0 and amount must be >= 0")
        validated["radius"] = int(round(validated["radius"]))
        return validated

    @staticmethod
    def _validate_halo_params(params: dict) -> dict:
        validated = {key: float(value) for key, value in params.items()}
        if not all(np.isfinite(value) for value in validated.values()):
            raise ValueError("halo suppression parameters must be finite")
        if validated["radius"] <= 0 or validated["amount"] < 0:
            raise ValueError("radius must be > 0 and amount must be >= 0")
        if validated["halo_threshold"] < 0 or validated["max_overshoot"] <= 0:
            raise ValueError("halo thresholds must be non-negative and max_overshoot must be > 0")
        return validated

    @staticmethod
    def _validate_bilateral_params(params: dict) -> dict:
        validated = {key: float(value) for key, value in params.items()}
        if not all(np.isfinite(value) for value in validated.values()):
            raise ValueError("bilateral sharpening parameters must be finite")
        if validated["diameter"] == 0 or validated["sigma_color"] <= 0:
            raise ValueError("diameter must be non-zero and sigma_color must be > 0")
        if validated["sigma_space"] <= 0 or validated["amount"] < 0:
            raise ValueError("sigma_space must be > 0 and amount must be >= 0")
        validated["diameter"] = int(round(validated["diameter"]))
        return validated

    def _coring_edge_aware_sharpen(
        self, channel: np.ndarray, radius: float = 1.0, amount: float = 0.75,
        coring_threshold: float = 0.01, edge_threshold: float = 0.03,
    ) -> np.ndarray:
        working = channel.astype(np.float32, copy=False)
        blurred = cv2.GaussianBlur(working, (0, 0), radius)
        detail = working - blurred
        magnitude = np.abs(detail)
        coring = np.where(
            magnitude <= coring_threshold,
            0.0,
            np.sign(detail) * (magnitude - coring_threshold),
        )
        edge_gain = self._smoothstep(magnitude, edge_threshold * 0.5, edge_threshold)
        gain = edge_gain * coring
        self._last_sharpen_diagnostics = {
            "detail_abs_mean": float(np.mean(magnitude)),
            "coring_suppressed_fraction": float(np.mean(magnitude <= coring_threshold)),
            "edge_gain_mean": float(np.mean(edge_gain)),
        }
        return working + amount * gain

    @staticmethod
    def _guided_filter(guide: np.ndarray, source: np.ndarray, radius: int, eps: float) -> np.ndarray:
        window = 2 * radius + 1
        mean_guide = cv2.boxFilter(guide, -1, (window, window), normalize=True)
        mean_source = cv2.boxFilter(source, -1, (window, window), normalize=True)
        corr_guide = cv2.boxFilter(guide * guide, -1, (window, window), normalize=True)
        corr_cross = cv2.boxFilter(guide * source, -1, (window, window), normalize=True)
        variance = np.maximum(corr_guide - mean_guide * mean_guide, 0.0)
        covariance = corr_cross - mean_guide * mean_source
        linear_a = covariance / (variance + eps)
        linear_b = mean_source - linear_a * mean_guide
        mean_a = cv2.boxFilter(linear_a, -1, (window, window), normalize=True)
        mean_b = cv2.boxFilter(linear_b, -1, (window, window), normalize=True)
        return mean_a * guide + mean_b

    def _guided_filter_sharpen(
        self, channel: np.ndarray, radius: int = 2, eps: float = 1e-3,
        amount: float = 0.75,
    ) -> np.ndarray:
        working = channel.astype(np.float32, copy=False)
        base = self._guided_filter(working, working, radius, eps)
        detail = working - base
        self._last_sharpen_diagnostics = {
            "detail_abs_mean": float(np.mean(np.abs(detail))),
            "base_abs_mean": float(np.mean(np.abs(base))),
            "guided_radius": int(radius),
        }
        return working + amount * detail

    def _halo_suppressed_sharpen(
        self, channel: np.ndarray, radius: float = 1.0, amount: float = 0.75,
        halo_threshold: float = 0.02, max_overshoot: float = 0.04,
    ) -> np.ndarray:
        working = channel.astype(np.float32, copy=False)
        blurred = cv2.GaussianBlur(working, (0, 0), radius)
        detail = working - blurred
        magnitude = np.abs(detail)
        gain = np.where(magnitude >= halo_threshold, 1.0, magnitude / max(halo_threshold, 1e-12))
        candidate = working + amount * gain * detail
        local_min = cv2.erode(working, np.ones((3, 3), np.uint8))
        local_max = cv2.dilate(working, np.ones((3, 3), np.uint8))
        lower = local_min - max_overshoot
        upper = local_max + max_overshoot
        output = np.minimum(np.maximum(candidate, lower), upper)
        self._last_sharpen_diagnostics = {
            "detail_abs_mean": float(np.mean(magnitude)),
            "halo_limited_fraction": float(np.mean((candidate < lower) | (candidate > upper))),
            "candidate_output_abs_mean": float(np.mean(np.abs(candidate - working))),
        }
        return output

    def _bilateral_detail_sharpen(
        self, channel: np.ndarray, diameter: int = 5, sigma_color: float = 0.08,
        sigma_space: float = 2.0, amount: float = 0.75,
    ) -> np.ndarray:
        working = channel.astype(np.float32, copy=False)
        base = cv2.bilateralFilter(
            working, diameter, sigma_color, sigma_space, borderType=cv2.BORDER_REFLECT101
        )
        detail = working - base
        self._last_sharpen_diagnostics = {
            "detail_abs_mean": float(np.mean(np.abs(detail))),
            "base_abs_mean": float(np.mean(np.abs(base))),
            "bilateral_diameter": int(diameter),
        }
        return working + amount * detail

    def _rppg_temporal_consistency_sharpen(
        self,
        channel: np.ndarray,
        radius: float = 1.0,
        amount: float = 0.75,
        temporal_alpha: float = 0.2,
        detail_threshold: float = 0.01,
        edge_low: float = 0.01,
        edge_high: float = 0.03,
        motion_threshold: float = 0.05,
    ) -> np.ndarray:
        """Sharpen persistent Y detail while rejecting transient detail and motion."""
        working = channel.astype(np.float32, copy=False)
        blurred = cv2.GaussianBlur(working, (0, 0), radius)
        current_detail = working - blurred
        state_signature = (
            working.shape,
            radius,
            amount,
            temporal_alpha,
            detail_threshold,
            edge_low,
            edge_high,
            motion_threshold,
        )

        reset_reason = None
        if self._rppg_stable_detail is None or self._rppg_previous_y is None:
            reset_reason = "initialization"
        elif self._rppg_state_signature != state_signature:
            old_shape = self._rppg_state_signature[0]
            reset_reason = "shape_changed" if old_shape != working.shape else "parameters_changed"

        if reset_reason is not None:
            stable_detail = current_detail.copy()
            discrepancy = np.zeros_like(current_detail)
            temporal_gate = np.ones_like(current_detail)
            motion_score = 0.0
            motion_detected = False
        else:
            frame_difference = np.abs(working - self._rppg_previous_y)
            motion_score = float(np.mean(frame_difference, dtype=np.float64))
            motion_detected = motion_score > motion_threshold

            if motion_detected:
                self._rppg_stable_detail = current_detail.copy()
                self._rppg_previous_y = working.copy()
                self._rppg_state_signature = state_signature
                self._last_rppg_diagnostics = {
                    "state_reset": True,
                    "state_reset_reason": "global_motion",
                    "motion_detected": True,
                    "motion_score": motion_score,
                    "instantaneous_detail_abs_mean": float(np.mean(np.abs(current_detail))),
                    "stable_detail_abs_mean": float(np.mean(np.abs(current_detail))),
                    "detail_discrepancy_abs_mean": 0.0,
                    "temporal_gate_mean": 0.0,
                    "edge_gate_mean": 0.0,
                    "combined_gate_mean": 0.0,
                    "combined_gate_active_fraction": 0.0,
                }
                return working.copy()

            previous_stable = self._rppg_stable_detail
            discrepancy = np.abs(current_detail - previous_stable)
            stable_detail = (
                temporal_alpha * current_detail
                + (1.0 - temporal_alpha) * previous_stable
            )
            temporal_gate = np.exp(-np.square(discrepancy / detail_threshold))

        edge_gate = self._smoothstep(np.abs(stable_detail), edge_low, edge_high)
        combined_gate = temporal_gate * edge_gate
        sharpened = working + amount * combined_gate * stable_detail

        self._rppg_stable_detail = stable_detail.copy()
        self._rppg_previous_y = working.copy()
        self._rppg_state_signature = state_signature
        self._last_rppg_diagnostics = {
            "state_reset": reset_reason is not None,
            "state_reset_reason": reset_reason,
            "motion_detected": motion_detected,
            "motion_score": motion_score,
            "instantaneous_detail_abs_mean": float(np.mean(np.abs(current_detail))),
            "stable_detail_abs_mean": float(np.mean(np.abs(stable_detail))),
            "detail_discrepancy_abs_mean": float(np.mean(discrepancy)),
            "temporal_gate_mean": float(np.mean(temporal_gate)),
            "edge_gate_mean": float(np.mean(edge_gate)),
            "combined_gate_mean": float(np.mean(combined_gate)),
            "combined_gate_active_fraction": float(np.mean(combined_gate > 1e-6)),
        }
        return sharpened
    
    def _laplacian_sharpen(self, channel: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        拉普拉斯锐化 - (float32 兼容)
        """
        # 拉普拉斯核
        kernel = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        # 计算拉普拉斯
        laplacian = convolve(channel, kernel, mode='reflect')
        
        # 添加回原图像
        sharpened = channel + strength * laplacian
        
        # ===================================================================
        # 🌟 [修改] 零损耗数据流：移除所有的 np.clip，保留锐化产生的负下冲和高光上冲
        # ===================================================================
        return sharpened
    
    def _unsharp_mask(self, channel: np.ndarray, radius: float = 1.0, 
                        amount: float = 1.0, threshold: float = 0) -> np.ndarray:
        """
        非锐化掩蔽 (USM) - (float32 兼容)
        """
        # 高斯模糊
        # cv2.GaussianBlur 完美支持 float32
        blurred = cv2.GaussianBlur(channel, (0, 0), radius)
        
        # 计算锐化掩模
        mask = channel - blurred
        
        # 应用阈值
        if threshold > 0:
            mask = np.where(np.abs(mask) >= threshold, mask, 0)
        
        # 应用锐化
        sharpened = channel + amount * mask
        return sharpened
    
    def _high_pass_sharpen(self, channel: np.ndarray, radius: int = 5, 
                           strength: float = 1.5) -> np.ndarray:
        """
        高通滤波锐化 - (float32 兼容)
        """
        # 低通滤波（模糊）
        # cv2.boxFilter 支持 float32 (ddepth=-1)
        radius = max(1, int(round(radius)))
        low_pass = cv2.boxFilter(channel, -1, (radius, radius))
        
        # 高通 = 原图 - 低通
        high_pass = channel - low_pass
        
        # 增强高频
        sharpened = channel + strength * high_pass
        return sharpened
    
    def _sobel_sharpen(self, channel: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Sobel边缘增强 - (float32 兼容)
        """
        # Sobel算子
        # cv2.Sobel 支持 float32 输入, 并通过 cv2.CV_32F 指定 float32 输出
        sobel_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        
        # 梯度幅值
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 添加边缘信息
        sharpened = channel + strength * gradient
        return sharpened
    
    def _gaussian_unsharp(self, channel: np.ndarray, sigma: float = 1.0, 
                          amount: float = 1.5) -> np.ndarray:
        """
        高斯非锐化掩蔽 - (float32 兼容)
        """
        # 高斯模糊
        # scipy.ndimage.gaussian_filter 完美支持 float32
        blurred = gaussian_filter(channel, sigma=sigma)
        
        # 锐化
        sharpened = channel + amount * (channel - blurred)
        return sharpened
    
    def _adaptive_sharpen(self, channel: np.ndarray, strength: float = 1.0, 
                          window_size: int = 5) -> np.ndarray:
        """
        自适应锐化 - (float32 兼容)
        """
        # 计算局部标准差（对比度）
        # scipy.ndimage.uniform_filter 完美支持 float32
        mean = uniform_filter(channel, size=window_size)
        mean_sq = uniform_filter(channel**2, size=window_size)
        variance = mean_sq - mean**2
        std_dev = np.sqrt(np.maximum(variance, 0))
        
        # 归一化标准差作为自适应权重
        std_max = std_dev.max()
        if std_max > 0:
            adaptive_weight = std_dev / (std_max + 1e-6)
        else:
            adaptive_weight = np.zeros_like(std_dev)
        
        # 拉普拉斯锐化
        kernel = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        laplacian = convolve(channel, kernel, mode='reflect')
        
        # 自适应应用锐化
        sharpened = channel + strength * adaptive_weight * laplacian
        return sharpened
    
    def _wiener_sharpen(self, channel: np.ndarray, noise_variance: float = 0.001, 
                        strength: float = 1.0) -> np.ndarray:
        """
        维纳滤波锐化 - (float32 兼容)
        """
        # FFT变换
        f_transform = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f_transform)
        
        # 创建理想的锐化滤波器（高通）
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        
        x = np.arange(cols) - ccol
        y = np.arange(rows) - crow
        X, Y = np.meshgrid(x, y)
        D = np.sqrt(X**2 + Y**2)
        D_max = D.max()
        
        # 维纳滤波器
        # H 是高频增强滤波器
        if D_max > 0:
            H = 1 + strength * (D / (D_max + 1e-6))
        else:
            H = np.ones_like(D)
            
        # 信号功率谱的粗略估计 |S(f)|^2
        signal_power = np.abs(f_shift)**2
        
        wiener_filter = H / (1 + noise_variance / (signal_power + 1e-6))
        
        # 应用滤波器
        f_filtered = f_shift * wiener_filter
        
        # 逆FFT
        f_ishift = np.fft.ifftshift(f_filtered)
        sharpened = np.fft.ifft2(f_ishift)
        sharpened = np.abs(sharpened) # 取实部
        
        return sharpened

# # sharpen.py
# import numpy as np
# import cv2
# from scipy.ndimage import convolve, gaussian_filter, uniform_filter


# class Sharpen:
#     """
#     锐化模块：提供多种锐化算法。
    
#     所有方法均假设输入为 np.float32 [0, 1] YUV图像，
#     输出也为 np.float32 [0, 1] YUV图像。
#     锐化操作将只在 Y (亮度) 通道上执行。
#     """
    
#     def execute(self, yuv_image: np.ndarray, algorithm: str = 'unsharp_mask', **kwargs) -> np.ndarray:
#         """
#         执行图像锐化。通常只对Y通道（亮度）进行锐化。
        
#         Args:
#             yuv_image: 输入的YUV图像 (np.float32, 范围 [0, 1])
#             algorithm: 锐化算法选择
#                 'laplacian': 拉普拉斯锐化
#                 'unsharp_mask': 非锐化掩蔽 (USM)
#                 'high_pass': 高通滤波锐化
#                 'sobel': Sobel边缘增强
#                 'gaussian_unsharp': 高斯USM (更平滑)
#                 'adaptive': 自适应锐化
#                 'wiener': 维纳滤波锐化
#             **kwargs: 算法特定参数
        
#         Returns:
#             锐化后的YUV图像 (np.float32, 范围 [0, 1])
#         """
#         print(f"Executing Sharpening using algorithm: {algorithm}")
        
#         # 确保输入是 float32
#         if yuv_image.dtype != np.float32:
#             print(f"警告: Sharpen 模块接收到非 float32 图像 (dtype: {yuv_image.dtype})。")
#             if yuv_image.max() > 1.0:
#                  yuv_image = yuv_image.astype(np.float32) / 65535.0
#             else:
#                  yuv_image = yuv_image.astype(np.float32)

#         # 只对Y通道进行锐化，保持UV通道不变
#         y_channel, u_channel, v_channel = cv2.split(yuv_image)
        
#         if algorithm == 'laplacian':
#             y_sharpened = self._laplacian_sharpen(y_channel, **kwargs)
#         elif algorithm == 'unsharp_mask':
#             y_sharpened = self._unsharp_mask(y_channel, **kwargs)
#         elif algorithm == 'high_pass':
#             y_sharpened = self._high_pass_sharpen(y_channel, **kwargs)
#         elif algorithm == 'sobel':
#             y_sharpened = self._sobel_sharpen(y_channel, **kwargs)
#         elif algorithm == 'gaussian_unsharp':
#             y_sharpened = self._gaussian_unsharp(y_channel, **kwargs)
#         elif algorithm == 'adaptive':
#             y_sharpened = self._adaptive_sharpen(y_channel, **kwargs)
#         elif algorithm == 'wiener':
#             y_sharpened = self._wiener_sharpen(y_channel, **kwargs)
#         else:
#             raise ValueError(f"Unknown sharpening algorithm: {algorithm}")
        
#         # 合并通道
#         return cv2.merge([y_sharpened, u_channel, v_channel])
    
#     def _laplacian_sharpen(self, channel: np.ndarray, strength: float = 1.0) -> np.ndarray:
#         """
#         拉普拉斯锐化 - (float32 兼容)
        
#         Args:
#             channel: Y 通道 (float32, [0, 1])
#             strength: 锐化强度 (0-2)
#         """
#         # 拉普拉斯核
#         kernel = np.array([
#             [0, -1, 0],
#             [-1, 4, -1],
#             [0, -1, 0]
#         ], dtype=np.float32)
        
#         # channel 已经是 float32 [0, 1]
        
#         # 计算拉普拉斯
#         laplacian = convolve(channel, kernel, mode='reflect')
        
#         # 添加回原图像
#         sharpened = channel + strength * laplacian
#         return np.clip(sharpened, 0, 1)
    
#     def _unsharp_mask(self, channel: np.ndarray, radius: float = 1.0, 
#                         amount: float = 1.0, threshold: float = 0) -> np.ndarray:
#         """
#         非锐化掩蔽 (USM) - (float32 兼容)
        
#         Args:
#             channel: Y 通道 (float32, [0, 1])
#             radius: 高斯模糊半径
#             amount: 锐化强度
#             threshold: 阈值, 避免锐化噪声 (范围应为 [0, 1], e.g., 5/255.0)
#         """
#         # channel 已经是 float32 [0, 1]
        
#         # 高斯模糊
#         # cv2.GaussianBlur 完美支持 float32 [0, 1]
#         blurred = cv2.GaussianBlur(channel, (0, 0), radius)
        
#         # 计算锐化掩模
#         mask = channel - blurred
        
#         # 应用阈值
#         if threshold > 0:
#             mask = np.where(np.abs(mask) >= threshold, mask, 0)
        
#         # 应用锐化
#         sharpened = channel + amount * mask
#         return np.clip(sharpened, 0, 1)
    
#     def _high_pass_sharpen(self, channel: np.ndarray, radius: int = 5, 
#                            strength: float = 1.5) -> np.ndarray:
#         """
#         高通滤波锐化 - (float32 兼容)
        
#         Args:
#             channel: Y 通道 (float32, [0, 1])
#             radius: 滤波半径
#             strength: 锐化强度
#         """
#         # channel 已经是 float32 [0, 1]
        
#         # 低通滤波（模糊）
#         # cv2.boxFilter 支持 float32 [0, 1] (ddepth=-1)
#         low_pass = cv2.boxFilter(channel, -1, (radius, radius))
        
#         # 高通 = 原图 - 低通
#         high_pass = channel - low_pass
        
#         # 增强高频
#         sharpened = channel + strength * high_pass
#         return np.clip(sharpened, 0, 1)
    
#     def _sobel_sharpen(self, channel: np.ndarray, strength: float = 1.0) -> np.ndarray:
#         """
#         Sobel边缘增强 - (float32 兼容)
        
#         Args:
#             channel: Y 通道 (float32, [0, 1])
#             strength: 锐化强度
#         """
#         # channel 已经是 float32 [0, 1]
        
#         # Sobel算子
#         # cv2.Sobel 支持 float32 [0, 1] 输入,
#         # 并通过 cv2.CV_32F 指定 float32 输出
#         sobel_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
#         sobel_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        
#         # 梯度幅值
#         gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        
#         # 添加边缘信息
#         sharpened = channel + strength * gradient
#         return np.clip(sharpened, 0, 1)
    
#     def _gaussian_unsharp(self, channel: np.ndarray, sigma: float = 1.0, 
#                           amount: float = 1.5) -> np.ndarray:
#         """
#         高斯非锐化掩蔽 - (float32 兼容)
        
#         Args:
#             channel: Y 通道 (float32, [0, 1])
#             sigma: 高斯标准差
#             amount: 锐化强度
#         """
#         # channel 已经是 float32 [0, 1]
        
#         # 高斯模糊
#         # scipy.ndimage.gaussian_filter 完美支持 float32 [0, 1]
#         blurred = gaussian_filter(channel, sigma=sigma)
        
#         # 锐化
#         sharpened = channel + amount * (channel - blurred)
#         return np.clip(sharpened, 0, 1)
    
#     def _adaptive_sharpen(self, channel: np.ndarray, strength: float = 1.0, 
#                           window_size: int = 5) -> np.ndarray:
#         """
#         自适应锐化 - (float32 兼容)
        
#         Args:
#             channel: Y 通道 (float32, [0, 1])
#             strength: 基础锐化强度
#             window_size: 局部窗口大小
#         """
#         # channel 已经是 float32 [0, 1]
        
#         # 计算局部标准差（对比度）
#         # scipy.ndimage.uniform_filter 完美支持 float32 [0, 1]
#         mean = uniform_filter(channel, size=window_size)
#         mean_sq = uniform_filter(channel**2, size=window_size)
#         variance = mean_sq - mean**2
#         std_dev = np.sqrt(np.maximum(variance, 0))
        
#         # 归一化标准差作为自适应权重
#         std_max = std_dev.max()
#         if std_max > 0:
#             adaptive_weight = std_dev / (std_max + 1e-6)
#         else:
#             adaptive_weight = np.zeros_like(std_dev)
        
#         # 拉普拉斯锐化
#         kernel = np.array([
#             [0, -1, 0],
#             [-1, 4, -1],
#             [0, -1, 0]
#         ], dtype=np.float32)
#         laplacian = convolve(channel, kernel, mode='reflect')
        
#         # 自适应应用锐化
#         sharpened = channel + strength * adaptive_weight * laplacian
#         return np.clip(sharpened, 0, 1)
    
#     def _wiener_sharpen(self, channel: np.ndarray, noise_variance: float = 0.001, 
#                         strength: float = 1.0) -> np.ndarray:
#         """
#         维纳滤波锐化 - (float32 兼容)
        
#         Args:
#             channel: Y 通道 (float32, [0, 1])
#             noise_variance: 噪声方差估计 (应在 [0, 1] 范围内)
#             strength: 锐化强度
#         """
#         # channel 已经是 float32 [0, 1]
        
#         # FFT变换
#         f_transform = np.fft.fft2(channel)
#         f_shift = np.fft.fftshift(f_transform)
        
#         # 创建理想的锐化滤波器（高通）
#         rows, cols = channel.shape
#         crow, ccol = rows // 2, cols // 2
        
#         x = np.arange(cols) - ccol
#         y = np.arange(rows) - crow
#         X, Y = np.meshgrid(x, y)
#         D = np.sqrt(X**2 + Y**2)
#         D_max = D.max()
        
#         # 维纳滤波器
#         # H 是高频增强滤波器
#         if D_max > 0:
#             H = 1 + strength * (D / (D_max + 1e-6))
#         else:
#             H = np.ones_like(D)
            
#         # 信号功率谱的粗略估计 |S(f)|^2
#         signal_power = np.abs(f_shift)**2
        
#         # 维纳滤波器: H_wiener = H * [ |S(f)|^2 / (|S(f)|^2 + |N(f)|^2) ]
#         # |N(f)|^2 是噪声功率谱，我们用估计的 noise_variance 
#         # H * [ 1 / (1 + |N(f)|^2 / |S(f)|^2) ]
#         # 注意: 原代码中的 wiener_filter 形式是 H / (1 + N/S)，这是对的
#         wiener_filter = H / (1 + noise_variance / (signal_power + 1e-6))
        
#         # 应用滤波器
#         f_filtered = f_shift * wiener_filter
        
#         # 逆FFT
#         f_ishift = np.fft.ifftshift(f_filtered)
#         sharpened = np.fft.ifft2(f_ishift)
#         sharpened = np.abs(sharpened) # 取实部
        
#         return np.clip(sharpened, 0, 1)
