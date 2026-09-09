"""Compact, deterministic operator metrics for ISP probe telemetry."""
from __future__ import annotations

from typing import Any

import numpy as np


EPS = np.finfo(np.float64).eps


def _sampleView(array: np.ndarray, maxSide: int = 256) -> tuple[np.ndarray, int]:
    height, width = array.shape[:2]
    step = max(1, int(np.ceil(max(height, width) / maxSide)))
    return array[::step, ::step], step


def _gradientMagnitude(channel: np.ndarray) -> np.ndarray:
    values = np.asarray(channel, dtype=np.float64)
    dy, dx = np.gradient(values)
    return np.sqrt(dx * dx + dy * dy)


def _laplacianProxy(channel: np.ndarray) -> np.ndarray:
    values = np.asarray(channel, dtype=np.float64)
    return (
        -4.0 * values
        + np.roll(values, 1, axis=0)
        + np.roll(values, -1, axis=0)
        + np.roll(values, 1, axis=1)
        + np.roll(values, -1, axis=1)
    )


def operatorActivity(
    before: np.ndarray,
    after: np.ndarray,
    primaryChannel: int | None = None,
) -> dict[str, Any]:
    """Summarize operator dose on a deterministic spatial sample."""
    if before.shape != after.shape or before.size == 0:
        return {"metric_status": "shape_mismatch"}
    beforeSample, step = _sampleView(np.asarray(before))
    afterSample, _ = _sampleView(np.asarray(after))
    residual = afterSample.astype(np.float64) - beforeSample.astype(np.float64)
    absolute = np.abs(residual)
    metrics: dict[str, Any] = {
        "spatial_sample_step": int(step),
        "residual_mean": float(np.mean(residual)),
        "residual_abs_mean": float(np.mean(absolute)),
        "residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "residual_abs_max": float(np.max(absolute)),
        "changed_fraction_sampled": float(np.mean(absolute > 1e-12)),
        "output_below_zero_fraction_sampled": float(np.mean(afterSample < 0.0)),
        "output_above_one_fraction_sampled": float(np.mean(afterSample > 1.0)),
    }
    if beforeSample.ndim == 3:
        metrics["channel_residual_abs_mean"] = [
            float(np.mean(absolute[..., index])) for index in range(beforeSample.shape[2])
        ]
        metrics["channel_dc_delta"] = [
            float(np.mean(residual[..., index])) for index in range(beforeSample.shape[2])
        ]
        channelIndex = 0 if primaryChannel is None else int(primaryChannel)
        inputPrimary = beforeSample[..., channelIndex]
        outputPrimary = afterSample[..., channelIndex]
    else:
        inputPrimary = beforeSample
        outputPrimary = afterSample

    gradientBefore = _gradientMagnitude(inputPrimary)
    gradientAfter = _gradientMagnitude(outputPrimary)
    gradientBeforeMean = float(np.mean(gradientBefore))
    gradientAfterMean = float(np.mean(gradientAfter))
    highBefore = float(np.sqrt(np.mean(np.square(_laplacianProxy(inputPrimary)))))
    highAfter = float(np.sqrt(np.mean(np.square(_laplacianProxy(outputPrimary)))))
    threshold = float(np.quantile(gradientBefore, 0.75))
    edgeMask = gradientBefore >= threshold
    primaryResidual = np.abs(outputPrimary.astype(np.float64) - inputPrimary.astype(np.float64))
    metrics.update({
        "primary_gradient_energy_before": gradientBeforeMean,
        "primary_gradient_energy_after": gradientAfterMean,
        "primary_gradient_gain": gradientAfterMean / max(gradientBeforeMean, EPS),
        "primary_high_frequency_rms_before": highBefore,
        "primary_high_frequency_rms_after": highAfter,
        "primary_high_frequency_gain": highAfter / max(highBefore, EPS),
        "edge_threshold_p75": threshold,
        "edge_residual_abs_mean": float(np.mean(primaryResidual[edgeMask])) if np.any(edgeMask) else 0.0,
        "flat_residual_abs_mean": float(np.mean(primaryResidual[~edgeMask])) if np.any(~edgeMask) else 0.0,
    })
    return metrics


def matrixMetrics(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    singular = np.linalg.svd(values, compute_uv=False)
    return {
        "matrix": values.tolist(),
        "matrix_determinant": float(np.linalg.det(values)),
        "matrix_condition_number": float(np.linalg.cond(values)),
        "matrix_singular_values": singular.tolist(),
        "matrix_operator_gain": float(singular[0]),
        "matrix_row_sums": np.sum(values, axis=1).tolist(),
        "matrix_off_diagonal_l1": float(np.sum(np.abs(values - np.diag(np.diag(values))))),
    }


def roundTripMetrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    if original.shape != reconstructed.shape:
        return {"roundtrip_rms": float("nan"), "roundtrip_abs_max": float("nan")}
    residual = reconstructed.astype(np.float64) - original.astype(np.float64)
    return {
        "roundtrip_bias": float(np.mean(residual)),
        "roundtrip_abs_mean": float(np.mean(np.abs(residual))),
        "roundtrip_rms": float(np.sqrt(np.mean(residual * residual))),
        "roundtrip_abs_max": float(np.max(np.abs(residual))),
    }
