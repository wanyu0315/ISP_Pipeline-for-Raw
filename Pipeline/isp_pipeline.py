# isp_pipeline.py
from typing import List, Any, Dict
import copy
import rawpy
import numpy as np
import os


class ISPPipeline:
    """
    一个灵活的ISP处理管道。
    通过在初始化时传入不同的处理模块实例来组装。
    """
    def __init__(self, modules: List[Any]):
        """
        初始化ISP管道。
        
        Args:
            modules: 一个包含ISP处理模块实例的列表。
                     每个模块必须有一个 `execute` 方法。
        """
        self.modules = modules
        self.module_param_aliases = {
            "whitebalanceraw": "whitebalance",
        }

    @staticmethod
    def _safe_param_value(value: Any) -> Any:
        """将模块参数压缩为可写入 JSON 的只读科研遥测。"""
        if isinstance(value, np.ndarray):
            return {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                str(key): ISPPipeline._safe_param_value(item)
                for key, item in value.items()
                if key != "defect_map"
            }
        if isinstance(value, (list, tuple)):
            return [ISPPipeline._safe_param_value(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    @staticmethod
    def _array_summary(value: Any) -> Dict[str, Any]:
        if not isinstance(value, np.ndarray) or value.size == 0:
            return {}
        array = value.astype(np.float64, copy=False)
        finite = np.isfinite(array)
        if not np.any(finite):
            return {"finite_fraction": 0.0}
        valid = array[finite]
        return {
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "finite_fraction": float(np.mean(finite)),
            "clipped_low_fraction": float(np.mean(valid <= 0.0)),
            "clipped_high_fraction": float(np.mean(valid >= 1.0)),
        }

    @classmethod
    def _module_telemetry(
        cls,
        module: Any,
        module_name: str,
        module_params: Dict[str, Any],
        input_data: Any,
        output_data: Any,
    ) -> Dict[str, Any]:
        telemetry = {
            "module": module.__class__.__name__,
            "module_key": module_name,
            "configured_params": cls._safe_param_value(module_params),
            "input_summary": cls._array_summary(input_data),
            "output_summary": cls._array_summary(output_data),
        }
        if isinstance(input_data, np.ndarray) and isinstance(output_data, np.ndarray):
            if input_data.shape == output_data.shape:
                residual = output_data.astype(np.float64) - input_data.astype(np.float64)
                telemetry["residual_summary"] = cls._array_summary(residual)
                telemetry["changed_fraction"] = float(np.mean(residual != 0.0))
        applied = getattr(module, "last_probe_telemetry", None)
        if isinstance(applied, dict):
            telemetry["applied_state"] = copy.deepcopy(cls._safe_param_value(applied))
        return telemetry

    def process(self, raw_file_path: str, params: Dict[str, Any] = None) -> np.ndarray:
        """
        执行完整的ISP处理流程。
        Args:
            raw_file_path: RAW图像文件的路径。
            params: 一个字典，包含每个步骤所需的参数。
                    键是模块类的名称(小写)，值是传递给该模块 `execute` 方法的参数字典。
                    例如:
                    {
                        'demosaic': {'algorithm': 'AHD'},
                        'whitebalance': {'algorithm': 'perfect_reflector', 'percentile': 99.8}
                    }
        Returns:
            处理完成后的最终图像 (Numpy array)。
        """
        if params is None:
            params = {}

        print("--- ISP Pipeline Start ---")
        
  
       # 初始数据就是文件路径字符串
        processed_data = raw_file_path

        # 2. 按顺序执行每个模块
        upstream_telemetry = None
        for module in self.modules:
            module_name = module.__class__.__name__.lower()
            param_key = self.module_param_aliases.get(module_name, module_name)
            raw_module_params = params.get(param_key, {})
            module_params = dict(raw_module_params) if isinstance(raw_module_params, dict) else {}

            # 自动化配置会带控制字段 enabled，但它不属于各模块 execute() 的算法参数。
            module_params.pop('enabled', None)

            if getattr(module, "is_pipeline_probe", False):
                module_params["upstream_telemetry"] = upstream_telemetry

            # 后续模块会接收前一个模块处理后的Numpy数组
            input_data = processed_data
            processed_data = module.execute(processed_data, **module_params)
            if not getattr(module, "is_pipeline_probe", False):
                upstream_telemetry = self._module_telemetry(
                    module,
                    param_key,
                    raw_module_params if isinstance(raw_module_params, dict) else {},
                    input_data,
                    processed_data,
                )
        
        print("--- ISP Pipeline Finished ---")
        return processed_data
