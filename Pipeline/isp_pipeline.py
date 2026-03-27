# isp_pipeline.py
from typing import List, Any, Dict
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
        for module in self.modules:
            module_name = module.__class__.__name__.lower()
            module_params = params.get(module_name, {})
            
            # 后续模块会接收前一个模块处理后的Numpy数组
            processed_data = module.execute(processed_data, **module_params)
        
        print("--- ISP Pipeline Finished ---")
        return processed_data