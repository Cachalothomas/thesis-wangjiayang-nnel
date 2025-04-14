from pathlib import Path

import torch

# logger预设
from WJYML.base_logger import logger

def save_torch_in_onnx(torch_model: torch.nn.Module, path: Path, input_size: int):
    """将torch模型转换成onnx格式并序列化到到硬盘"""
    onnx_path = path.with_suffix(".onnx")
    try:
        # 使用动态轴指定 batch size 维度为动态维度
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        # 使用 with 语句确保资源的正确释放
        with torch.no_grad():
            # 创建一个虚拟输入，指定批次大小为 1，其余维度与模型输入相同
            dummy_input = torch.randn(1, input_size, device="cuda:0")
            # 将 PyTorch 模型导出为 ONNX 模型，并指定输出名字为 "output"
            torch.onnx.export(
                torch_model,
                dummy_input,
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                opset_version=11,
            )
    except Exception as e:
        logger.warning(e)
        torch.cuda.empty_cache()

    else:
        pass
        # logger.info(f"finish export onnx model in {onnx_path}")


def save_torch_in_pth(torch_model: torch.nn.Module, path: Path):
    """将torch模型转换成pth格式并序列化到到硬盘"""
    pth_path = path.with_suffix(".pth")
    try:
        torch.save(torch_model, str(pth_path))

    except Exception as e:
        logger.warning(e)
        logger.warning("might be RuntimeError: CUDA out of memory.")
    else:
        pass
        # logger.info(f"finish export pth model in {pth_path}")
