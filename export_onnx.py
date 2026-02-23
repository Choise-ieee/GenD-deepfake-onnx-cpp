"""
导出 GenD 模型为 ONNX 格式
"""

import sys
from pathlib import Path
import argparse

import torch
import numpy as np

THIS = Path(__file__).resolve()
ROOT = THIS.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hf.modeling_gend import GenD as GenD_HF


def export_onnx(model_id: str, output_path: str, opset: int = 14):
    """导出模型为ONNX格式"""
    print(f"Loading model: {model_id}")
    model = GenD_HF.from_pretrained(model_id)
    model.eval()
    
    # 获取预处理函数
    preproc = model.feature_extractor.preprocess
    
    # 创建dummy输入 (batch_size=1, channels=3, height=224, width=224)
    # CLIP模型的输入尺寸通常是224x224
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"Exporting to ONNX: {output_path}")
    print(f"Opset version: {opset}")
    print(f"Input shape: {dummy_input.shape}")
    
    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
    )
    
    print(f"✅ Exported to: {output_path}")
    
    # 验证
    print("\nValidating ONNX model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model is valid")
    
    # 测试推理
    print("\nTesting ONNX inference...")
    import onnxruntime as ort
    
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    inputs = {sess.get_inputs()[0].name: dummy_input.numpy()}
    outputs = sess.run(None, inputs)
    
    print(f"Output shape: {outputs[0].shape}")
    probs = torch.softmax(torch.from_numpy(outputs[0]), dim=-1)
    print(f"Sample output (probs): {probs[0].numpy()}")
    
    # 对比PyTorch输出
    with torch.no_grad():
        pytorch_out = model(dummy_input)
        pytorch_probs = torch.softmax(pytorch_out, dim=-1)
    
    diff = np.abs(probs.numpy() - pytorch_probs.numpy()).max()
    print(f"Max difference from PyTorch: {diff:.6f}")
    
    if diff < 1e-4:
        print("✅ ONNX export successful! Outputs match PyTorch.")
    else:
        print(f"⚠️ Warning: Outputs differ by {diff}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GenD model to ONNX")
    parser.add_argument("--model", type=str, default="yermandy/GenD_CLIP_L_14", help="Model ID")
    parser.add_argument("--output", type=str, default="gend.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()
    
    export_onnx(args.model, args.output, args.opset)
