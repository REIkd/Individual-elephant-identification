"""
模型导出工具
将训练好的PyTorch模型导出为ONNX格式，便于部署
"""

import torch
import torch.nn as nn
from torchvision import models
import json

def export_to_onnx(model_path='best_elephant_model.pth',
                   class_names_path='class_names.json',
                   output_path='elephant_model.onnx'):
    """导出模型为ONNX格式"""
    
    device = torch.device('cpu')
    
    # 加载类别信息
    with open(class_names_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    
    num_classes = len(class_names)
    
    # 构建模型
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 导出ONNX
    print(f"正在导出模型到 {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ 模型已导出到 {output_path}")
    print(f"  - 输入维度: (batch_size, 3, 224, 224)")
    print(f"  - 输出维度: (batch_size, {num_classes})")
    print(f"  - 类别数量: {num_classes}")
    
    # 验证导出的模型
    try:
        import onnx
        import onnxruntime as ort
        
        # 检查模型
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("\n✓ ONNX模型验证通过")
        
        # 测试推理
        ort_session = ort.InferenceSession(output_path)
        
        # 创建测试输入
        import numpy as np
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # 运行推理
        outputs = ort_session.run(None, {'input': test_input})
        print(f"✓ ONNX推理测试成功，输出形状: {outputs[0].shape}")
        
    except ImportError:
        print("\n提示: 安装 onnx 和 onnxruntime 可以验证导出的模型")
        print("运行: pip install onnx onnxruntime")
    except Exception as e:
        print(f"\n警告: 模型验证失败 - {e}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='导出模型为ONNX格式')
    parser.add_argument('--model', type=str, default='best_elephant_model.pth',
                       help='PyTorch模型路径')
    parser.add_argument('--classes', type=str, default='class_names.json',
                       help='类别名称文件')
    parser.add_argument('--output', type=str, default='elephant_model.onnx',
                       help='输出的ONNX文件路径')
    
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.classes, args.output)
