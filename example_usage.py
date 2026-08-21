"""
使用示例 - 如何在您的代码中集成大象识别功能
"""

from predict import ElephantClassifier
from pathlib import Path

# ==================== 示例 1: 基础使用 ====================
def example_basic():
    """最简单的使用方式"""
    print("=" * 60)
    print("示例 1: 基础使用")
    print("=" * 60)
    
    # 初始化分类器
    classifier = ElephantClassifier(
        model_path='best_elephant_model.pth',
        class_names_path='class_names.json'
    )
    
    # 识别单张图片
    image_path = "威武/威武-1.jpg"
    
    if Path(image_path).exists():
        elephant_name, confidence, all_probs = classifier.predict(image_path)
        
        print(f"\n图片: {image_path}")
        print(f"识别结果: {elephant_name}")
        print(f"置信度: {confidence:.2f}%")
    else:
        print(f"图片不存在: {image_path}")

# ==================== 示例 2: 批量处理 ====================
def example_batch():
    """批量处理多张图片"""
    print("\n" + "=" * 60)
    print("示例 2: 批量处理")
    print("=" * 60)
    
    classifier = ElephantClassifier()
    
    # 获取测试图片列表
    test_images = []
    for folder in Path('.').iterdir():
        if folder.is_dir() and not folder.name.startswith('.'):
            images = list(folder.glob('*.jpg'))[:2]  # 每个类别取2张
            test_images.extend(images)
    
    # 批量识别
    results = []
    for img_path in test_images[:5]:  # 只处理前5张作为示例
        elephant_name, confidence, _ = classifier.predict(str(img_path))
        results.append({
            'image': img_path.name,
            'predicted': elephant_name,
            'confidence': confidence
        })
        print(f"✓ {img_path.name}: {elephant_name} ({confidence:.1f}%)")
    
    return results

# ==================== 示例 3: 置信度过滤 ====================
def example_confidence_filter():
    """根据置信度过滤结果"""
    print("\n" + "=" * 60)
    print("示例 3: 置信度过滤")
    print("=" * 60)
    
    classifier = ElephantClassifier()
    
    # 设置置信度阈值
    CONFIDENCE_THRESHOLD = 80.0
    
    # 获取测试图片
    test_images = list(Path('威武').glob('*.jpg'))[:5]
    
    high_confidence_results = []
    low_confidence_results = []
    
    for img_path in test_images:
        elephant_name, confidence, _ = classifier.predict(str(img_path))
        
        if confidence >= CONFIDENCE_THRESHOLD:
            high_confidence_results.append((img_path.name, elephant_name, confidence))
            print(f"🟢 {img_path.name}: {elephant_name} ({confidence:.1f}%)")
        else:
            low_confidence_results.append((img_path.name, elephant_name, confidence))
            print(f"🔴 {img_path.name}: {elephant_name} ({confidence:.1f}%) - 需要人工确认")
    
    print(f"\n高置信度: {len(high_confidence_results)}")
    print(f"低置信度: {len(low_confidence_results)}")

# ==================== 示例 4: Top-K 结果 ====================
def example_topk():
    """获取前K个最可能的结果"""
    print("\n" + "=" * 60)
    print("示例 4: Top-K 结果")
    print("=" * 60)
    
    classifier = ElephantClassifier()
    
    # 获取一张测试图片
    test_images = list(Path('威武').glob('*.jpg'))
    
    if test_images:
        img_path = test_images[0]
        elephant_name, confidence, all_probs = classifier.predict(str(img_path))
        
        print(f"\n图片: {img_path.name}")
        print(f"\nTop-3 预测结果:")
        
        # 排序获取Top-3
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        
        for i, (name, prob) in enumerate(sorted_probs[:3], 1):
            print(f"{i}. {name}: {prob:.2f}%")

# ==================== 示例 5: 集成到监控系统 ====================
def example_monitoring_system():
    """模拟集成到监控系统"""
    print("\n" + "=" * 60)
    print("示例 5: 监控系统集成")
    print("=" * 60)
    
    import time
    from datetime import datetime
    
    classifier = ElephantClassifier()
    
    # 模拟摄像头捕获的图片队列
    captured_images = list(Path('威武').glob('*.jpg'))[:3]
    
    print("\n模拟实时监控...")
    
    for img_path in captured_images:
        # 记录时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 识别
        elephant_name, confidence, _ = classifier.predict(str(img_path))
        
        # 记录日志
        log_entry = {
            'timestamp': timestamp,
            'camera_id': 'CAM-01',
            'elephant': elephant_name,
            'confidence': confidence,
            'image': str(img_path)
        }
        
        print(f"[{timestamp}] 检测到: {elephant_name} (置信度: {confidence:.1f}%)")
        
        # 如果是高价值目标，触发告警
        if elephant_name in ['威武', '威望']:
            print(f"  ⚠️  VIP大象出现，触发通知")
        
        time.sleep(0.5)  # 模拟延迟

# ==================== 示例 6: 性能统计 ====================
def example_performance():
    """统计模型性能"""
    print("\n" + "=" * 60)
    print("示例 6: 性能统计")
    print("=" * 60)
    
    import time
    
    classifier = ElephantClassifier()
    
    # 获取测试图片
    test_images = list(Path('威武').glob('*.jpg'))[:10]
    
    # 测量处理时间
    times = []
    
    print(f"\n处理 {len(test_images)} 张图片...")
    
    for img_path in test_images:
        start_time = time.time()
        elephant_name, confidence, _ = classifier.predict(str(img_path))
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000  # 转换为毫秒
        times.append(processing_time)
    
    # 统计
    import numpy as np
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\n性能统计:")
    print(f"  平均处理时间: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  最快: {min_time:.2f} ms")
    print(f"  最慢: {max_time:.2f} ms")
    print(f"  理论FPS: {1000/avg_time:.1f}")

# ==================== 示例 7: 错误处理 ====================
def example_error_handling():
    """正确的错误处理"""
    print("\n" + "=" * 60)
    print("示例 7: 错误处理")
    print("=" * 60)
    
    try:
        # 初始化分类器
        classifier = ElephantClassifier()
        
        # 测试不存在的图片
        test_cases = [
            "存在的图片.jpg",
            "不存在的图片.jpg",
            "威武/威武-1.jpg"
        ]
        
        for img_path in test_cases:
            try:
                if Path(img_path).exists():
                    elephant_name, confidence, _ = classifier.predict(img_path)
                    print(f"✓ {img_path}: {elephant_name} ({confidence:.1f}%)")
                else:
                    print(f"✗ {img_path}: 文件不存在")
                    
            except Exception as e:
                print(f"✗ {img_path}: 处理失败 - {e}")
    
    except FileNotFoundError:
        print("错误: 模型文件不存在，请先训练模型")
    except Exception as e:
        print(f"错误: {e}")

# ==================== 主函数 ====================
def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("        大象识别系统 - 使用示例集合")
    print("="*60)
    
    # 检查模型是否存在
    if not Path('best_elephant_model.pth').exists():
        print("\n错误: 找不到训练好的模型！")
        print("请先运行: python train.py")
        return
    
    # 运行示例
    examples = [
        ("基础使用", example_basic),
        ("批量处理", example_batch),
        ("置信度过滤", example_confidence_filter),
        ("Top-K结果", example_topk),
        ("监控系统集成", example_monitoring_system),
        ("性能统计", example_performance),
        ("错误处理", example_error_handling)
    ]
    
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    print("0. 运行全部示例")
    
    choice = input("\n请选择要运行的示例 (0-7): ").strip()
    
    if choice == '0':
        # 运行所有示例
        for name, func in examples:
            try:
                func()
                input("\n按回车继续...")
            except Exception as e:
                print(f"示例 {name} 运行失败: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        # 运行选中的示例
        name, func = examples[int(choice) - 1]
        try:
            func()
        except Exception as e:
            print(f"示例运行失败: {e}")
    else:
        print("无效的选择")

if __name__ == '__main__':
    main()
