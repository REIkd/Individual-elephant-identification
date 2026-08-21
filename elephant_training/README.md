# 大象个体识别 · 训练部署包

本目录包含**完整训练流程**，可单独打包给他人部署，无需拷贝整个识别项目。

## 目录结构

```
elephant_training/
  dataset/              ← 训练数据（按象名分子文件夹 + 可选 VOC XML）
  elephant_net.py       ← 模型骨干（EfficientNet-V2-S / ResNet50）
  train.py                ← 标准训练（VOC 主体框裁剪）
  train_industrial.py     ← 工业化训练（主体框 + 特征部位框）
  finetune.py             ← 增量微调（新照片加入后）
  import_new_photos.py    ← 新照片 YOLO 裁象体并写入 dataset
  compare_training_runs.py  ← 对比两次训练权重
  plot_training_convergence.py  ← 从 train.log 生成收敛曲线
  classifier.py           ← 单张图评测（对比脚本用）
  yolo_crop.py            ← YOLO 裁象体工具
  paths.py                ← 路径与环境变量
  requirements.txt
  reports/                ← 训练报告与曲线输出
```

## 环境准备

### Windows

```powershell
cd elephant_training
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# GPU 版 PyTorch 请按 https://pytorch.org 选择 CUDA 版本安装
```

### Linux

```bash
cd elephant_training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

也可运行：`scripts/setup.ps1`（Windows）或 `scripts/setup.sh`（Linux）。

## 数据格式

### 方式 A：标准训练（`train.py`）

```
dataset/
  安妮/
    安妮-001.jpg
    安妮-001.xml    ← VOC 标注，第 1 个框为象体
  玛丽亚/
    ...
```

### 方式 B：工业化训练（`train_industrial.py`，推荐）

```
dataset/
  印东（身体特征框）/
    印东-350.jpg
    印东-350.xml    ← 第 1 框：主体；后续框：鼻子/腿等特征
```

文件夹名可带「（身体特征框）」后缀，程序会自动解析象名。

## 训练命令

```bash
# 1. 标准训练（EfficientNet-V2-S，两阶段：分类头 → 全模型微调）
python train.py

# 2. 工业化训练（主体+特征联合，验证仅用主体框）
python train_industrial.py

# 3. 新照片增量微调
python import_new_photos.py --src ../新拍照片
python finetune.py --epochs 20

# 4. 对比两次权重
python compare_training_runs.py --model_a run_a.pth --model_b run_b.pth --test_dir dataset

# 5. 绘制收敛曲线（需 train.log）
python plot_training_convergence.py --log train.log --model best_elephant_model.pth
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `best_elephant_model.pth` | 最佳验证准确率权重（部署到云端/Pi） |
| `class_names.json` | 类别名列表（训练时自动生成） |
| `training_curves.png` | 损失/准确率曲线 |
| `train.log` | 建议 `python train.py 2>&1 \| tee train.log` 保存 |

## 环境变量（可选）

| 变量 | 默认 | 说明 |
|------|------|------|
| `ELEPHANT_ARCH` | `efficientnet_v2_s` | 骨干网络 |
| `ELEPHANT_BATCH` | `16` | batch size |
| `ELEPHANT_DATASET` | `./dataset` | 数据集目录 |
| `ELEPHANT_MODEL` | `./best_elephant_model.pth` | 输出权重路径 |

## 部署到识别服务

训练完成后，将 `best_elephant_model.pth` 与 `class_names.json` 复制到云端项目根目录，替换旧权重并重启 `cloud_server.py` 即可。

## 与主项目关系

主项目根目录的 `train.py` 等脚本为**兼容入口**，会自动转调本目录。给他人部署时，**只需打包 `elephant_training/` 文件夹及 `dataset/` 数据**即可。
