# Ecological Informatics 稿件修订说明

> 对应文件：`manuscript_ecological_informatics/main.tex`  
> 若你本地有 `paper.txt`，建议以 `main.tex` 为准（LaTeX 为投稿正式稿）

---

## 一、标题（定稿 — 方案 C）

**A fine-grained re-identification approach for Asian elephants with morphological feature-region supervision**

中文：**基于形态学特征区域监督的亚洲象细粒度个体重识别方法**

| 语法成分 | 英文 | 作用 |
|----------|------|------|
| **主语** | a fine-grained re-identification **approach** | 重识别**方法/技术**（主贡献） |
| **对象** | for Asian elephants | 物种 |
| **修饰/创新点** | with morphological feature-region **supervision** | 特征标注作为监督机制 |

---

## 二、叙事重心（勿混淆）

| 研究重点（主贡献） | 表述方式 | 非重点 |
|--------------------|----------|--------|
| **细粒度 re-ID 方法** | 标题主语 approach | camera-trap 工程 |
| **形态学特征区域监督** | with … supervision | 纯标注工具论文 |
| EfficientNet-V2-S + 联合训练 | Methods § Feature-region supervised re-ID model | 两阶段架构进标题 |
| 三次 run 泛化对比 | Results 主结论 | 99.85% 库内 alone |

---

## 三、Highlights（已更新）

1. Morphological feature-region annotation for elephant re-ID
2. Body-only vs body+feature: library 99.85% vs field failure → field gain
3. Three-run comparison: annotation governs generalisation
4. Detect--classify--stabilise inference pipeline
5. IR camera-trap uploads as ecological use case  

---

## 四、三次训练迭代（已写入 main.tex）

| 次序 | 图像库 val | 实地/YOLO 识别 | 结论 |
|------|------------|----------------|------|
| **Run 1** | 尚可 | **很低** | 仅整体框，只认库内风格 |
| **Run 2** | **99.85%** | **仍很低** | 库内极高 ≠ 实地可用 |
| **Run 3** | `[待填]` | **显著提升** | 特征框联合训练 → **泛化能力** |

**论文核心论点（已写入 Abstract / Results / Discussion）：**
> 整体框训练能在参考图像库验证集上达到很高准确率，但对实地拍摄、YOLO 裁剪泛化差；加入特征部位框联合训练后，实地识别正确率显著提升，说明**标注方式 + 训练模式**对模型泛化有实质贡献。

**Table `tab:generalization` 待填数值（最重要）：**
- Run 2 vs Run 3 在同一批实地照片上的 top-1 %
- Run 2 vs Run 3 在同一批 YOLO 视频帧上的 top-1 %
- 样本量 N、是否同一 17 象 closed-set

**一键对比命令（有权重后）：**
```bash
cd elephant_training
python compare_training_runs.py --model_feature run3.pth --model_body run2_body_only.pth --test_dir dataset
python plot_training_convergence.py --log train.log --model best_elephant_model.pth
```

---

## 五、还需补充的实验（投稿前建议）

### 必做（支撑三次训练叙事）

| # | 实验 | 目的 | 产出 |
|---|------|------|------|
| E1 | **Run 2 vs Run 3 实地准确率** | **论文主结论**：特征框训练提升泛化 | Table `tab:generalization` ← **最优先** |
| E2 | **Run 2 vs Run 3 YOLO 裁剪准确率** | 与部署场景一致 | 同上表第二行 |
| E3 | **17 象逐类准确率** | 主结果 Table 2 | `predict.py` / 批量评估 → `tab:placeholder` |
| E4 | **混淆矩阵** | 看易混个体对 | 17×17 矩阵（body val） |

### 强烈建议（re-ID + 野外应用）

| # | 实验 | 目的 |
|---|------|------|
| E5 | **YOLO 裁剪端到端准确率** | 验证集用 YOLO 框而非 VOC 框，测 domain shift |
| E6 | **track 锁定前后 flicker** | 有/无 identity lock、有/无 `allowed_elephants` 的标签切换次数 |
| E7 | **红外野外子集人工一致性** | 抽样 N 段 UOVision 视频，专家标注 vs 系统输出 |
| E8 | **Run 3 消融** | 仅 body 训练 vs body+feature（同 epoch 预算） |

### 可选（Discussion 加分）

| # | 实验 |
|---|------|
| E9 | 候选过滤：17 类全开 vs 7 象白名单的准确率/误识 |
| E10 | 开放集拒识（未知个体）初步曲线 |
| E11 | 与 ResNet50 / 纯 metric learning 基线对比 |

---

## 六、还需准备的矢量图（PDF/SVG，Ecological Informatics 常用）

| 图号 | 内容 | 状态 | 制作方式 |
|------|------|------|----------|
| **Fig 1** | **Re-ID 流程图**：camera-trap 视频 → YOLO+tracker → classifier → track lock → 标注输出 | ❌ 占位 | 建议 Inkscape/draw.io → `figures/pipeline.pdf` |
| **Fig 2** | **VOC 标注示意**：同一张图上的 body 框 + feature 框（鼻子/腿等） | ❌ 缺 | 选 1 张脱敏参考图 + 矢量框 |
| **Fig 3** | **库内 vs 实地准确率对比**（Run2 vs Run3 分组柱状图） | ❌ 缺 | 有 E1/E2 数值后绘制 ← **主图** |
| **Fig 4** | 三次训练收敛曲线（可选，次要） | ⚠️ 部分 | `training_curves.svg` 仅 Run 2 |
| **Fig 5** | **混淆矩阵**（库内 Run2；实地 Run2 vs Run3 各一） | ❌ 缺 | 实地混淆矩阵更能说明问题 |
| **Fig 6** | **特征空间 t-SNE/PCA**（验证集 embedding） | ✅ 已有 SVG | `reports/training/feature_tsne.svg`、`feature_pca.svg` → 转 PDF 放入 `figures/` |
| **Fig 7** | **类间/类内分离度** | ✅ 已有 SVG | `reports/training/class_separation.svg` |
| **Fig 8**（可选） | **track 锁定前后** 同一视频帧序列对比 | ❌ 缺 | 2–3 帧截图 + 矢量标注 |
| **Fig 9**（可选） | **camera-trap 部署示意**（5 台 IR → 4G → 云端 GPU） | ❌ 缺 | 简图即可，勿强调 Pi 直播 |

**已有可复用矢量资源：**
- `reports/training/training_curves.svg`
- `reports/training/training_finetune_zoom.svg`
- `reports/training/feature_tsne.svg`
- `reports/training/feature_pca.svg`
- `reports/training/class_separation.svg`

**投稿前：** 将 SVG 转为期刊要求的 PDF/EPS，统一字体（Arial/Helvetica），线宽 ≥ 0.5 pt。

---

## 七、你仍可补充（元数据）

- Table 2 逐象准确率  
- 红外野外事件与人工标注的一致性（Secondary evaluation）  
- Figure：建议画 **检测→跟踪→分类→锁定** 流程图，而非 Pi+直播架构图  

- `[facility name, location]`、`[PLACEHOLDER]` 作者单位
- Table 2：17 象逐类准确率（跑 `predict.py` 批量评估）
- Figure 1：`figures/pipeline.pdf` re-ID 流程图（非 Pi 直播架构）
- Figure 2–3：标注示意 + 三次训练曲线
- Figure 4–6：混淆矩阵、t-SNE（见 `reports/training/`）
- Results 中 `[X] ms` 延迟、网络条件
- Ethics、Acknowledgements、Data availability
- `references.bib` 占位文献替换为正式条目

---

## 八、与 Ecological Informatics Aims 的对应（写 Cover Letter 可用）

| 期刊关注点 | 本文对应 |
|------------|----------|
| Image-based monitoring | YOLO + EfficientNet 个体识别 |
| Deep / machine learning | 两阶段迁移学习、99.85% val accuracy |
| Sensor & multimedia data acquisition | Pi + 5× UOVision 4G IR |
| Internet-based archiving & sharing | REST API、/watch/clips、heartbeat |
| Tools for ecological data management | 相机 registry、批量 API 配置 |
| Informing management decisions | 在园个体过滤、标注录像供管理人员审阅 |

---

## 九、编译

```bash
cd manuscript_ecological_informatics
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

或上传 Overleaf，Main document = `main.tex`。

---

请你通读 `main.tex` 后告知：
1. 设施正式名称、地点是否可写入；
2. 日常在园是 7 头还是 17 头（可在 Methods 写 “trained 17, deployed subset of 7”）；
3. 是否需要把中文象名列入 Table 2。

我可据此做第二轮精修（含 Cover Letter 草稿）。
