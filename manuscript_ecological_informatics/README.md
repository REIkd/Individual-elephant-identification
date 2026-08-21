# Ecological Informatics — Overleaf 投稿稿

## 在 Overleaf 中使用

1. 打开 [Overleaf](https://www.overleaf.com) → **New Project** → **Upload Project**
2. 将整个 `manuscript_ecological_informatics` 文件夹打包为 `.zip` 上传
3. 设置 **Main document** 为 `main.tex`
4. 编译器选择 **pdfLaTeX**；首次编译后运行 **BibTeX**，再编译两次

Overleaf 已内置 `elsarticle` 文档类，无需额外上传 `.cls` 文件。

## 文件说明

| 文件 | 用途 |
|------|------|
| `main.tex` | 主稿（IMRaD：Introduction / Methods / Results / Discussion / Conclusion） |
| `highlights_body.tex` | 主 PDF 内 Highlights 区块（便于审稿预览） |
| `highlights.tex` | 单独 Highlights 文件（投稿系统可单独上传） |
| `references.bib` | BibTeX 参考文献（初稿占位，提交前需核对） |
| `figures/` | 插图目录（自行添加 PDF/PNG） |

## Highlights 规则（Elsevier）

- 3–5 条 bullet
- **每条 ≤ 85 字符（含空格）**
- 投稿时可上传 `highlights.tex` 或转为 Word，文件名含 `Highlights`

当前 5 条字符数（供核对）：

```
Edge--cloud pipeline IDs 17 Asian elephants from live and IR video  → 62
Body+part VOC training reaches 99.85% on held-out body crops        → 58
YOLO tracking with identity caching stabilizes video labels         → 57
Five 4G UOVision cameras integrated via open cloud APIs             → 53
Web service enables live MJPEG, clips, and IR replay                 → 51
```

## 提交前必改

- 所有 `[PLACEHOLDER]`、`[Name]`、`[X]`
- 作者、单位、邮箱
- Table 1 填入 `predict.py` 批量评估结果
- Fig. 1 替换为 `figures/architecture.pdf`
- 核对 `references.bib` 中 `note = {Placeholder...}` 条目

## 编译命令（本地）

```bash
cd manuscript_ecological_informatics
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
