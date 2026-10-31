# 照片去水印工具 · Watermark Remover

> Windows 本地离线去水印工具 —— 涂抹水印，AI 修复，一键保存。
> 不联网、不上传任何图片，模型全部在本机推理。

**作者：[dayongdi](https://github.com/dayongdi)**

[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Engine](https://img.shields.io/badge/engine-LaMa%20%7C%20Telea%20%7C%20NS-green)]()
[![License](https://img.shields.io/badge/license-未指定-lightgrey)]()

---

## 这是什么

一个给普通用户用的桌面小工具：把照片上的水印涂掉，程序用 AI 把涂掉的地方
"补"回来，效果接近原图。全部离线运行，隐私安全，适合处理证件照、截图、
素材图等不想上传到在线服务的图片。

**👉 [下载免安装版（exe）](https://github.com/dayongdi/Watermark-remover-tool/releases/latest)**

## 功能

- **三种修复引擎**：LaMa（AI，效果最好）/ Telea（极快）/ NS（极快，边缘柔和）
- **GPU 加速**：自动检测 NVIDIA 显卡，实测单瓦片 0.099s（CPU 1.67s，**17 倍**）
- **高清分块推理**：按模型原生分辨率处理，不再是"缩到 512 再放大"
- **自动识别水印**：半透明文字 / 深色文字 / 角标 Logo / 平铺文字
- **顺手的编辑体验**：拖拽打开、滚轮缩放、右键平移、按住空格对比原图、Ctrl+Z 撤销
- **可反复修**：修复结果成为新底图，可以继续涂、继续修，效果叠加

## 效果数据

1600×1067 照片、大面积文字水印 + 半透明色块，修复区指标：

| 方案 | 耗时 | 修复区清晰度（拉普拉斯方差） |
| --- | --- | --- |
| 整块缩放到 512（旧做法） | 7.2s | 2.3 |
| **分块原生分辨率 + GPU（现版本）** | **1.4s** | **6.3** |

自动识别准确率（默认灵敏度）：白字水印 100%、深色字 100%、角标 Logo 72%、
平铺文字 90%，无水印照片误检率 0.02%。

## 快速开始

### 方式一：下载 exe（推荐给普通用户）

到 [Releases](https://github.com/dayongdi/Watermark-remover-tool/releases/latest) 下载免安装包，
解压后双击 `照片去水印.exe`。**注意 `models` 文件夹必须和 exe 放在一起**，AI 修复才能用。
无需安装 Python，无需管理员权限。

### 方式二：源码运行（可用 GPU 加速）

```bash
git clone git@github.com:dayongdi/Watermark-remover-tool.git
cd Watermark-remover-tool

pip install numpy opencv-python pillow tkinterdnd2 onnxruntime
python download_models.py        # 下载模型（198MB，不随仓库分发）
python app.py                    # 或双击 启动.bat
```

想要 GPU 加速（可选，强烈推荐）：

```bash
pip install onnxruntime-gpu==1.22.0
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
            nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-cuda-runtime-cu12 nvidia-nvjitlink-cu12
```

不需要装 CUDA Toolkit，pip 会带运行库。程序启动自动探测 GPU，失败则回退 CPU。

## 使用方法

1. 拖入照片（或点「打开照片…」）
2. 点「识别并标红水印」，检查红色区域：多涂的用橡皮擦掉、漏的用画笔补
3. 点「开始去水印」
4. 满意后点「保存结果」

| 操作 | 说明 |
| --- | --- |
| 左键涂抹 | 圈出要去除的区域 |
| 右键单击 | 切换画笔 / 橡皮 |
| 右键拖动 | 平移画布 |
| 滚轮 | 缩放（以鼠标为中心） |
| 按住空格 | 对比原图 |
| Ctrl+Z | 撤销一笔 |
| Esc | 清空涂抹 |
| `[` / `]` | 画笔变小 / 变大 |

## 模型

| 模型 | 大小 | 说明 |
| --- | --- | --- |
| `lama_fp32.onnx`（512） | 208MB | 默认。纹理最锐利、速度最快 |
| `lama_1024.onnx` | 198MB | 感受野更大，整块大 Logo / 色块的结构更稳，纹理略平滑、慢约 70% |

界面下拉里直接切换。用 `python download_models.py 1024` 拉第二个。

模型来源：[Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX)、
[Vincent615/lama-2048-onnx](https://huggingface.co/Vincent615/lama-2048-onnx)。
模型文件超过 GitHub 单文件 100MB 限制，因此不入库，用脚本下载。

## 技术要点

**分块推理（高清模式）**
按模型原生分辨率切瓦片，只计算与掩膜相交的瓦片；瓦片间重叠 1/4，用余弦权重
加权融合，消除接缝。小面积涂抹几乎没有额外开销。

**GPU 加速的两个坑**
- `onnxruntime 1.31` 要求 CUDA 13，Windows 没有现成 cu13 轮子 → 固定用 `1.22.0` 配 CUDA 12
- 光 `os.add_dll_directory()` 不够，ORT 加载 provider DLL 时依赖系统搜索顺序，必须把
  nvidia 运行库目录同时塞进 `PATH`（见 `core._setup_cuda_dlls`）

**自动识别水印**
形态学 top-hat / black-hat 提取细笔画 → 连通域过滤（面积 / 长宽比 / 填充率）→
按间距聚类，要求"成组且字高一致"或"靠近画面边缘" → 边缘带实心色块单独检测。
参数由 `tune_detect.py` 在 4 种水印样式 + 1 张干净图上扫描选出。

## 打包成免安装 exe

```bash
pip install pyinstaller
pyinstaller build.spec --noconfirm --distpath dist --workpath build
```

产物 `dist/WatermarkRemover.exe`（约 85MB，走 CPU 推理）。打包后自检：

```bash
WatermarkRemover.exe --selftest     # 结果写入同目录 _selftest.txt
```

> `build.spec` 会剔除 CUDA 后端（307MB 且 exe 里没有 CUDA 运行库），所以 exe 用 CPU；
> 需要 GPU 请跑源码版。

## 目录结构

```
watermark-remover/
├── app.py              界面（tkinter）
├── core.py             修复与识别算法（LaMa / Telea / NS / 水印检测 / GPU 选择）
├── download_models.py  模型下载（8 线程分块）
├── selftest.py         修复自测（合成带水印图并跑三种引擎）
├── test_auto.py        识别自测（4 种水印样式 + 无水印对照）
├── tune_detect.py      识别参数扫描
├── compare_tiling.py   分块 vs 整块缩放 的 A/B 对比
├── build.spec          PyInstaller 打包配置
├── 启动.bat            双击启动（自动探测装了依赖的 Python）
├── dist_readme.txt     免安装版随包说明（打包时复制进 dist）
└── 使用说明.md          详细操作说明
```

## 环境要求

- Windows 10 / 11（64 位）
- 源码运行需 Python 3.10+
- GPU 加速需 NVIDIA 显卡（可选，没有也能用）

## 已知限制

- 水印覆盖到人脸、重要细节时，AI 只能"猜"内容，可能不自然
- 超大涂抹区域（如整幅平铺水印）建议用 1024 模型，或分区域多次修复
- 自动识别是"建议"不是全自动，复杂照片需要人工微调
- 目前仅支持 Windows（核心算法跨平台，界面依赖 win32 API）

## 更新日志

### v1.1（2026-10-31）
- 新增 GPU 加速（onnxruntime-gpu + CUDA 12 运行库），推理快 17 倍
- 新增高清分块推理：修复区清晰度 2.3 → 6.3，端到端 7.2s → 1.4s
- 新增 1024 模型可选（大区域结构更稳）
- 修复自动识别退化：平铺文字 18% → 90%，角标 Logo 0% → 72%
- 新增 `download_models.py` 多线程模型下载

### v1.0（2026-10-30）
- 首个版本：LaMa / Telea / NS 三引擎、自动识别水印、拖拽打开、打包 exe

## 作者

**dayongdi** · <https://github.com/dayongdi>

如果这个工具帮到你，欢迎 Star ⭐

## License

未指定。如需开源协议请自行添加 LICENSE 文件。
