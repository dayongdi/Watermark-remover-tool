# 照片去水印工具（Windows · 本地离线）

一个完全在本机运行的桌面去水印工具：涂抹水印 → AI 修复 → 保存。
不联网、不上传任何图片，模型全部本地推理。

- **LaMa 图像修复模型**，支持 GPU 加速（RTX 4060 Ti 实测单瓦片 0.099s，CPU 为 1.67s，**17 倍**）
- **原生分辨率分块推理**，不再是"整块缩到 512 再放大"——修复区清晰度提升 2.7 倍
- **自动识别水印**（半透明文字 / 深色文字 / 角标 Logo / 平铺文字）
- 拖拽打开、无限撤销、反复叠加修复、修复前后对比

## 效果数据

1600×1067 照片、大面积文字水印 + 半透明色块，修复区指标：

| 方案 | 耗时 | 修复区清晰度（拉普拉斯方差） |
| --- | --- | --- |
| 整块缩放到 512（旧） | 7.2s | 2.3 |
| **分块原生分辨率 + GPU（现）** | **1.4s** | **6.3** |

自动识别（默认灵敏度）：白字水印 100%、深色字 100%、角标 Logo 72%、平铺文字 90%，
无水印照片误检率 0.02%。

## 快速开始

```bash
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

不需要安装 CUDA Toolkit，pip 会带上运行库；程序启动时自动探测 GPU，失败则回退 CPU。

## 使用

1. 拖入照片（或点「打开照片…」）
2. 点「识别并标红水印」，检查红色区域：多涂的用橡皮擦掉、漏的用画笔补
3. 点「开始去水印」
4. 满意后保存

| 操作 | 说明 |
| --- | --- |
| 左键涂抹 | 圈出要去除的区域 |
| 右键单击 | 切换画笔 / 橡皮 |
| 右键拖动 | 平移画布 |
| 滚轮 | 缩放 |
| 按住空格 | 对比原图 |
| Ctrl+Z | 撤销一笔 |
| Esc | 清空涂抹 |
| `[` / `]` | 画笔大小 |

## 模型

| 模型 | 大小 | 说明 |
| --- | --- | --- |
| `lama_fp32.onnx`（512） | 208MB | 默认。纹理最锐利、速度最快 |
| `lama_1024.onnx` | 198MB | 感受野更大，整块大 Logo / 色块的结构更稳，纹理略平滑、慢约 70% |

两个都在界面下拉里切换，用 `python download_models.py 1024` 拉第二个。
来源：[Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX)、
[Vincent615/lama-2048-onnx](https://huggingface.co/Vincent615/lama-2048-onnx)。

## 技术要点

**分块推理（高清模式）**
按模型原生分辨率切瓦片，只计算与掩膜相交的瓦片；瓦片间重叠 1/4，用余弦权重加权
融合，消除接缝。小面积涂抹几乎零额外开销。

**GPU 加速的两个坑**
- `onnxruntime 1.31` 要求 CUDA 13，Windows 没有现成 cu13 轮子 → 固定用 `1.22.0` 配 CUDA 12
- 光 `os.add_dll_directory()` 不够，ORT 加载 provider DLL 时依赖系统搜索顺序，必须把
  nvidia 运行库目录同时塞进 `PATH`（见 `core._setup_cuda_dlls`）

**自动识别水印**
形态学 top-hat / black-hat 提取细笔画 → 连通域过滤（面积/长宽比/填充率）→
按间距聚类，要求"成组且字高一致"或"靠近画面边缘"→ 边缘带实心色块单独检测。
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
└── 使用说明.md          详细说明
```

## 已知限制

- 水印覆盖到人脸、重要细节时，AI 只能"猜"内容，可能不自然
- 超大涂抹区域（如整幅平铺水印）建议用 1024 模型，或分区域多次修复
- 自动识别是"建议"不是全自动，复杂照片需要人工微调
- Windows only（依赖 win32 API 做热键/DPI；核心算法跨平台）

## License

未指定。如需开源请自行添加 LICENSE 文件。
