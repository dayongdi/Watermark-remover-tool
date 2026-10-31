# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("tkinterdnd2", "onnxruntime"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# 不打包 CUDA/TensorRT 后端（307MB+，且 exe 里没有 CUDA 运行库也用不上），
# exe 走 CPU 推理；源码版可用 GPU。
def _keep(src: str) -> bool:
    low = src.lower()
    return "providers_cuda" not in low and "providers_tensorrt" not in low \
        and "\\nvidia\\" not in low and "/nvidia/" not in low


binaries = [(s, d) for (s, d) in binaries if _keep(s)]
datas = [(s, d) for (s, d) in datas if _keep(s)]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "matplotlib", "scipy", "pandas", "IPython", "PyQt5", "PySide2",
        "PyQt6", "notebook", "jupyter", "pytest", "setuptools",
        "rapidocr_onnxruntime", "torch", "tensorflow",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)


# hook 会在 Analysis 阶段再塞一遍 CUDA provider，这里对最终清单做二次过滤
def _keep_entry(entry) -> bool:
    low = entry[0].lower().replace("/", "\\")
    bad = ("providers_cuda", "providers_tensorrt", "\\nvidia\\",
           "opencv_videoio_ffmpeg500")
    return not any(b in low for b in bad)


binaries_f = [e for e in a.binaries if _keep_entry(e)]
datas_f = [e for e in a.datas if _keep_entry(e)]

exe = EXE(
    pyz,
    a.scripts,
    binaries_f,
    datas_f,
    [],
    name="WatermarkRemover",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=None,
)
