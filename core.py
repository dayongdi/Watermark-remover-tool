# -*- coding: utf-8 -*-
"""
水印去除核心算法模块
------------------------------------------------
两种修复引擎：
  1. LaMa 深度修复（onnxruntime 本地推理，效果最好）
  2. OpenCV 传统修复（Telea / Navier-Stokes，秒出，适合小面积/纯色背景）

所有计算都在本机完成，不联网、不上传。
"""
from __future__ import annotations

import os
import shutil
import sys
import urllib.request

import cv2
import numpy as np

# 打包成 exe 后，模型目录跟随 exe 所在目录；源码运行时跟随本文件目录
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lama_fp32.onnx")      # 512 分辨率（旧）
MODEL_1024 = os.path.join(MODEL_DIR, "lama_1024.onnx")      # 1024 分辨率（高清，优先）
MODEL_2048 = os.path.join(MODEL_DIR, "lama_2048.onnx")      # 2048 分辨率（超大）

# 优先用分辨率最高的模型
MODEL_CANDIDATES = [MODEL_2048, MODEL_1024, MODEL_PATH]

# 国内可直连的镜像（huggingface 原站多数网络环境不通）
MODEL_URLS = {
    MODEL_1024: [
        "https://hf-mirror.com/Vincent615/lama-2048-onnx/resolve/main/lama_1024.onnx",
    ],
    MODEL_PATH: [
        "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
    ],
}
MIN_MODEL_BYTES = 100 * 1024 * 1024


def best_model_path() -> str | None:
    for p in MODEL_CANDIDATES:
        if os.path.isfile(p) and os.path.getsize(p) > MIN_MODEL_BYTES:
            return p
    return None


# ---------------------------------------------------------------- 图片读写
def imread_any(path: str) -> np.ndarray:
    """支持中文/特殊字符路径的读取，返回 BGR。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"无法读取图片：{path}")
    return img


def imwrite_any(path: str, bgr: np.ndarray, quality: int = 96) -> None:
    """支持中文路径的写出，按扩展名自动选择编码。"""
    ext = os.path.splitext(path)[1].lower() or ".png"
    if ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    elif ext == ".webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)]
    else:
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    ok, buf = cv2.imencode(ext, bgr, params)
    if not ok:
        raise IOError(f"编码失败：{ext}")
    buf.tofile(path)


# ---------------------------------------------------------------- 模型管理
def has_model() -> bool:
    return best_model_path() is not None


def _http_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length") or 0)


def _fetch_range(url: str, start: int, end: int, path: str,
                 progress_cb, lock, counter, total: int):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
        while True:
            b = r.read(256 * 1024)
            if not b:
                break
            f.write(b)
            with lock:
                counter[0] += len(b)
                done = counter[0]
            if progress_cb:
                progress_cb(done, total)


def download_model(progress_cb=None, target: str | None = None) -> str:
    """多线程分块下载模型（单连接在国内镜像上只有几十 KB/s，必须并发）。"""
    import threading

    target = target or MODEL_1024
    os.makedirs(MODEL_DIR, exist_ok=True)
    last_err = None
    parts: list[str] = []
    for url in MODEL_URLS.get(target, []):
        try:
            size = _http_size(url)
            if size < MIN_MODEL_BYTES:
                raise IOError(f"文件大小异常：{size}")
            n = 8
            chunk = (size + n - 1) // n
            parts = [f"{target}.part{i}" for i in range(n)]
            lock = threading.Lock()
            counter = [0]

            def worker(i: int):
                start = i * chunk
                end = min(size - 1, start + chunk - 1)
                _fetch_range(url, start, end, parts[i], progress_cb, lock, counter, size)

            threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                       for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            got = sum(os.path.getsize(p) for p in parts if os.path.exists(p))
            if got != size:
                raise IOError(f"分块不完整 {got}/{size}")

            with open(target, "wb") as out:
                for p in parts:
                    with open(p, "rb") as f:
                        shutil.copyfileobj(f, out)
            for p in parts:
                try:
                    os.remove(p)
                except OSError:
                    pass
            return target
        except Exception as e:  # noqa: BLE001
            last_err = e
            for p in parts:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
    raise RuntimeError(f"模型下载失败：{last_err}")


# ---------------------------------------------------------------- 运行设备
DEVICE_INFO = {"provider": "CPU", "detail": "CPUExecutionProvider", "model": ""}


def _setup_cuda_dlls():
    """把 pip 安装的 nvidia 运行库目录加入 DLL 搜索路径（Windows）。"""
    if os.name != "nt":
        return
    try:
        import glob
        import importlib.util

        bases: list[str] = []
        try:
            spec = importlib.util.find_spec("nvidia")
            if spec and spec.submodule_search_locations:
                bases += [str(p) for p in spec.submodule_search_locations]
        except Exception:  # noqa: BLE001
            pass
        for cand in (os.path.join(sys.prefix, "Lib", "site-packages", "nvidia"),
                     os.path.join(sys.base_prefix, "Lib", "site-packages", "nvidia")):
            if os.path.isdir(cand):
                bases.append(cand)

        dirs: list[str] = []
        for base in bases:
            dirs += glob.glob(os.path.join(base, "*", "bin"))
            dirs += glob.glob(os.path.join(base, "nvidia", "*", "bin"))
        for d in dirs:
            try:
                os.add_dll_directory(d)
            except Exception:  # noqa: BLE001
                pass
        # ORT 加载 provider DLL 时按系统搜索顺序找依赖，光加 DLL 目录不够，还要进 PATH
        if dirs:
            os.environ["PATH"] = ";".join(dirs) + ";" + os.environ.get("PATH", "")
        DEVICE_INFO["dll_dirs"] = len(dirs)
    except Exception:  # noqa: BLE001
        pass


def _make_session(path: str):
    _setup_cuda_dlls()
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    avail = ort.get_available_providers()
    providers: list = []
    if "CUDAExecutionProvider" in avail:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(path, sess_options=so, providers=providers)
    used = sess.get_providers()
    DEVICE_INFO["provider"] = "GPU" if "CUDAExecutionProvider" in used else "CPU"
    DEVICE_INFO["detail"] = ", ".join(used)
    return sess


# ---------------------------------------------------------------- LaMa 引擎
class LamaEngine:
    """懒加载的 LaMa ONNX 推理封装（支持 GPU、分块高清推理）。"""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or best_model_path() or MODEL_PATH
        self.sess = None
        self.in_names = None
        self.fixed_size = None  # (H, W) 若模型是静态尺寸
        self.mask_invert = False
        self.tile = 512

    def load(self):
        if self.sess is not None:
            return self
        try:
            import onnxruntime  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "当前 Python 环境缺少 onnxruntime，无法使用 AI 修复。\n"
                "解决：pip install onnxruntime，或改用「快速修复（Telea）。"
            ) from e

        self.sess = _make_session(self.model_path)
        ins = self.sess.get_inputs()
        self.in_names = [i.name for i in ins]
        if len(ins) >= 2:
            shp = ins[0].shape
            if len(shp) == 4 and isinstance(shp[2], int) and isinstance(shp[3], int):
                if shp[2] > 0 and shp[3] > 0:
                    self.fixed_size = (int(shp[2]), int(shp[3]))
        self.tile = self.fixed_size[0] if self.fixed_size else 1024
        DEVICE_INFO["model"] = os.path.basename(self.model_path)
        return self

    # ---- 预处理：img BGR uint8, mask uint8(0/255) -> 模型输入
    def _run(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        if self.fixed_size:
            th, tw = self.fixed_size
            if H <= th and W <= tw:
                # 小区域：反射填充到模型尺寸，不做缩放，保留细节
                img_r = cv2.copyMakeBorder(img, 0, th - H, 0, tw - W, cv2.BORDER_REFLECT)
                mask_r = cv2.copyMakeBorder(mask, 0, th - H, 0, tw - W,
                                            cv2.BORDER_CONSTANT, value=0)
            else:
                img_r = cv2.resize(img, (tw, th),
                                   interpolation=cv2.INTER_AREA if th < H else cv2.INTER_LINEAR)
                mask_r = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        else:
            th = max(8, H + (-H % 8))
            tw = max(8, W + (-W % 8))
            if (th, tw) != (H, W):
                img_r = cv2.copyMakeBorder(img, 0, th - H, 0, tw - W, cv2.BORDER_REFLECT)
                mask_r = cv2.copyMakeBorder(mask, 0, th - H, 0, tw - W,
                                            cv2.BORDER_CONSTANT, value=0)
            else:
                img_r, mask_r = img, mask

        x = img_r[:, :, ::-1].astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]
        m = (mask_r > 0).astype(np.float32)[None, None]
        if self.mask_invert:
            m = 1.0 - m

        feeds = {self.in_names[0]: x}
        if len(self.in_names) >= 2:
            feeds[self.in_names[1]] = m
        out = self.sess.run(None, feeds)[0]

        out = np.asarray(out)
        if out.ndim == 4:
            out = out[0]
        out = np.transpose(out, (1, 2, 0))
        if out.shape[2] == 1:
            out = np.repeat(out, 3, axis=2)
        # 兼容两种导出：输出 0-1 或 0-255
        if out.max() <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = out[:, :, ::-1].copy()  # RGB->BGR

        if out.shape[0] > H or out.shape[1] > W:
            out = out[:H, :W].copy()
        if out.shape[:2] != (H, W):
            out = cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
        return out

    # ---- 快速模式：整块缩放（旧行为）
    def inpaint_fast(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        LIMIT = 1536
        if max(H, W) > LIMIT:
            s = LIMIT / max(H, W)
            small = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
            m_small = cv2.resize(mask, (small.shape[1], small.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            out = self._run(small, m_small)
            return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
        return self._run(img, mask)

    # ---- 高清模式：原生分辨率分块 + 加权融合
    def inpaint_tiled(self, img: np.ndarray, mask: np.ndarray,
                      feather: int = 2, progress=None) -> np.ndarray:
        H, W = img.shape[:2]
        tile = int(self.tile or 512)
        ov = max(32, tile // 4)
        step = max(1, tile - ov)

        ys = _tile_positions(H, tile, step)
        xs = _tile_positions(W, tile, step)
        hit = mask > 0

        jobs = []
        for y in ys:
            for x in xs:
                y1, x1 = min(y + tile, H), min(x + tile, W)
                if hit[y:y1, x:x1].any():
                    jobs.append((y, x, y1, x1))
        if not jobs:
            return img

        weight = _tile_weight(tile, ov)
        acc = np.zeros((H, W, 3), np.float32)
        wsum = np.zeros((H, W, 1), np.float32)
        sigma = max(0.8, feather * 0.6)

        for i, (y, x, y1, x1) in enumerate(jobs):
            sub = img[y:y1, x:x1]
            sm = (mask[y:y1, x:x1] > 0).astype(np.uint8) * 255
            ph, pw = tile - sub.shape[0], tile - sub.shape[1]
            if ph or pw:
                sub_p = cv2.copyMakeBorder(sub, 0, ph, 0, pw, cv2.BORDER_REFLECT)
                sm_p = cv2.copyMakeBorder(sm, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
            else:
                sub_p, sm_p = sub, sm
            pred = self._run(sub_p, sm_p)[: sub.shape[0], : sub.shape[1]]

            alpha = cv2.GaussianBlur((sm > 0).astype(np.float32), (0, 0), sigma)
            w = weight[: sub.shape[0], : sub.shape[1], None] * alpha[:, :, None]
            acc[y:y1, x:x1] += pred.astype(np.float32) * w
            wsum[y:y1, x:x1] += w
            if progress:
                pct = int(45 + 45 * (i + 1) / len(jobs))
                progress(pct, f"高清分块 {i + 1}/{len(jobs)}")

        out = img.astype(np.float32).copy()
        ok = wsum[:, :, 0] > 1e-6
        out[ok] = acc[ok] / wsum[ok]
        return np.clip(out, 0, 255).astype(np.uint8)

    def inpaint(self, img: np.ndarray, mask: np.ndarray, hd: bool = True,
                feather: int = 2, progress=None) -> np.ndarray:
        if hd:
            return self.inpaint_tiled(img, mask, feather=feather, progress=progress)
        return self.inpaint_fast(img, mask)


def _tile_positions(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, step))
    if not pos or pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


def _tile_weight(tile: int, ov: int) -> np.ndarray:
    """瓦片权重：边缘用余弦斜坡衰减，重叠区平滑融合，消除接缝。"""
    w = np.ones((tile, tile), np.float32)
    k = min(ov, tile // 2)
    if k > 1:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))).astype(np.float32)
        w[:k, :] *= ramp[:, None]
        w[-k:, :] *= ramp[::-1][:, None]
        w[:, :k] *= ramp[None, :]
        w[:, -k:] *= ramp[::-1][None, :]
    return np.maximum(w, 1e-3)


_ENGINE_CACHE: dict[str, LamaEngine] = {}
ACTIVE_MODEL: str | None = None   # None = 自动挑可用的最高分辨率


def get_lama(path: str | None = None) -> LamaEngine:
    p = path or ACTIVE_MODEL or best_model_path() or MODEL_PATH
    key = os.path.normcase(p)
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = LamaEngine(p).load()
    return _ENGINE_CACHE[key]


def set_active_model(path: str | None):
    global ACTIVE_MODEL
    ACTIVE_MODEL = path


# ---------------------------------------------------------------- 统一入口
ENGINES = {
    "lama": "AI 精细修复（LaMa）",
    "telea": "快速修复（Telea）",
    "ns": "平滑修复（Navier-Stokes）",
}


def _expand_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.dilate(mask, k)


def detect_watermark(bgr: np.ndarray, sensitivity: int = 5, *,
                     min_area_f: float = 0.000008,
                     big_area_f: float = 0.0012,
                     blob_fill: float = 0.30,
                     thr_override: float | None = None) -> tuple[np.ndarray, int]:
    """
    自动识别疑似水印区域（半透明文字 / 角标 / 平铺文字）。

    思路：水印几乎都是"细笔画的、与背景有稳定对比度的、成组成行出现"的元素。
      1. 形态学 top-hat / black-hat 提取细笔画（比背景亮/暗的窄结构）
      2. 阈值 + 连通域筛选（面积、长宽比、填充率）
      3. 按间距聚类，要求"成组且字高一致"或"靠近画面边缘"，滤掉草地/树枝等杂纹理
      4. 返回膨胀后的掩膜，交给用户确认 / 微调

    sensitivity: 1~10，越大越激进（识别更多）
    返回 (mask, 区域组数)
    """
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    k = max(3, int(min(H, W) * 0.015))
    k = k if k % 2 == 1 else k + 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern)
    blk = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kern)
    resp = cv2.max(top, blk)
    resp = cv2.GaussianBlur(resp, (0, 0), 1.0)

    thr = float(thr_override if thr_override is not None
                else np.clip(58 - sensitivity * 5, 12, 62))
    binm = (resp >= thr).astype(np.uint8) * 255
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
    min_area = max(8, int(min_area_f * W * H))
    max_area = 0.08 * W * H
    ids_ok = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area or a > max_area:
            continue
        if w < 4 or h < 4 or w / h > 25 or h / w > 25:
            continue
        fill = a / float(w * h)
        if fill < 0.03 or fill > 0.92:
            continue
        ids_ok.append(i)

    empty = np.zeros((H, W), np.uint8)
    if not ids_ok:
        return empty, 0

    lut = np.zeros(n, bool)
    lut[ids_ok] = True
    comp_mask = (lut[labels]).astype(np.uint8) * 255

    # 聚类：把同一行/同一个水印的字符连起来
    ck = max(5, int(min(H, W) * 0.03))
    ck = ck if ck % 2 == 1 else ck + 1
    clusters = cv2.dilate(comp_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))
    n2, lab2 = cv2.connectedComponents(clusters, 8)

    out = np.zeros((H, W), np.uint8)
    kept = 0
    for j in range(1, n2):
        sel = lab2 == j
        if not sel.any():
            continue
        ids = np.unique(labels[sel & (comp_mask > 0)])
        ids = ids[ids > 0]
        if len(ids) == 0:
            continue
        hs = [stats[i][3] for i in ids]
        med = float(np.median(hs))
        consistent = sum(1 for hh in hs if 0.45 * med <= hh <= 2.2 * med)
        area = int(sum(stats[i][4] for i in ids))
        x0 = min(stats[i][0] for i in ids)
        y0 = min(stats[i][1] for i in ids)
        x1 = max(stats[i][0] + stats[i][2] for i in ids)
        y1 = max(stats[i][1] + stats[i][3] for i in ids)
        near_border = (x0 < 0.22 * W) or (y0 < 0.22 * H) or (x1 > 0.78 * W) or (y1 > 0.78 * H)

        is_text_line = len(ids) >= 3 and consistent >= 3
        is_border_mark = near_border and len(ids) >= 2 and consistent >= 2
        is_big_mark = area > big_area_f * W * H
        if not (is_text_line or is_border_mark or is_big_mark):
            continue

        lut2 = np.zeros(n, bool)
        lut2[ids] = True
        out |= ((lut2[labels]) & (comp_mask > 0)).astype(np.uint8) * 255
        kept += 1

    if kept:
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # ---- 实心角标 / 方块水印：大块均匀区域，与局部背景对比强，且靠近画面边缘
    kb = max(9, int(min(H, W) * 0.05))
    kb = kb if kb % 2 == 1 else kb + 1
    bg = cv2.medianBlur(gray, kb)
    diff = cv2.absdiff(gray, bg)
    sb = (diff > 42).astype(np.uint8) * 255
    sb = cv2.morphologyEx(sb, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    band = np.zeros((H, W), np.uint8)
    bx, by = int(W * 0.25), int(H * 0.25)
    band[:by, :] = 255
    band[H - by:, :] = 255
    band[:, :bx] = 255
    band[:, W - bx:] = 255
    sb = cv2.bitwise_and(sb, band)

    n3, l3, s3, _ = cv2.connectedComponentsWithStats(sb, 8)
    min_blob = max(400, int(0.0006 * W * H))
    for i in range(1, n3):
        x, y, w, h, a = s3[i]
        if a < min_blob or a > 0.05 * W * H:
            continue
        if w < 12 or h < 12:
            continue
        ar = w / float(h)
        if ar < 0.25 or ar > 4.0:
            continue
        if a / float(w * h) < blob_fill:  # 要求较"实"
            continue
        blob = (l3 == i).astype(np.uint8) * 255
        out = cv2.bitwise_or(out, blob)
        kept += 1

    if kept:
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return out, kept


def _bbox_with_margin(mask: np.ndarray, margin: int):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    H, W = mask.shape[:2]
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(H, int(ys.max()) + 1 + margin)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(W, int(xs.max()) + 1 + margin)
    return y0, y1, x0, x1


def remove_watermark(
    bgr: np.ndarray,
    mask: np.ndarray,
    engine: str = "lama",
    expand: int = 3,
    feather: int = 2,
    hd: bool = True,
    progress=None,
) -> np.ndarray:
    """
    在 mask 标记的区域做内容修复。

    bgr     : 原图 BGR uint8
    mask    : 单通道 uint8，非 0 表示要修复
    engine  : lama / telea / ns
    expand  : 掩膜外扩像素（吃掉水印抗锯齿边）
    feather : 边缘羽化像素（融合接缝）
    hd      : True=分块原生分辨率推理（清晰，推荐）；False=整块缩放（快）
    progress: 回调 progress(percent:int, text:str)
    """
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    mask = (mask > 0).astype(np.uint8) * 255
    if mask.shape[:2] != bgr.shape[:2]:
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    if not mask.any():
        raise ValueError("还没有涂抹任何需要去除的区域")

    mask_x = _expand_mask(mask, expand)

    if progress:
        progress(5, "准备区域…")

    H, W = bgr.shape[:2]
    use_crop = max(H, W) > 1200
    if use_crop:
        if engine != "lama":
            margin = 40
        else:
            margin = 256 if hd else 128
        y0, y1, x0, x1 = _bbox_with_margin(mask_x, margin)
        work_img = bgr[y0:y1, x0:x1].copy()
        work_mask = mask_x[y0:y1, x0:x1].copy()
    else:
        y0, y1, x0, x1 = 0, H, 0, W
        work_img = bgr.copy()
        work_mask = mask_x.copy()

    if progress:
        progress(20, "修复中…")

    if engine == "lama":
        if progress:
            progress(25, "加载模型（首次较慢）…")
        eng = get_lama()
        if hd:
            filled = eng.inpaint(work_img, work_mask, hd=True,
                                 feather=feather, progress=progress)
            # 分块模式内部已完成加权融合，直接回贴
            out = bgr.copy()
            out[y0:y1, x0:x1] = filled
            if progress:
                progress(100, "完成")
            return out
        if progress:
            progress(40, "模型推理中…")
        filled = eng.inpaint(work_img, work_mask, hd=False)
    elif engine == "telea":
        filled = cv2.inpaint(work_img, work_mask, 5, cv2.INPAINT_TELEA)
    elif engine == "ns":
        filled = cv2.inpaint(work_img, work_mask, 7, cv2.INPAINT_NS)
    else:
        raise ValueError(f"未知引擎：{engine}")

    if progress:
        progress(80, "融合接缝…")

    alpha = work_mask.astype(np.float32) / 255.0
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.8, feather * 0.6))
    a = alpha[:, :, None]

    out = bgr.copy()
    blended = work_img.astype(np.float32) * (1 - a) + filled.astype(np.float32) * a
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

    if progress:
        progress(100, "完成")
    return out
