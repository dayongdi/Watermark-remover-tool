# -*- coding: utf-8 -*-
"""对比：当前整块缩放到 512 vs 分块（tile）原生分辨率推理。"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from selftest import make_photo

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")


def tiled_inpaint(eng, img, mask, tile=512, overlap=96, feather=2):
    """按 512 瓦片在原生分辨率推理，只回填 mask 覆盖的像素。"""
    H, W = img.shape[:2]
    out = img.astype(np.float32).copy()
    step = tile - overlap
    ys = list(range(0, max(1, H - tile + 1), step)) + [max(0, H - tile)]
    xs = list(range(0, max(1, W - tile + 1), step)) + [max(0, W - tile)]
    ys, xs = sorted(set(ys)), sorted(set(xs))
    n = 0
    for y in ys:
        for x in xs:
            y1, x1 = min(y + tile, H), min(x + tile, W)
            m = mask[y:y1, x:x1]
            if not m.any():
                continue
            sub = img[y:y1, x:x1]
            sm = (m > 0).astype(np.uint8) * 255
            # 补到 512（反射），保证模型输入固定尺寸
            ph, pw = tile - sub.shape[0], tile - sub.shape[1]
            sub_p = cv2.copyMakeBorder(sub, 0, ph, 0, pw, cv2.BORDER_REFLECT)
            sm_p = cv2.copyMakeBorder(sm, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
            pred = eng._run(sub_p, sm_p)[: sub.shape[0], : sub.shape[1]]
            a = cv2.GaussianBlur((sm > 0).astype(np.float32), (0, 0), max(0.8, feather * 0.6))
            a = a[:, :, None]
            region = out[y:y1, x:x1]
            out[y:y1, x:x1] = region * (1 - a) + pred.astype(np.float32) * a
            n += 1
    print(f"    瓦片数: {n}")
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    clean = make_photo(1600, 1067)
    # 造一个「大块」水印，模拟真实难例（文字 + 半透明色块）
    dirty = clean.copy()
    h, w = dirty.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.putText(dirty, "SAMPLE  WATERMARK  2026", (180, 900),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, (250, 250, 250), 7, cv2.LINE_AA)
    cv2.putText(mask, "SAMPLE  WATERMARK  2026", (180, 900),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, 255, 13, cv2.LINE_AA)
    cv2.rectangle(dirty, (900, 200), (1250, 420), (235, 235, 235), -1)
    cv2.rectangle(mask, (900, 200), (1250, 420), 255, -1)

    core.imwrite_any(os.path.join(OUT, "cmp_1_原图.jpg"), clean)
    core.imwrite_any(os.path.join(OUT, "cmp_2_带水印.jpg"), dirty)

    eng = core.get_lama()
    mask_x = core._expand_mask(mask, 3)

    # A: 当前方案（整块缩放到 512）
    t0 = time.time()
    y0, y1, x0, x1 = core._bbox_with_margin(mask_x, 128)
    sub = dirty[y0:y1, x0:x1].copy()
    sm = mask_x[y0:y1, x0:x1].copy()
    pred = eng.inpaint(sub, sm)
    outA = dirty.copy()
    a = cv2.GaussianBlur((sm > 0).astype(np.float32), (0, 0), 1.2)[:, :, None]
    outA[y0:y1, x0:x1] = np.clip(sub * (1 - a) + pred * a, 0, 255).astype(np.uint8)
    tA = time.time() - t0
    print(f"  A 整块缩放: {tA:.2f}s  裁切区 {sub.shape[1]}x{sub.shape[0]} → 512")

    # B: 分块原生分辨率
    t0 = time.time()
    outB = tiled_inpaint(eng, dirty, mask_x)
    tB = time.time() - t0
    print(f"  B 分块原生: {tB:.2f}s")

    core.imwrite_any(os.path.join(OUT, "cmp_3_当前方案.jpg"), outA)
    core.imwrite_any(os.path.join(OUT, "cmp_4_分块方案.jpg"), outB)

    m = mask > 0
    for name, img in (("A 整块缩放", outA), ("B 分块原生", outB)):
        err = np.abs(img.astype(np.int16) - clean.astype(np.int16))[m].mean()
        # 掩膜区的清晰度：拉普拉斯方差
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)[m].var()
        print(f"  {name}: 掩膜内误差 {err:.2f}  清晰度(拉普拉斯方差) {lap:.1f}")


if __name__ == "__main__":
    main()
