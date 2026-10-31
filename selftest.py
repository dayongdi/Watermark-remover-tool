# -*- coding: utf-8 -*-
"""自测脚本：造一张带水印的图，跑两种引擎，输出对比图。"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
os.makedirs(OUT, exist_ok=True)


def make_photo(w=1600, h=1067):
    """合成一张有纹理、渐变、细节的"照片"，便于检验修复质量。"""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.zeros((h, w, 3), np.float32)
    base[..., 0] = 120 + 80 * (yy / h)                      # B
    base[..., 1] = 140 + 60 * (1 - yy / h) + 20 * np.sin(xx / 180)
    base[..., 2] = 90 + 70 * (xx / w)

    # 远处山脊
    for k, off in enumerate([0.55, 0.68, 0.80]):
        ridge = (h * off - 90 * np.sin(xx / (220 + k * 40) + k)).astype(np.int32)
        m = yy.astype(np.int32) > ridge
        col = np.array([70 - k * 12, 95 - k * 15, 110 - k * 18], np.float32)
        base[m] = base[m] * 0.25 + col * 0.75

    # 草地纹理
    noise = rng.normal(0, 6, (h, w, 1)).astype(np.float32)
    grass = yy > h * 0.80
    base[grass] = base[grass] * 0.9 + noise[grass] + \
        np.array([30, 55, 25], np.float32) * (xx[grass][:, None] / w)

    # 一些细节物：石头/树枝
    for _ in range(40):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(int(h * 0.6), h))
        r = int(rng.integers(6, 26))
        col = tuple(int(c) for c in rng.integers(40, 130, 3))
        cv2.ellipse(base, (cx, cy), (r, max(3, r // 2)), int(rng.integers(0, 180)),
                    0, 360, col, -1, cv2.LINE_AA)

    img = np.clip(base, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    return img


def add_watermark(img):
    out = img.copy()
    h, w = out.shape[:2]
    txt = "© PHOTO STOCK 2026"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(1.0, w / 1100)
    th = int(3 * scale)
    (tw, tht), _ = cv2.getTextSize(txt, font, scale, th)
    x = int(w * 0.5 - tw / 2)
    y = int(h * 0.86)

    overlay = out.copy()
    cv2.putText(overlay, txt, (x, y), font, scale, (255, 255, 255), th, cv2.LINE_AA)
    out = cv2.addWeighted(overlay, 0.62, out, 0.38, 0)      # 半透明白字
    cv2.putText(out, txt, (x, y), font, scale, (30, 30, 30), max(1, th - 2), cv2.LINE_AA)

    # 再压一个角标
    cv2.circle(out, (w - 90, 70), 34, (240, 240, 240), -1, cv2.LINE_AA)
    cv2.putText(out, "LOGO", (w - 138, 78), font, 0.6 * scale, (60, 60, 60), 2, cv2.LINE_AA)

    mask = np.zeros((h, w), np.uint8)
    cv2.putText(mask, txt, (x, y), font, scale, 255, th + 6, cv2.LINE_AA)
    cv2.circle(mask, (w - 90, 70), 38, 255, -1, cv2.LINE_AA)
    return out, mask


def main():
    clean = make_photo()
    dirty, mask = add_watermark(clean)
    core.imwrite_any(os.path.join(OUT, "1_原图.jpg"), clean)
    core.imwrite_any(os.path.join(OUT, "2_带水印.jpg"), dirty)
    core.imwrite_any(os.path.join(OUT, "3_掩膜.png"), mask)

    engines = ["telea", "ns"]
    if core.has_model():
        engines.insert(0, "lama")
    else:
        print("[!] 没有 AI 模型，跳过 LaMa")

    for eng in engines:
        t0 = time.time()
        out = core.remove_watermark(dirty, mask, engine=eng, expand=3, feather=2)
        dt = time.time() - t0
        p = os.path.join(OUT, f"4_修复_{eng}.jpg")
        core.imwrite_any(p, out)
        diff = np.abs(out.astype(np.int16) - clean.astype(np.int16)).mean()
        print(f"{eng:6s} 用时 {dt:5.2f}s   与无 watermark 原图平均误差 {diff:.2f}")


if __name__ == "__main__":
    main()
