# -*- coding: utf-8 -*-
"""自动识别水印的自测：多种水印样式 + 无水印对照（查误检）。"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from selftest import make_photo

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
os.makedirs(OUT, exist_ok=True)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def wm_text_white(img):
    out, gt = img.copy(), np.zeros(img.shape[:2], np.uint8)
    h, w = out.shape[:2]
    txt, sc, th = "© PHOTO STOCK 2026", max(1.0, w / 1100), 3
    (tw, _), _ = cv2.getTextSize(txt, FONT, sc, th)
    x, y = int(w * .5 - tw / 2), int(h * .86)
    ov = out.copy()
    cv2.putText(ov, txt, (x, y), FONT, sc, (255, 255, 255), th, cv2.LINE_AA)
    out = cv2.addWeighted(ov, .62, out, .38, 0)
    cv2.putText(gt, txt, (x, y), FONT, sc, 255, th + 6, cv2.LINE_AA)
    return out, gt


def wm_text_dark(img):
    out, gt = img.copy(), np.zeros(img.shape[:2], np.uint8)
    h, w = out.shape[:2]
    txt, sc, th = "SAMPLE IMAGE", max(1.0, w / 1200), 3
    x, y = int(w * .06), int(h * .10)
    ov = out.copy()
    cv2.putText(ov, txt, (x, y), FONT, sc, (10, 10, 10), th, cv2.LINE_AA)
    out = cv2.addWeighted(ov, .55, out, .45, 0)
    cv2.putText(gt, txt, (x, y), FONT, sc, 255, th + 6, cv2.LINE_AA)
    return out, gt


def wm_logo(img):
    out, gt = img.copy(), np.zeros(img.shape[:2], np.uint8)
    h, w = out.shape[:2]
    c = (w - 96, 78)
    cv2.circle(out, c, 40, (245, 245, 245), -1, cv2.LINE_AA)
    cv2.circle(out, c, 40, (60, 60, 60), 3, cv2.LINE_AA)
    cv2.putText(out, "LOGO", (c[0] - 44, c[1] + 8), FONT, .62, (50, 50, 50), 2, cv2.LINE_AA)
    cv2.circle(gt, c, 46, 255, -1, cv2.LINE_AA)
    return out, gt


def wm_tiled(img):
    out, gt = img.copy(), np.zeros(img.shape[:2], np.uint8)
    h, w = out.shape[:2]
    txt = "STOCK PHOTO"
    layer = np.zeros_like(out)
    gtl = np.zeros((h, w), np.uint8)
    for y in range(-h, h * 2, 190):
        for x in range(-w, w * 2, 380):
            cv2.putText(layer, txt, (x, y), FONT, .9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(gtl, txt, (x, y), FONT, .9, 255, 2, cv2.LINE_AA)
    m = gtl > 0
    out[m] = (out[m].astype(np.float32) * .55 + 255 * .45).astype(np.uint8)
    return out, gtl


def report(name, img, gt, sens=5):
    det, groups = core.detect_watermark(img, sens)
    d = det > 0
    g = gt > 0
    hit = int((d & g).sum())
    recall = hit / max(1, g.sum())
    cover = d.sum() / d.size
    print(f"{name:14s} 组数={groups:3d}  召回={recall*100:5.1f}%  覆盖={cover*100:5.2f}%")
    vis = img.copy()
    if d.any():
        vis[d] = (vis[d].astype(np.float32) * .45 + np.array([50, 50, 240]) * .55).astype(np.uint8)
    core.imwrite_any(os.path.join(OUT, f"auto_{name}.jpg"), vis)
    return recall, cover


def main():
    base = make_photo()
    cases = {
        "白字水印": wm_text_white(base),
        "深色字": wm_text_dark(base),
        "角标Logo": wm_logo(base),
        "平铺文字": wm_tiled(base),
    }
    for name, (img, gt) in cases.items():
        report(name, img, gt)
    # 无水印对照：应几乎不检出
    report("无水印对照", base, np.zeros(base.shape[:2], np.uint8))


if __name__ == "__main__":
    main()
