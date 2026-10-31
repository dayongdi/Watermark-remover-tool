# -*- coding: utf-8 -*-
"""识别参数扫描：在 4 种水印 + 1 张干净图上找平衡点（召回高、误检 0）。"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from selftest import make_photo
from test_auto import wm_logo, wm_text_dark, wm_text_white, wm_tiled


def evaluate(params):
    base = make_photo()
    cases = [wm_text_white(base), wm_text_dark(base), wm_logo(base), wm_tiled(base)]
    recalls = []
    for img, gt in cases:
        det, _ = core.detect_watermark(img, 5, **params)
        d, g = det > 0, gt > 0
        recalls.append(float((d & g).sum()) / max(1, g.sum()))
    det, _ = core.detect_watermark(base, 5, **params)   # 干净图误检
    fp = float((det > 0).sum()) / det.size
    return recalls, fp


def main():
    grid = {
        "thr_override": [33, 28, 24],
        "min_area_f": [0.000008, 0.000015],
        "big_area_f": [0.0012, 0.002],
        "blob_fill": [0.30, 0.42],
    }
    keys = list(grid)
    best = None
    print(f"{'thr':>4} {'minA':>8} {'bigA':>7} {'fill':>5} | "
          f"{'白字':>6} {'深色':>6} {'Logo':>6} {'平铺':>6} | {'误检':>7} | 综合")
    for combo in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, combo))
        r, fp = evaluate(p)
        score = min(r) - fp * 20
        flag = ""
        if best is None or score > best[0]:
            best = (score, p, r, fp)
            flag = " ★"
        print(f"{p['thr_override']:>4} {p['min_area_f']:>8.6f} {p['big_area_f']:>7.4f} "
              f"{p['blob_fill']:>5.2f} | {r[0]*100:5.1f}% {r[1]*100:5.1f}% "
              f"{r[2]*100:5.1f}% {r[3]*100:5.1f}% | {fp*100:6.2f}% | {score:.3f}{flag}")
    print("\n最优:", best[1], "召回", [round(x * 100, 1) for x in best[2]], "误检 %.2f%%" % (best[3] * 100))


if __name__ == "__main__":
    main()
