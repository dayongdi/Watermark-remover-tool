# -*- coding: utf-8 -*-
"""
照片水印去除工具 —— 本地桌面版
------------------------------------------------
用法：双击「启动.bat」，或命令行 python app.py [图片路径]

操作：
  左键涂抹     = 圈出水印（红色覆盖区）
  右键拖动/中键 = 平移画布
  右键单击     = 切换画笔 / 橡皮
  滚轮         = 缩放
  按住空格     = 对比原图
"""
from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import core

APP_TITLE = "照片去水印工具 · 本地离线版"
SUPPORTED = [
    ("图片文件", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
    ("所有文件", "*.*"),
]
MAX_SIDE = 6000 if sys.maxsize > 2 ** 32 else 2600  # 32 位进程内存有限，降低大图上限


class App:
    def __init__(self, root: tk.Tk, initial: str | None = None, dnd_ok: bool = False):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1320x880")
        self.root.minsize(960, 640)
        self.root.configure(bg="#f3f4f6")
        self._dnd_ok = dnd_ok

        # ---- 图像状态
        self.orig0: np.ndarray | None = None
        self.base: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.strokes: list[dict] = []
        self.history: list[np.ndarray] = []
        self.src_path: str | None = None

        # ---- 视图状态
        self.zoom = 1.0
        self.ox = 0.0
        self.oy = 0.0
        self.photo = None
        self._pan = None
        self._cur = None
        self._temp_items: list[int] = []
        self._show_orig = False
        self._busy = False
        self._r3 = None

        # ---- 参数
        self.tool = tk.StringVar(value="brush")
        self.brush = tk.IntVar(value=40)
        self.expand = tk.IntVar(value=3)
        self.feather = tk.IntVar(value=2)
        self.sens = tk.IntVar(value=6)
        self.engine_label = tk.StringVar()
        self.show_mask = tk.BooleanVar(value=True)
        self._engine_keys: list[str] = []
        self._scales: dict[str, ttk.Scale] = {}
        self._scale_vals: dict[str, ttk.Label] = {}

        self._build_ui()
        self._register_dnd()
        self._bind_keys()
        self._update_engine_choices()

        if initial and os.path.isfile(initial):
            self.root.after(120, lambda: self.load_image(initial))

    # ------------------------------------------------------------ 界面
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        base_font = ("Microsoft YaHei UI", 9)
        style.configure(".", font=base_font)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Hint.TLabel", foreground="#6b7280", font=("Microsoft YaHei UI", 8))
        style.configure("Go.TButton", font=("Microsoft YaHei UI", 10, "bold"), padding=(6, 9))

        body = tk.Frame(self.root, bg="#f3f4f6")
        body.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        wrap = tk.Frame(body, bg="#2f3136", bd=0)
        wrap.grid(row=0, column=0, sticky="nsew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(wrap, bg="#2f3136", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-2>", self._pan_start)
        self.canvas.bind("<B2-Motion>", self._pan_move)
        self.canvas.bind("<Button-3>", self._r3_press)
        self.canvas.bind("<B3-Motion>", self._r3_move)
        self.canvas.bind("<ButtonRelease-3>", self._r3_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Configure>", self._on_configure)

        panel = tk.Frame(body, bg="#ffffff", width=254,
                         highlightbackground="#e5e7eb", highlightthickness=1)
        panel.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        panel.pack_propagate(False)
        self._build_panel(panel)

        bar = tk.Frame(self.root, bg="#f3f4f6", height=26)
        bar.pack(fill="x", padx=12, pady=(6, 8))
        self.status = ttk.Label(bar, text="打开一张照片开始", style="Hint.TLabel")
        self.status.pack(side="left")
        self.zoom_lbl = ttk.Label(bar, text="", style="Hint.TLabel")
        self.zoom_lbl.pack(side="right")

        self._draw_placeholder()

    def _build_panel(self, p: tk.Frame):
        P = 14
        ttk.Label(p, text="照片去水印", style="Title.TLabel").pack(anchor="w", pady=(14, 2), padx=P)
        ttk.Label(p, text="全部处理在本机完成，不上传任何图片",
                  style="Hint.TLabel").pack(anchor="w", pady=(0, 10), padx=P)

        ttk.Button(p, text="打开照片…", command=self.pick_file).pack(fill="x", padx=P)
        self.btn_save = ttk.Button(p, text="保存结果…", command=self.save_result, state="disabled")
        self.btn_save.pack(fill="x", pady=(6, 0), padx=P)

        self._sep(p)

        ttk.Label(p, text="涂抹工具").pack(anchor="w", padx=P)
        row = tk.Frame(p, bg="#ffffff")
        row.pack(fill="x", pady=(4, 6), padx=P)
        ttk.Radiobutton(row, text="画笔", value="brush", variable=self.tool,
                        command=self._tool_changed).pack(side="left")
        ttk.Radiobutton(row, text="橡皮", value="erase", variable=self.tool,
                        command=self._tool_changed).pack(side="left", padx=(14, 0))

        self._slider(p, "brush", "画笔大小", self.brush, 4, 200)

        row2 = tk.Frame(p, bg="#ffffff")
        row2.pack(fill="x", pady=(6, 0), padx=P)
        ttk.Button(row2, text="撤销一笔", command=self.undo_stroke).pack(
            side="left", expand=True, fill="x")
        ttk.Button(row2, text="清空", command=self.clear_mask).pack(
            side="left", expand=True, fill="x", padx=(6, 0))

        self._sep(p)

        # 自动识别
        ttk.Label(p, text="自动识别水印").pack(anchor="w", padx=P)
        self.btn_auto = ttk.Button(p, text="识别并标红水印", command=self.auto_detect)
        self.btn_auto.pack(fill="x", pady=(4, 2), padx=P)
        self._slider(p, "sens", "识别灵敏度", self.sens, 1, 10)
        ttk.Label(p, text="识别后请检查：多涂的用橡皮擦掉，漏掉的手动补涂。",
                  style="Hint.TLabel", wraplength=214, justify="left").pack(
            anchor="w", pady=(2, 0), padx=P)

        self._sep(p)

        ttk.Label(p, text="修复引擎").pack(anchor="w", padx=P)
        self.cb_engine = ttk.Combobox(p, textvariable=self.engine_label, state="readonly")
        self.cb_engine.pack(fill="x", pady=(4, 2), padx=P)

        self.hd = tk.BooleanVar(value=True)
        ttk.Checkbutton(p, text="高清分块修复（清晰，稍慢）", variable=self.hd).pack(
            anchor="w", padx=P, pady=(2, 0))

        self.model_label = tk.StringVar()
        self.cb_model = ttk.Combobox(p, textvariable=self.model_label, state="readonly")
        self.cb_model.pack(fill="x", pady=(6, 0), padx=P)
        self.cb_model.bind("<<ComboboxSelected>>", lambda e: self._model_changed())

        self.dev_lbl = ttk.Label(p, text="", style="Hint.TLabel", wraplength=214,
                                 justify="left")
        self.dev_lbl.pack(anchor="w", padx=P, pady=(2, 0))

        # 模型下载区（缺模型时显示）
        self.model_box = tk.Frame(p, bg="#ffffff")
        self.btn_model = ttk.Button(self.model_box, text="下载 AI 模型（198 MB）",
                                    command=self.download_model)
        self.btn_model.pack(fill="x")
        self.model_hint = ttk.Label(
            self.model_box, style="Hint.TLabel", wraplength=214, justify="left",
            text="未检测到 AI 模型。现在只能用「快速修复」，效果一般；下载后可获得明显更好的效果。")
        self.model_hint.pack(fill="x", pady=(4, 0))

        self._slider(p, "expand", "边缘外扩（吃边）", self.expand, 0, 12)
        self._slider(p, "feather", "接缝羽化", self.feather, 0, 8)

        self._sep(p)

        self.btn_run = ttk.Button(p, text="开始去水印", style="Go.TButton",
                                  command=self.run_remove, state="disabled")
        self.btn_run.pack(fill="x", pady=(2, 6), padx=P)

        self.btn_undo_repair = ttk.Button(p, text="撤销修复", command=self.undo_repair,
                                          state="disabled")
        self.btn_undo_repair.pack(fill="x", padx=P)

        self.pb = ttk.Progressbar(p, mode="determinate", maximum=100)
        self.pb.pack(fill="x", pady=(12, 4), padx=P)
        self.pb_lbl = ttk.Label(p, text="", style="Hint.TLabel", wraplength=214)
        self.pb_lbl.pack(anchor="w", padx=P)

        tip = tk.Frame(p, bg="#f9fafb")
        tip.pack(fill="x", side="bottom", pady=12, padx=10)
        ttk.Label(tip, text="左键涂抹 · 右键拖动平移 · 右键单击切换橡皮 · 滚轮缩放 · 空格对比原图",
                  style="Hint.TLabel", wraplength=210, justify="left").pack(
            anchor="w", padx=8, pady=6)

    def _sep(self, p):
        ttk.Separator(p, orient="horizontal").pack(fill="x", pady=10, padx=14)

    def _slider(self, p, key, text, var, lo, hi):
        wrap = tk.Frame(p, bg="#ffffff")
        wrap.pack(fill="x", padx=14, pady=(6, 0))
        head = tk.Frame(wrap, bg="#ffffff")
        head.pack(fill="x")
        ttk.Label(head, text=text).pack(side="left")
        val = ttk.Label(head, text=str(var.get()), style="Hint.TLabel")
        val.pack(side="right")

        def on_change(v, var=var, val=val):
            iv = int(float(v))
            var.set(iv)
            val.config(text=str(iv))

        s = ttk.Scale(wrap, from_=lo, to=hi, orient="horizontal", command=on_change)
        s.set(var.get())
        s.pack(fill="x")
        self._scales[key] = s
        self._scale_vals[key] = val

    def _draw_placeholder(self):
        self.canvas.delete("all")
        cw = max(self.canvas.winfo_width(), 400)
        ch = max(self.canvas.winfo_height(), 300)
        self.canvas.create_text(cw // 2, ch // 2 - 22, text="把照片拖进窗口",
                                fill="#9ca3af", font=("Microsoft YaHei UI", 16))
        self.canvas.create_text(cw // 2, ch // 2 + 12,
                                text="（也可以点左侧「打开照片…」）",
                                fill="#6b7280", font=("Microsoft YaHei UI", 9))
        self.canvas.create_text(cw // 2, ch // 2 + 46,
                                text="用画笔涂掉水印（或点「识别并标红水印」）→ 开始去水印",
                                fill="#6b7280", font=("Microsoft YaHei UI", 9))

    def _on_configure(self, e):
        if self.base is None:
            self._draw_placeholder()
        else:
            self.render()

    # ------------------------------------------------------------ 快捷键
    def _bind_keys(self):
        r = self.root
        r.bind("<space>", lambda e: self._set_show_orig(True))
        r.bind("<KeyRelease-space>", lambda e: self._set_show_orig(False))
        r.bind("<Control-z>", lambda e: self.undo_stroke())
        r.bind("<Control-s>", lambda e: self.save_result())
        r.bind("<Control-o>", lambda e: self.pick_file())
        r.bind("<Escape>", lambda e: self.clear_mask())
        r.bind("<bracketleft>", lambda e: self._nudge_brush(-6))
        r.bind("<bracketright>", lambda e: self._nudge_brush(6))

    def _nudge_brush(self, d):
        v = int(np.clip(self.brush.get() + d, 4, 200))
        self.brush.set(v)
        self._scales["brush"].set(v)
        self._scale_vals["brush"].config(text=str(v))

    def _tool_changed(self):
        self.canvas.config(cursor="crosshair" if self.tool.get() == "brush" else "dotbox")

    def _set_show_orig(self, on: bool):
        if self.orig0 is None or self._show_orig == on:
            return
        self._show_orig = on
        self.render()

    # ------------------------------------------------------------ 打开 / 保存
    def pick_file(self):
        path = filedialog.askopenfilename(title="选择照片", filetypes=SUPPORTED)
        if path:
            self.load_image(path)

    # ------------------------------------------------------------ 拖拽打开
    def _register_dnd(self):
        if not self._dnd_ok:
            return
        try:
            from tkinterdnd2 import DND_FILES
            for w in (self.root, self.canvas):
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<DropEnter>>", self._drop_enter)
                w.dnd_bind("<<DropLeave>>", self._drop_leave)
                w.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:  # noqa: BLE001
            self._dnd_ok = False

    def _drop_enter(self, e):
        if self.base is None:
            self.canvas.config(bg="#3d4248")
        self._flash("松开鼠标即可载入照片")

    def _drop_leave(self, e):
        self.canvas.config(bg="#2f3136")

    def _on_drop(self, e):
        self.canvas.config(bg="#2f3136")
        try:
            paths = [p.strip("{}") for p in self.root.tk.splitlist(e.data)]
        except Exception:  # noqa: BLE001
            paths = [e.data.strip("{}")]
        imgs = [p for p in paths
                if os.path.splitext(p)[1].lower() in
                (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")]
        if not imgs:
            self._flash("拖入的不是图片文件")
            return
        self.load_image(imgs[0])
        if len(imgs) > 1:
            self._flash(f"已载入第一张；一次只支持处理一张（还有 {len(imgs)-1} 张未载入）")

    # ------------------------------------------------------------ 自动识别
    def auto_detect(self):
        if self._busy or self.base is None:
            return
        if self.mask is None:
            return
        self.btn_auto.config(state="disabled")
        self.root.update_idletasks()
        try:
            mask, groups = core.detect_watermark(self.base, self.sens.get())
        finally:
            self.btn_auto.config(state="normal")

        if not mask.any():
            self._flash("没识别到明显水印：可调高「识别灵敏度」再试，或手动涂抹")
            return
        stroke = {"type": "auto", "layer": mask.copy()}
        self._rasterize(stroke)
        self.strokes.append(stroke)
        self.render()
        self.btn_run.config(state="normal")
        self._flash(f"识别到 {groups} 处疑似水印并已标红——请检查："
                    "多涂的用橡皮擦掉，漏掉的用画笔补上，然后点「开始去水印」")

    def load_image(self, path: str):
        try:
            img = core.imread_any(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("打开失败", str(e))
            return

        h, w = img.shape[:2]
        if max(h, w) > MAX_SIDE:
            s = MAX_SIDE / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        self.orig0 = img.copy()
        self.base = img.copy()
        self.mask = np.zeros((h, w), np.uint8)
        self.strokes.clear()
        self.history.clear()
        self.src_path = path

        self.canvas.focus_set()
        self.canvas.update_idletasks()
        self._fit_view()
        self.render()

        self.btn_run.config(state="normal")
        self.btn_save.config(state="normal")
        self.btn_undo_repair.config(state="disabled")
        self.pb.config(value=0)
        self.pb_lbl.config(text="")
        self._flash(f"已打开 {os.path.basename(path)}   ·   {w}×{h}   ·   涂抹水印后点击「开始去水印」")

    def save_result(self):
        if self.base is None:
            return
        stem, ext = os.path.splitext(os.path.basename(self.src_path or "photo.png"))
        ext = ext.lower() or ".png"
        if ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
            ext = ".png"
        init_dir = os.path.dirname(self.src_path) if self.src_path else ""
        path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=ext,
            initialdir=init_dir,
            initialfile=f"{stem}_无水印{ext}",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("WEBP", "*.webp")],
        )
        if not path:
            return
        try:
            core.imwrite_any(path, self.base, quality=96)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("保存失败", str(e))
            return
        self._flash(f"已保存：{path}")

    # ------------------------------------------------------------ 视图
    def _fit_view(self):
        if self.base is None:
            return
        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 200)
        h, w = self.base.shape[:2]
        self.zoom = min(cw / w, ch / h, 1.0)
        self.ox = (cw - w * self.zoom) / 2
        self.oy = (ch - h * self.zoom) / 2

    def _to_image(self, x, y):
        return (x - self.ox) / self.zoom, (y - self.oy) / self.zoom

    def _on_wheel(self, e):
        if self.base is None:
            return
        ix, iy = self._to_image(e.x, e.y)
        nz = float(np.clip(self.zoom * (1.18 if e.delta > 0 else 1 / 1.18), 0.04, 10.0))
        self.zoom = nz
        self.ox = e.x - ix * nz
        self.oy = e.y - iy * nz
        self.render()

    def _pan_start(self, e):
        self._pan = (e.x, e.y, self.ox, self.oy)

    def _pan_move(self, e):
        if not self._pan:
            return
        x0, y0, ox, oy = self._pan
        self.ox = ox + (e.x - x0)
        self.oy = oy + (e.y - y0)
        self.render()

    def render(self):
        c = self.canvas
        if self.base is None:
            return
        cw, ch = c.winfo_width(), c.winfo_height()
        if cw < 10 or ch < 10:
            return

        src = self.orig0 if self._show_orig else self.base
        z = self.zoom
        H, W = src.shape[:2]

        vx0 = int(np.floor(-self.ox / z))
        vy0 = int(np.floor(-self.oy / z))
        vx1 = int(np.ceil((cw - self.ox) / z))
        vy1 = int(np.ceil((ch - self.oy) / z))

        cx0, cy0 = max(0, vx0), max(0, vy0)
        cx1, cy1 = min(W, vx1), min(H, vy1)
        if cx1 <= cx0 or cy1 <= cy0:
            c.delete("all")
            return

        dw = max(1, int(round((cx1 - cx0) * z)))
        dh = max(1, int(round((cy1 - cy0) * z)))
        view = cv2.resize(src[cy0:cy1, cx0:cx1], (dw, dh),
                          interpolation=cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR)

        if (not self._show_orig) and self.show_mask.get() and self.mask is not None and self.mask.any():
            ms = cv2.resize(self.mask[cy0:cy1, cx0:cx1], (dw, dh),
                            interpolation=cv2.INTER_NEAREST)
            hit = ms > 0
            if hit.any():
                view = view.copy()
                view[hit] = (view[hit].astype(np.float32) * 0.45 +
                             np.array([60, 60, 240], np.float32) * 0.55).astype(np.uint8)

        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        c.delete("all")
        c.create_image(int(round(cx0 * z + self.ox)), int(round(cy0 * z + self.oy)),
                       image=self.photo, anchor="nw")
        self.zoom_lbl.config(text=f"缩放 {z * 100:.0f}%   {W}×{H}")

    # ------------------------------------------------------------ 涂抹
    def _on_press(self, e):
        if self.base is None:
            return
        ix, iy = self._to_image(e.x, e.y)
        w_img = max(1.0, self.brush.get() / max(self.zoom, 1e-6))
        self._cur = {"pts": [(ix, iy)], "w": w_img, "erase": self.tool.get() == "erase"}
        self._paint_temp(e.x, e.y, e.x, e.y)

    def _on_drag(self, e):
        if self._cur is None:
            return
        ix, iy = self._to_image(e.x, e.y)
        prev = self._cur["pts"][-1]
        self._cur["pts"].append((ix, iy))
        self._paint_temp(prev[0] * self.zoom + self.ox, prev[1] * self.zoom + self.oy, e.x, e.y)

    def _paint_temp(self, x0, y0, x1, y1):
        color = "#e5e7eb" if self._cur["erase"] else "#ef4444"
        item = self.canvas.create_line(x0, y0, x1, y1, fill=color,
                                       width=max(2, self.brush.get()),
                                       capstyle="round", stipple="gray50")
        self._temp_items.append(item)

    def _on_release(self, e):
        if self._cur is None:
            return
        stroke = self._cur
        self._cur = None
        for it in self._temp_items:
            self.canvas.delete(it)
        self._temp_items.clear()
        self._rasterize(stroke)
        self.strokes.append(stroke)
        self.render()

    def _rasterize(self, stroke: dict):
        if self.mask is None:
            return
        if stroke.get("type") == "auto":  # 自动识别结果作为一层整体叠加
            cv2.bitwise_or(self.mask, stroke["layer"], dst=self.mask)
            return
        val = 0 if stroke["erase"] else 255
        thick = max(1, int(round(stroke["w"])))
        pts = [(int(round(x)), int(round(y))) for x, y in stroke["pts"]]
        if len(pts) == 1:
            cv2.circle(self.mask, pts[0], max(1, thick // 2), val, -1, lineType=cv2.LINE_AA)
            return
        for i in range(1, len(pts)):
            cv2.line(self.mask, pts[i - 1], pts[i], val, thick, lineType=cv2.LINE_AA)
        for pt in (pts[0], pts[-1]):
            cv2.circle(self.mask, pt, max(1, thick // 2), val, -1, lineType=cv2.LINE_AA)

    def _rebuild_mask(self):
        if self.mask is None:
            return
        self.mask[:] = 0
        for s in self.strokes:
            self._rasterize(s)

    def undo_stroke(self):
        if not self.strokes:
            return
        self.strokes.pop()
        self._rebuild_mask()
        self.render()

    def clear_mask(self):
        if self.mask is None:
            return
        self.strokes.clear()
        self.mask[:] = 0
        self.render()

    # ------------------------------------------------------------ 右键：平移 + 单击切换
    def _r3_press(self, e):
        self._r3 = [e.x, e.y, self.ox, self.oy, False]

    def _r3_move(self, e):
        if not self._r3:
            return
        if abs(e.x - self._r3[0]) + abs(e.y - self._r3[1]) > 3:
            self._r3[4] = True
        self.ox = self._r3[2] + (e.x - self._r3[0])
        self.oy = self._r3[3] + (e.y - self._r3[1])
        self.render()

    def _r3_release(self, e):
        if self._r3 and not self._r3[4]:
            self.tool.set("erase" if self.tool.get() == "brush" else "brush")
            self._tool_changed()
            self._flash("已切换为" + ("橡皮" if self.tool.get() == "erase" else "画笔"))
        self._r3 = None

    # ------------------------------------------------------------ 引擎
    def _update_engine_choices(self):
        ready = core.has_model()
        keys, names = [], []
        for k, label in core.ENGINES.items():
            if k == "lama" and not ready:
                continue
            keys.append(k)
            names.append(label)
        self._engine_keys = keys
        self.cb_engine["values"] = names
        if names:
            idx = 0 if ready else len(names) - 1
            self.cb_engine.current(idx)
            self.engine_label.set(names[idx])

        if ready:
            self.model_box.pack_forget()
        else:
            self.model_box.pack(fill="x", padx=14, pady=(4, 2), before=self._scales["expand"].master)
        self._refresh_models()

    MODEL_OPTIONS = [
        ("标准 512（推荐 · 纹理清晰）", "lama_fp32.onnx"),
        ("大区域 1024（结构更稳 · 稍慢）", "lama_1024.onnx"),
        ("超大 2048（需另下模型）", "lama_2048.onnx"),
    ]

    def _refresh_models(self):
        opts = []
        for label, fn in self.MODEL_OPTIONS:
            p = os.path.join(core.MODEL_DIR, fn)
            if os.path.isfile(p) and os.path.getsize(p) > core.MIN_MODEL_BYTES:
                opts.append((label, p))
        self._model_opts = opts
        self.cb_model["values"] = [o[0] for o in opts]
        if not opts:
            self.model_label.set("")
            self.dev_lbl.config(text="推理设备：未加载模型")
            return
        idx = 0
        for i, (_, p) in enumerate(opts):
            if os.path.basename(p) == "lama_fp32.onnx":
                idx = i
        self.cb_model.current(idx)
        self.model_label.set(opts[idx][0])
        core.set_active_model(opts[idx][1])
        self._refresh_device_label()

    def _model_changed(self):
        i = self.cb_model.current()
        if 0 <= i < len(getattr(self, "_model_opts", [])):
            core.set_active_model(self._model_opts[i][1])
            self._refresh_device_label()
            self._flash("已切换模型：" + self.model_label.get())

    def _refresh_device_label(self):
        p = core.ACTIVE_MODEL or core.best_model_path()
        if not p:
            self.dev_lbl.config(text="推理设备：未加载模型")
            return
        name = os.path.basename(p)
        res = {"lama_2048.onnx": 2048, "lama_1024.onnx": 1024}.get(name, 512)
        dev = core.DEVICE_INFO.get("provider", "CPU")
        if not core.DEVICE_INFO.get("model"):
            # 模型还没加载，先看运行库有没有 GPU 后端
            try:
                import onnxruntime as ort
                dev = "GPU" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPU"
            except Exception:  # noqa: BLE001
                dev = "CPU"
        detail = "GPU 加速" if dev == "GPU" else "CPU"
        self.dev_lbl.config(text=f"{name}（{res}px） · 推理 {detail}")

    def current_engine(self) -> str:
        i = self.cb_engine.current()
        if 0 <= i < len(self._engine_keys):
            return self._engine_keys[i]
        return "telea"

    # ------------------------------------------------------------ 模型下载
    def download_model(self):
        if self._busy:
            return
        self._busy = True
        self.btn_model.config(state="disabled")
        self.pb.config(mode="determinate", value=0)
        self.pb_lbl.config(text="正在下载模型…")

        def worker():
            def cb(done, total):
                pct = int(done * 100 / total) if total else 0
                self.root.after(0, lambda: self._set_pb(pct, f"下载中 {pct}%"))

            try:
                core.download_model(cb)
                self.root.after(0, self._after_model)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                self.root.after(0, lambda: self._model_failed(msg))

        threading.Thread(target=worker, daemon=True).start()

    def _after_model(self):
        self._busy = False
        self.pb.config(value=100)
        self.pb_lbl.config(text="模型就绪，已启用 AI 精细修复")
        self._update_engine_choices()
        messagebox.showinfo("完成", "AI 模型已就绪，现在默认使用「AI 精细修复」。")

    def _model_failed(self, msg):
        self._busy = False
        self.btn_model.config(state="normal")
        self.pb_lbl.config(text="模型下载失败")
        messagebox.showerror("下载失败", msg + "\n\n可稍后重试，或先用快速修复。")

    # ------------------------------------------------------------ 修复
    def run_remove(self):
        if self._busy or self.base is None:
            return
        if self.mask is None or not self.mask.any():
            messagebox.showinfo("提示", "请先用画笔涂掉要去除的水印区域。")
            return
        engine = self.current_engine()
        if engine == "lama" and not core.has_model():
            messagebox.showinfo("提示", "AI 模型还没下载，请先点下载按钮，或改用快速修复。")
            return

        self._busy = True
        self.btn_run.config(state="disabled")
        self.pb.config(mode="determinate", value=0)
        self.pb_lbl.config(text="处理中…")

        base = self.base.copy()
        mask = self.mask.copy()
        expand = self.expand.get()
        feather = self.feather.get()
        hd = bool(self.hd.get())

        def cb(pct, text):
            self.root.after(0, lambda: self._set_pb(pct, text))

        def worker():
            try:
                out = core.remove_watermark(base, mask, engine=engine, expand=expand,
                                            feather=feather, hd=hd, progress=cb)
                self.root.after(0, lambda: self._apply_result(out))
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                self.root.after(0, lambda: self._run_failed(msg))

        threading.Thread(target=worker, daemon=True).start()

    def _set_pb(self, pct, text):
        self.pb.config(value=pct)
        self.pb_lbl.config(text=text)

    def _apply_result(self, out):
        self._busy = False
        self.btn_run.config(state="normal")
        self.history.append(self.base)
        self.base = out
        self.strokes.clear()
        if self.mask is not None:
            self.mask[:] = 0
        self.btn_undo_repair.config(state="normal")
        self.pb.config(value=100)
        self.pb_lbl.config(text="完成 · 可继续补涂，或保存结果")
        self._refresh_device_label()
        self.render()
        self._flash("已修复 · 有残留就继续涂；按住空格可对比原图")

    def _run_failed(self, msg):
        self._busy = False
        self.btn_run.config(state="normal")
        self.pb_lbl.config(text="处理失败")
        messagebox.showerror("处理失败", msg)

    def undo_repair(self):
        if not self.history:
            return
        self.base = self.history.pop()
        self.strokes.clear()
        if self.mask is not None:
            self.mask[:] = 0
        self.btn_undo_repair.config(state="normal" if self.history else "disabled")
        self.pb_lbl.config(text="已撤销上一步修复")
        self.render()

    def _flash(self, text: str):
        self.status.config(text=text)


def _frozen_selftest() -> int:
    """打包后自检：验证依赖、模型路径、拖拽库、GUI 能否创建。结果写入 exe 同目录。"""
    lines = []
    ok = True
    try:
        import platform
        lines.append(f"Python {platform.python_version()} {platform.machine()} "
                     f"{'64bit' if sys.maxsize > 2 ** 32 else '32bit'}")
        lines.append(f"frozen={getattr(sys, 'frozen', False)} exe={sys.executable}")
        lines.append(f"cv2={cv2.__version__} numpy={np.__version__}")

        lines.append(f"has_model={core.has_model()} model_dir={core.MODEL_DIR}")

        # 修复引擎自检
        h, w = 320, 320
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = np.dstack([(xx / w * 200 + 30), (yy / h * 180 + 40),
                         ((xx + yy) / (w + h) * 160 + 50)]).astype(np.uint8)
        img = cv2.GaussianBlur(img, (0, 0), 2)
        m = np.zeros((h, w), np.uint8)
        cv2.putText(m, "TEST", (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 8, cv2.LINE_AA)
        for eng in ("telea", "ns", "lama"):
            if eng == "lama" and not core.has_model():
                lines.append("engine lama: SKIP (no model)")
                continue
            try:
                out = core.remove_watermark(img, m, engine=eng, expand=2, feather=2)
                lines.append(f"engine {eng}: OK {out.shape}")
            except Exception as e:  # noqa: BLE001
                ok = False
                lines.append(f"engine {eng}: FAIL {type(e).__name__}: {e}")

        # 拖拽库 + GUI 自检
        try:
            from tkinterdnd2 import TkinterDnD, DND_FILES
            root = TkinterDnD.Tk()
            root.withdraw()
            c = tk.Canvas(root)
            c.drop_target_register(DND_FILES)
            ver = root.tk.call("package", "require", "tkdnd")
            lines.append(f"tkdnd: OK {ver}")
            root.destroy()
        except Exception as e:  # noqa: BLE001
            ok = False
            lines.append(f"tkdnd: FAIL {type(e).__name__}: {e}")

        # 自动识别自检
        try:
            dm, g = core.detect_watermark(img, 6)
            lines.append(f"detect: OK groups={g} px={int((dm > 0).sum())}")
        except Exception as e:  # noqa: BLE001
            ok = False
            lines.append(f"detect: FAIL {type(e).__name__}: {e}")

        lines.append("RESULT: " + ("PASS" if ok else "FAIL"))
    except Exception as e:  # noqa: BLE001
        lines.append(f"FATAL {type(e).__name__}: {e}")
        ok = False

    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
    path = os.path.join(base, "_selftest.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv:
        sys.exit(_frozen_selftest())

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001
        pass

    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()   # 支持文件拖入的 Tk 根窗口
        dnd_ok = True
    except Exception:  # noqa: BLE001
        root = tk.Tk()           # 没装 tkinterdnd2 时退化为普通窗口
        dnd_ok = False

    initial = sys.argv[1] if len(sys.argv) > 1 else None
    App(root, initial, dnd_ok)
    root.mainloop()


if __name__ == "__main__":
    main()
