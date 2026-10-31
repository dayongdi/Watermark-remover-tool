# -*- coding: utf-8 -*-
"""
下载 AI 模型
================================================
模型体积 198~208MB，超过 GitHub 单文件限制，所以不随仓库分发，用这个脚本拉取。

用法：
    python download_models.py          # 下载默认的 512 模型（推荐）
    python download_models.py 512      # 同上
    python download_models.py 1024     # 下载大区域 1024 模型
    python download_models.py all      # 两个都下

脚本用 8 线程分块并发下载（国内镜像单连接只有几十 KB/s），支持断点续传式的分块重试。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

TARGETS = {
    "512": core.MODEL_PATH,
    "1024": core.MODEL_1024,
}


def human(n: float) -> str:
    return f"{n / 1048576:.1f} MB"


def fetch(path: str) -> bool:
    if os.path.isfile(path) and os.path.getsize(path) > core.MIN_MODEL_BYTES:
        print(f"✓ 已存在，跳过：{os.path.basename(path)}（{human(os.path.getsize(path))}）")
        return True
    print(f"↓ 下载 {os.path.basename(path)} …")

    def cb(done: int, total: int):
        pct = 100.0 * done / total if total else 0
        sys.stdout.write(f"\r  {human(done)} / {human(total)}  ({pct:5.1f}%)")
        sys.stdout.flush()

    try:
        core.download_model(progress_cb=cb, target=path)
    except Exception as e:  # noqa: BLE001
        print(f"\n✗ 失败：{e}")
        return False
    print(f"\n✓ 完成：{path}")
    return True


def main():
    arg = (sys.argv[1] if len(sys.argv) > 1 else "512").lower()
    if arg == "all":
        keys = list(TARGETS)
    elif arg in TARGETS:
        keys = [arg]
    else:
        print(__doc__)
        sys.exit(1)

    ok = all(fetch(TARGETS[k]) for k in keys)
    if ok:
        print("\n模型就绪，运行 启动.bat 或 python app.py 即可。")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
