#!/usr/bin/env python3
"""
app-draw — Blender 手写动画
=================================
工作流:
  1. 输入文本 → InkSight 生成手写 SVG
  2. LaTeX 引擎处理排版
  3. Blender Grease Pencil 生成动画

用法:
  python3 run.py --input text.txt --output /output
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("app-draw")


def main():
    parser = argparse.ArgumentParser(description="手写动画生成")
    parser.add_argument("--input", required=True, help="输入文本文件路径")
    parser.add_argument("--output", default="/output", help="输出目录")
    args = parser.parse_args()

    log.info("参数: input=%s output=%s", args.input, args.output)

    # ---- 以后在这里实现具体逻辑 ----


if __name__ == "__main__":
    main()
