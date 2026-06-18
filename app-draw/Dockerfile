# ============================================================
# app-draw — Blender 手写动画 (InkSight → SVG → Grease Pencil)
# ============================================================
# 基础镜像: 待定（需 Blender + InkSight 环境）
# 框架阶段: 仅搭建目录结构，后续填充具体代码
# ============================================================

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base

WORKDIR /app

# ---- 系统依赖 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 依赖（占位，后续细化） ----
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ---- 应用代码 ----
COPY . /app

# ---- 入口 ----
CMD ["python3", "run.py"]
