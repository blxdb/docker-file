# ============================================================
# app-tts — IndexTTS2 + Whisper 配音 & 字幕对齐
# 基础镜像：PyTorch + CUDA
# ============================================================
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

LABEL description="IndexTTS2 配音 + Whisper 字幕对齐"

WORKDIR /app

# ---- 系统依赖 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- 克隆 IndexTTS2 源码 ----
# 使用 ghfast.top 代理加速国内访问
RUN git clone https://ghfast.top/https://github.com/IndexTeam/IndexTTS-2.git /app/IndexTTS-2 \
    || git clone https://github.com/IndexTeam/IndexTTS-2.git /app/IndexTTS-2

# 设置 HuggingFace 镜像站（国内加速）
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DOWNLOAD_TIMEOUT=120

# ---- Python 依赖 ----
COPY app-tts/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- 预下载 WhisperX 中文对齐模型（避免运行时下载慢） ----
RUN python -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
# 中文对齐模型
snapshot_download('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn',
                  local_dir='/app/models/align-zh')
print('对齐模型已下载')
"

# ---- 模型权重（本地预下载后打包） ----
COPY models/indextts2 /app/models/indextts2
COPY models/whisper /app/models/whisper

# ---- 参考音频 & 输入文本 ----
COPY voices /app/voices
COPY app-tts/t.txt /app/t.txt

# ---- 入口代码 ----
COPY app-tts/run.py /app/run.py
RUN chmod +x /app/run.py

# ---- 输出目录 ----
VOLUME ["/output"]

ENTRYPOINT ["python", "/app/run.py"]
