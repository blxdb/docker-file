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

# ---- 克隆 IndexTTS2 源码（自动重试 3 次） ----
RUN for i in 1 2 3; do git clone https://github.com/IndexTeam/IndexTTS-2.git /app/IndexTTS-2 && break; sleep 5; done

# ---- Python 依赖 ----
COPY app-tts/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- 下载模型权重（构建时缓存，不提交 git） ----
# IndexTTS2 模型
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2', local_dir='/app/models/indextts2'); print('IndexTTS2 模型已下载')"
# Whisper 模型
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('openai/whisper-large-v3-turbo', local_dir='/app/models/whisper'); print('Whisper 模型已下载')"
# 中文对齐模型
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn', local_dir='/app/models/align-zh'); print('对齐模型已下载')"

# ---- 参考音频 & 输入文本 ----
COPY voices /app/voices
COPY app-tts/t.txt /app/t.txt

# ---- 入口代码 ----
COPY app-tts/run.py /app/run.py
RUN chmod +x /app/run.py

# ---- 输出目录 ----
VOLUME ["/output"]

ENTRYPOINT ["python", "/app/run.py"]
