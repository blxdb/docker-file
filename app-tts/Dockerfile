# ============================================================
# app-tts — 业务代码层（构建极快，几秒完成）
# 依赖 harbor 上已构建好的 base 镜像
# ============================================================
FROM harborpush.suanleme.cn/wangduoduo2026/app-tts-base:latest

LABEL description="IndexTTS2 配音 + Whisper 字幕对齐"

WORKDIR /app

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV HF_ENDPOINT=https://hf-mirror.com

# ---- 参考音频 & 输入文本 ----
COPY voices /app/voices
COPY app-tts/t.txt /app/t.txt

# ---- 入口代码 ----
COPY app-tts/run.py /app/run.py
RUN chmod +x /app/run.py

VOLUME ["/output"]
ENTRYPOINT ["python", "/app/run.py"]
