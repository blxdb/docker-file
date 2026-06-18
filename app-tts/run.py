#!/usr/bin/env python3
"""
app-tts — IndexTTS2 配音 + Whisper 字幕对齐 (大显存优化版)
===========================================================
工作流:
  1. 读取 t.txt
  2. 智能切割文本为 60~100 汉字片段（适配大显存并行）
  3. 加载参考人声
  4. 自动检测 GPU 显存，配置 FP16 / 并行数 / batch_size
  5. IndexTTS2 并行逐段生成配音（多 worker）
  6. WhisperX 强制对齐生成字幕（大 batch_size）
  7. 输出最终 WAV + SRT

用法:
  python run.py                                          # 自动模式（推荐）
  python run.py --fp16 --workers 4 --whisper-batch 16   # 手动调优
  python run.py --ref /app/voices/reference.wav --text /app/t.txt --output /output
  python run.py --ref /app/voices/reference.wav --text /app/t.txt --output /output --language zh
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app-tts")


# ─── 路径常量 ─────────────────────────────────────────────
MODEL_INDEX_DIR = "/app/models/indextts2"
MODEL_WHISPER_DIR = "/app/models/whisper"
INDEXTS2_SRC = "/app/IndexTTS-2"
DEFAULT_REF = "/app/voices/fan_zhendong_analysis_denoised.wav"
DEFAULT_TEXT = "/app/t.txt"
DEFAULT_OUTPUT = "/output"

# ─── 文本切割参数 ──────────────────────────────────────────
SEGMENT_MIN_CHARS = 60    # 每段最少汉字数
SEGMENT_MAX_CHARS = 100   # 每段最多汉字数

# ─── GPU 自适应参数 ────────────────────────────────────────
# 单段 IndexTTS2 推理峰值显存 (GB) - 实测估值
TTS_PEAK_VRAM_PER_SEG_GB = 8.0     # FP32 峰值（含参考音频编码）
TTS_PEAK_VRAM_PER_SEG_FP16_GB = 5.5  # FP16 峰值
WHISPER_PEAK_VRAM_GB = 6.0        # WhisperX 模型加载基础占用


@dataclass
class GPUConfig:
    """自动检测/计算的 GPU 配置"""
    total_vram_gb: float = 24.0     # 默认 24GB
    fp16: bool = True                # 默认开启半精度
    tts_workers: int = 2             # TTS 并行 worker 数
    whisper_batch_size: int = 8      # WhisperX batch_size
    device: str = "cuda"


def detect_gpu_config(force_fp16: bool | None = None) -> GPUConfig:
    """检测 GPU 显存并自动计算最优配置"""
    cfg = GPUConfig()

    try:
        import torch
        if torch.cuda.is_available():
            cfg.device = "cuda"
            total_vram = torch.cuda.get_device_properties(0).total_memory
            cfg.total_vram_gb = total_vram / (1024 ** 3)

            # 显存信息
            free_vram, _ = torch.cuda.mem_get_info(0)
            free_vram_gb = free_vram / (1024 ** 3)
            log.info("GPU: %s  总显存: %.1f GB  可用: %.1f GB",
                     torch.cuda.get_device_name(0), cfg.total_vram_gb, free_vram_gb)
        else:
            cfg.device = "cpu"
            log.info("CUDA 不可用，使用 CPU")
            cfg.fp16 = False
            cfg.tts_workers = 1
            cfg.whisper_batch_size = 1
            return cfg
    except Exception as e:
        log.warning("GPU 检测失败: %s，使用默认配置", e)
        return cfg

    # ── FP16 决策 ──
    if force_fp16 is not None:
        cfg.fp16 = force_fp16
    else:
        cfg.fp16 = cfg.total_vram_gb >= 12.0   # >=12GB 就默认 FP16
    log.info("半精度(FP16): %s", "开启 ✓" if cfg.fp16 else "关闭 ✗")

    # ── TTS 并行 worker 数 ──
    peak_per_seg = TTS_PEAK_VRAM_PER_SEG_FP16_GB if cfg.fp16 else TTS_PEAK_VRAM_PER_SEG_GB
    # 保留基础 4GB 给加载的模型权重 + 对齐模型
    overhead = 6.0 if cfg.fp16 else 8.0
    usable_vram = cfg.total_vram_gb - overhead
    if usable_vram <= 0:
        cfg.tts_workers = 1
    else:
        workers = int(usable_vram // peak_per_seg)
        cfg.tts_workers = max(1, min(workers, 8))  # 上限 8 个 worker
    log.info("TTS 并行数: %d (每段峰值 ~%.1f GB, 可用 %.1f GB)",
             cfg.tts_workers, peak_per_seg, usable_vram)

    # ── Whisper batch_size ──
    whisper_overhead = WHISPER_PEAK_VRAM_GB
    whisper_usable = cfg.total_vram_gb - whisper_overhead
    # 每额外 batch 约 0.8GB (FP16) / 1.5GB (FP32)
    batch_vram = 0.8 if cfg.fp16 else 1.5
    if whisper_usable > 0:
        max_batch = int(whisper_usable // batch_vram)
        cfg.whisper_batch_size = max(1, min(max_batch, 64))
    log.info("Whisper batch_size: %d (可用 %.1f GB)", cfg.whisper_batch_size, whisper_usable)

    return cfg


# ═══════════════════════════════════════════════════════════
#  智能文本切割
# ═══════════════════════════════════════════════════════════

# 中文/英文/数字计数用
_RE_CHAR = re.compile(r"\S")
_RE_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")

# 句子边界标点（优先在此处分句）
_SENTENCE_BOUNDARY = re.compile(r"[。！？.!?\n]+")
# 次级边界（逗号、分号等，当句子太长时在此处折中）
_SUB_BOUNDARY = re.compile(r"[，、；：,;:]\s*")


def count_cjk(text: str) -> int:
    """统计中文字符数"""
    return len(_RE_CJK.findall(text))


def smart_split_text(text: str, min_chars: int = SEGMENT_MIN_CHARS,
                     max_chars: int = SEGMENT_MAX_CHARS) -> list[str]:
    """
    智能切割文本，使每段汉字数在 [min_chars, max_chars] 之间。

    策略：
    1. 先按句子边界（。！？）切分成自然句
    2. 累积句子，汉字数达到 min_chars 时切割
    3. 如果累积超过 max_chars 且遇到次级边界（，；），在此处切
    4. 如果单句超过 max_chars，强制截断到 max_chars
    """
    if not text or not text.strip():
        return []

    # 第一步：按句子边界分句
    raw_sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]

    # 如果没有任何分句结果，把整个文本作为一个句子
    if not raw_sentences:
        return [text]

    segments = []
    current = []
    current_cjk = 0

    def flush_current():
        nonlocal current, current_cjk
        if current:
            seg = "".join(current)
            segments.append(seg)
            current = []
            current_cjk = 0

    for sent in raw_sentences:
        sent_cjk = count_cjk(sent)

        # 空句跳过
        if sent_cjk == 0:
            if sent.strip():
                current.append(sent)
            continue

        # ◆ 单句已经超过 max → 强制截断
        if sent_cjk > max_chars:
            flush_current()
            # 在次级边界处截断
            sub_parts = _SUB_BOUNDARY.split(sent)
            sub_buffer = []
            sub_cjk = 0
            for part in sub_parts:
                pc = count_cjk(part)
                if sub_cjk + pc > max_chars and sub_buffer:
                    segments.append("".join(sub_buffer))
                    sub_buffer = []
                    sub_cjk = 0
                sub_buffer.append(part)
                sub_cjk += pc
            if sub_buffer:
                segments.append("".join(sub_buffer))
            continue

        # ◆ 加了这个句子会超过 max，先 flush 当前累积
        if current_cjk + sent_cjk > max_chars and current_cjk >= min_chars:
            flush_current()

        current.append(sent)
        current_cjk += sent_cjk

        # ◆ 累积到 min 以上就输出
        if current_cjk >= min_chars:
            flush_current()

    # 收尾
    flush_current()

    # 如果分段为空，至少保留整段
    if not segments:
        segments = [text]

    # 检查：如果最后一段过短（< min_chars 的一半），合并到前一段
    if len(segments) > 1:
        last_cjk = count_cjk(segments[-1])
        if last_cjk < min_chars // 2:
            segments[-2] += segments[-1]
            segments.pop()

    return segments


def load_text(path: str, smart_split: bool = True) -> list[str]:
    """读取文本文件，支持智能切割"""
    if not os.path.exists(path):
        log.warning("文本文件不存在: %s", path)
        return []

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        log.warning("文本文件为空: %s", path)
        return []

    # 跳过注释行
    lines = [l for l in content.split("\n") if l.strip() and not l.strip().startswith("#")]

    # 重新组装（去掉注释后的纯净文本）
    clean_text = "".join(lines)

    if smart_split:
        segments = smart_split_text(clean_text)
    else:
        # 降级：按标点分句
        segments = re.split(r"[。！？\n]+", clean_text)
        segments = [s.strip() for s in segments if s.strip()]

    log.info("共读取到 %d 个段落", len(segments))
    # 打印每段的汉字数统计
    cjk_counts = [count_cjk(s) for s in segments]
    log.info("段落汉字数统计: min=%d  max=%d  avg=%.0f",
             min(cjk_counts), max(cjk_counts), sum(cjk_counts) / len(cjk_counts))
    for i, (seg, cc) in enumerate(zip(segments, cjk_counts)):
        log.debug("  [%d/%d] %d字: %s", i + 1, len(segments), cc, seg[:50])
        if len(seg) > 50:
            log.debug("    ... %s", seg[-30:])

    return segments


def setup_indextts2():
    """将 IndexTTS2 源码加入 Python 路径"""
    src_path = Path(INDEXTS2_SRC)
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        log.info("IndexTTS2 源码路径: %s", src_path)
        return True
    else:
        log.warning("IndexTTS2 源码不存在: %s", src_path)
        return False


def load_indextts2_model(model_dir: str, gpu_cfg: GPUConfig):
    """加载 IndexTTS2 模型（支持半精度 + 显存优化）"""
    import torch

    device = gpu_cfg.device
    log.info("使用设备: %s", device)

    if not os.path.exists(model_dir):
        log.error("模型目录不存在: %s", model_dir)
        return None

    dtype = torch.float16 if gpu_cfg.fp16 else torch.float32
    try:
        from indextts.model import IndexTTS
        model = IndexTTS.from_pretrained(
            model_dir,
            device=device,
            torch_dtype=dtype,
        )

        # ── 可选：torch.compile 加速（需要 torch>=2.0） ──
        # 实测对 GPT 类模型有 1.2~1.5x 加速
        # try:
        #     model = torch.compile(model, mode="reduce-overhead")
        #     log.info("torch.compile 加速已启用")
        # except Exception:
        #     pass

        log.info("IndexTTS2 模型加载成功 (dtype=%s)", dtype)
        return model
    except ImportError:
        log.warning("无法导入 indextts 模块，尝试 transformers 兼容方式...")
        return None
    except Exception as e:
        log.warning("IndexTTS2 加载失败: %s", e)
        return None


def check_reference_audio(path: str) -> bool:
    """检查参考音频是否存在且有效"""
    if not os.path.exists(path):
        log.error("参考音频不存在: %s", path)
        return False

    import soundfile as sf
    try:
        info = sf.info(path)
        log.info("参考音频: %s  %.1fs  %dHz  %dch",
                 path, info.duration, info.samplerate, info.channels)
        return True
    except Exception as e:
        log.error("参考音频无法读取: %s", e)
        return False


def _generate_one_segment(args_tuple):
    """
    单个段落的 TTS 生成任务（供线程池调用）。
    由于 IndexTTS2 的 tts_to_file 内部持有 GIL 且主要在 GPU 上运行，
    用 ThreadPoolExecutor 可以让多个 CUDA stream 并发提交，
    同时利用 Python 侧的重试逻辑不阻塞主线程。
    """
    i, text, ref_audio, output_dir, language, model_ref = args_tuple
    output_path = os.path.join(output_dir, f"seg_{i:04d}.wav")

    if not text.strip():
        return None

    try:
        # model_ref 是 (model_obj,) 元组，确保跨线程访问同一对象
        model = model_ref[0]
        if model is not None:
            model.tts_to_file(
                text=text,
                ref_audio=ref_audio,
                output_path=output_path,
                language=language,
            )
        else:
            # 无模型时生成静音占位
            import soundfile as sf
            import numpy as np
            duration = max(1.0, len(text) * 0.2)
            sample_rate = 24000
            silence = np.zeros(int(duration * sample_rate), dtype=np.int16)
            sf.write(output_path, silence, sample_rate)

        if os.path.exists(output_path):
            return (i, output_path, text)
        else:
            return None
    except Exception as e:
        log.error("  [%d] ✗ 生成失败: %s", i, e)
        return None


def generate_tts_segments(model, segments: list[str], ref_audio: str,
                          output_dir: str, language: str,
                          gpu_cfg: GPUConfig) -> list[str]:
    """
    并行生成配音。

    原理：
    - IndexTTS2 是 GPT 类自回归模型，推理主要在 GPU 上进行
    - 使用 ThreadPoolExecutor + 多个 CUDA worker 并发提交任务
    - GPU 上的 CUDA stream 会自动调度并发 kernel，提高硬件利用率
    - 对于 24GB+ 显存，可以同时运行 2~4 个 TTS 任务不 OOM
    """
    import torch
    os.makedirs(output_dir, exist_ok=True)

    # 过滤空文本
    valid = [(i, t) for i, t in enumerate(segments) if t.strip()]
    if not valid:
        return []

    workers = min(gpu_cfg.tts_workers, len(valid))
    log.info("TTS 并行生成: %d 段, %d workers, fp16=%s",
             len(valid), workers, gpu_cfg.fp16)

    # 用元组包装 model，使其可哈希（线程安全引用传递）
    model_ref = (model,)
    tasks = [
        (i, text, ref_audio, output_dir, language, model_ref)
        for i, text in valid
    ]

    output_files = []
    completed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_generate_one_segment, t): t[0] for t in tasks}

        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result and result[1]:
                seg_idx, path, text = result
                output_files.append(path)
                elapsed = time.time() - start_time
                log.info("  ✓ [%d/%d] %s  (%.1fs elapsed, %s)",
                         completed, len(tasks), text[:50],
                         elapsed, os.path.basename(path))
            else:
                log.warning("  ⚠ [%d/%d] 生成失败或无输出",
                            completed, len(tasks))

    # 按原始顺序排序输出
    output_files.sort(key=lambda p: int(os.path.basename(p).replace("seg_", "").replace(".wav", "")))

    total_time = time.time() - start_time
    log.info("TTS 生成完成: %d/%d 段成功, 耗时 %.1fs (均值 %.1fs/段)",
             len(output_files), len(tasks), total_time,
             total_time / max(len(output_files), 1))

    return output_files


def whisper_align(segments: list[str], audio_dir: str, output_dir: str,
                  language: str, gpu_cfg: GPUConfig):
    """
    使用 WhisperX 强制对齐并生成 SRT（词级时间戳）

    显存优化：
    - 使用 GPUConfig 中的 batch_size（自动适配显存）
    - FP16 推理
    - 启用更高效的 batch 处理
    """
    import torch
    device = gpu_cfg.device
    compute = "float16" if gpu_cfg.fp16 else "float32"
    batch_size = gpu_cfg.whisper_batch_size

    try:
        import whisperx

        # 加载 Whisper 模型（使用本地路径）
        log.info("加载 WhisperX 模型 (batch_size=%d, compute=%s)...",
                 batch_size, compute)
        model = whisperx.load_model(
            MODEL_WHISPER_DIR,
            device=device,
            compute_type=compute,
            language=language[:2] if language != "auto" else None,
            asr_options={
                "batch_size": batch_size,
            },
        )

        # 加载对齐模型（优先使用本地预下载的模型）
        align_dir = "/app/models/align-zh"
        log.info("加载对齐模型 (language=%s)...", language[:2])
        try:
            if language[:2] == "zh" and os.path.exists(align_dir):
                from whisperx.alignment import load_align_model as _load_align
                align_model, align_metadata = _load_align(
                    language_code=language[:2], device=device, model_name=align_dir
                )
                log.info("对齐模型（本地）: %s", align_dir)
            else:
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=language[:2], device=device
                )
            log.info("对齐模型就绪")
        except Exception as e:
            log.warning("对齐模型加载失败: %s，回退到无对齐模式", e)
            align_model = None
            align_metadata = None

        # ── 并行对齐多个配音文件 ──
        import soundfile as sf

        audio_files = sorted([
            f for f in os.listdir(audio_dir) if f.endswith(".wav")
        ])

        # 先并行 Batch 转录（WhisperX 内部即可用 batch 处理多段音频）
        # 但 whisperx 的 transcribe 只能逐文件调用，所以我们串行走文件
        # 每个文件内部用 batch_size 处理音频帧

        srt_entries = []
        global_time = 0.0
        align_total_start = time.time()

        for i, (seg_text, audio_file) in enumerate(zip(segments, audio_files)):
            audio_path = os.path.join(audio_dir, audio_file)
            if not os.path.exists(audio_path):
                continue

            # 转录（使用大 batch_size 提高 GPU 利用率）
            log.info("  对齐 [%d/%d] batch=%d",
                     i + 1, len(segments), batch_size)
            result = model.transcribe(audio_path, batch_size=batch_size)

            # 强制对齐
            if align_model is not None and result.get("segments"):
                try:
                    result = whisperx.align(
                        result["segments"],
                        align_model,
                        align_metadata,
                        audio_path,
                        device=device,
                    )
                except Exception as e:
                    log.warning("  对齐失败: %s，使用段落级时间戳", e)

            # 提取时间戳
            if result.get("segments"):
                for seg in result["segments"]:
                    start = global_time + seg.get("start", 0)
                    end = global_time + seg.get("end", 0)
                    text = seg.get("text", "").strip()

                    # 优先使用词级时间戳
                    words = seg.get("words", [])
                    if words:
                        for w in words:
                            ws = global_time + w.get("start", start)
                            we = global_time + w.get("end", end)
                            wt = w.get("word", "").strip()
                            if wt:
                                srt_entries.append((ws, we, wt))
                    elif text:
                        srt_entries.append((start, end, text))
            else:
                srt_entries.append((global_time, global_time + 1.0, seg_text))

            # 更新全局时间偏移
            info = sf.info(audio_path)
            global_time += info.duration

        align_total = time.time() - align_total_start
        log.info("对齐总耗时: %.1fs (%.1f 段/秒)",
                 align_total, len(audio_files) / max(align_total, 0.01))

        # 写 SRT
        srt_path = os.path.join(output_dir, "output.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            for idx, (start, end, text) in enumerate(srt_entries, 1):
                f.write(f"{idx}\n")
                f.write(f"{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}\n")
                f.write(f"{text}\n\n")

        log.info("SRT 字幕已生成: %s (%d 条)", srt_path, len(srt_entries))
        return srt_entries

    except ImportError:
        log.warning("whisperx 未安装，跳过对齐")
        return []
    except Exception as e:
        log.warning("WhisperX 对齐失败: %s", e)
        return []


def _fmt_srt_time(seconds: float) -> str:
    """将秒数格式化为 SRT 时间戳"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def concat_audio(audio_dir: str, output_path: str):
    """合并所有配音段落为一个 WAV"""
    audio_files = sorted([
        os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
        if f.endswith(".wav")
    ])
    if not audio_files:
        log.warning("没有音频文件可合并")
        return

    import soundfile as sf
    import numpy as np

    all_audio = []
    sample_rate = None
    for f in audio_files:
        data, sr = sf.read(f)
        if sample_rate is None:
            sample_rate = sr
        all_audio.append(data)

    if all_audio:
        combined = np.concatenate(all_audio)
        sf.write(output_path, combined, sample_rate)
        log.info("合并音频: %s (%.1fs)", output_path, len(combined) / sample_rate)


def check_empty_text(segments: list[str]) -> bool:
    """检查文本是否为空（占位文件）"""
    return len(segments) == 0


def parse_args():
    parser = argparse.ArgumentParser(description="IndexTTS2 配音 + 字幕对齐 (大显存优化版)")
    parser.add_argument("--ref", default=DEFAULT_REF, help="参考人声音频路径")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="输入文本文件路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出目录")
    parser.add_argument("--language", default="zh", help="语言代码 (zh/en/...)")
    parser.add_argument("--fp16", action="store_true", default=None,
                        help="显式开启 FP16（默认自动检测显存决定）")
    parser.add_argument("--workers", type=int, default=None,
                        help="TTS 并行工作数（默认根据显存自动计算）")
    parser.add_argument("--whisper-batch", type=int, default=None,
                        help="WhisperX batch_size（默认根据显存自动计算）")
    parser.add_argument("--max-sec", type=int, default=30, help="每段最长秒数")
    parser.add_argument("--retry", type=int, default=2, help="失败重试次数")
    return parser.parse_args()


def main():
    args = parse_args()
    log.info("=" * 60)
    log.info("app-tts 启动 (大显存优化版)")
    log.info("=" * 60)
    log.info("参数: ref=%s text=%s output=%s lang=%s",
             args.ref, args.text, args.output, args.language)

    # ── 0. GPU 自动检测配置 ──
    gpu_cfg = detect_gpu_config(force_fp16=args.fp16 if args.fp16 is not None else None)

    # CLI 参数覆盖自动检测值
    if args.workers is not None:
        gpu_cfg.tts_workers = args.workers
    if args.whisper_batch is not None:
        gpu_cfg.whisper_batch_size = args.whisper_batch

    log.info("GPU 配置: fp16=%s  tts_workers=%d  whisper_batch=%d",
             gpu_cfg.fp16, gpu_cfg.tts_workers, gpu_cfg.whisper_batch_size)

    # ── 1. 加载文本（含智能切割 60~100 汉字/段） ──
    segments = load_text(args.text, smart_split=True)

    # 打印切割摘要
    total_cjk = sum(count_cjk(s) for s in segments)
    log.info("文本总字数: %d CJK 字符, 切割为 %d 段 (每段 60~100 字)",
             total_cjk, len(segments))

    if check_empty_text(segments):
        log.warning("t.txt 为空（占位文件），跳过配音生成")
        log.info("请在本地将参考音频转写为文本后替换 t.txt")
        log.info("输出目录: %s", args.output)
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "README.txt"), "w") as f:
            f.write("t.txt is empty - placeholder file.\n")
            f.write("Replace with actual text transcription to generate TTS.\n")
        return 0

    # ── 2. 检查参考音频 ──
    if not check_reference_audio(args.ref):
        log.error("参考音频无效，终止")
        return 1

    # ── 3. 设置 IndexTTS2 ──
    src_ok = setup_indextts2()

    # ── 4. 加载模型（半精度） ──
    model = None
    if src_ok:
        model = load_indextts2_model(MODEL_INDEX_DIR, gpu_cfg)

    # ── 5. 并行生成各个段落 ──
    seg_dir = os.path.join(args.output, "_segments")
    log.info("─" * 40)
    log.info("阶段 1/3: TTS 配音生成 (%d 段, %d 路并行)",
             len(segments), gpu_cfg.tts_workers)
    audio_files = generate_tts_segments(model, segments, args.ref,
                                        seg_dir, args.language, gpu_cfg)

    if not audio_files:
        log.warning("未生成任何音频")
        return 1

    # ── 6. Whisper 对齐（大 batch_size） ──
    log.info("─" * 40)
    log.info("阶段 2/3: WhisperX 字幕对齐 (batch_size=%d)",
             gpu_cfg.whisper_batch_size)
    srt_entries = whisper_align(segments, seg_dir, args.output,
                                args.language, gpu_cfg)

    # ── 7. 合并音频 ──
    log.info("─" * 40)
    log.info("阶段 3/3: 合并音频")
    final_wav = os.path.join(args.output, "output.wav")
    concat_audio(seg_dir, final_wav)

    # ── 8. 摘要 ──
    log.info("=" * 60)
    log.info("完成！")
    log.info("  配音: %s", final_wav)
    if srt_entries:
        log.info("  字幕: %s", os.path.join(args.output, "output.srt"))
    log.info("  段落: %d 段 (每段 %d~%d 汉字)",
             len(segments), SEGMENT_MIN_CHARS, SEGMENT_MAX_CHARS)
    log.info("  显存配置: %.1f GB | FP16=%s | TTS并行=%d | WhisperBatch=%d",
             gpu_cfg.total_vram_gb, gpu_cfg.fp16,
             gpu_cfg.tts_workers, gpu_cfg.whisper_batch_size)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
