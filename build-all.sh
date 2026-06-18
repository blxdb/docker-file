#!/bin/bash
set -euo pipefail

# ============================================================
# 主打包脚本 — 构建所有 Docker 镜像并推送到共绩算力仓库
# 用法: ./build-all.sh [--push] [--version X.X.X]
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 加载 .env（如果存在）
if [ -f .env ]; then
    source .env
fi

# 生成版本号（时间戳）
VERSION="${VERSION:-$(date +%Y%m%d_%H%M%S)}"
BUILD_DATE=$(date +%Y-%m-%d)
BUILD_DIR="$SCRIPT_DIR/build-records/$BUILD_DATE"
mkdir -p "$BUILD_DIR"

BUILD_LOG="$BUILD_DIR/build.log"
SUMMARY="$BUILD_DIR/summary.txt"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$BUILD_LOG"
}

log "===== 开始构建: $VERSION ====="

# 构建各应用镜像
build_image() {
    local name=$1
    local dockerfile_path=$2
    local context_dir=${3:-$(dirname "$dockerfile_path")}
    local tag="${REGISTRY}/${NAMESPACE}/${name}:${VERSION}"

    log "构建 $name → $tag"
    docker build \
        --platform "$DOCKER_PLATFORM" \
        -t "$tag" \
        -f "$dockerfile_path" \
        "$context_dir"

    log "$name 构建完成"

    # 保存到本地 images/
    mkdir -p "$SCRIPT_DIR/images"
    local tar_path="$SCRIPT_DIR/images/${name}_${VERSION}.tar"
    log "保存镜像到 $tar_path"
    docker save "$tag" -o "$tar_path"

    # 生成推送命令提示
    echo "docker push $tag" >> "$SUMMARY"
}

build_image "app-tts"   "$SCRIPT_DIR/app-tts/Dockerfile" "$SCRIPT_DIR"
build_image "app-draw"  "$SCRIPT_DIR/app-draw"

# 汇总
{
    echo "===== 构建汇总: $VERSION ====="
    echo "构建日期: $BUILD_DATE"
    echo ""
    echo "镜像列表:"
    echo "  ${REGISTRY}/${NAMESPACE}/app-tts:${VERSION}"
    echo "  ${REGISTRY}/${NAMESPACE}/app-draw:${VERSION}"
    echo ""
    echo "推送命令:"
    echo "  docker login ${REGISTRY} --username=${NAMESPACE}"
    echo "  docker push ${REGISTRY}/${NAMESPACE}/app-tts:${VERSION}"
    echo "  docker push ${REGISTRY}/${NAMESPACE}/app-draw:${VERSION}"
} | tee -a "$SUMMARY"

log "===== 构建完成 ====="

# 如果传了 --push 参数则直接推送
if [ "${1:-}" = "--push" ]; then
    log "开始推送镜像..."
    echo "${REGISTRY_PASSWORD}" | docker login "$REGISTRY" --username="$NAMESPACE" --password-stdin
    while IFS= read -r cmd; do
        log "执行: $cmd"
        eval "$cmd"
    done < <(grep 'docker push' "$SUMMARY")
    log "推送完成"
fi
