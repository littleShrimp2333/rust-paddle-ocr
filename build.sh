#!/usr/bin/env bash
# build.sh — 构建 PP-OCR 独立分发自包含产物 (v4 / v5 / v6)
#
# 用法:
#   ./build.sh              # 构建全部三个版本
#   ./build.sh v6           # 只构建 v6
#   ./build.sh v4 v6        # 构建 v4 和 v6
#   ./build.sh --release    # release 模式构建全部
#   ./build.sh --target x86_64-unknown-linux-gnu v6  # 交叉编译 Linux 版本
#
# 产物:
#   dist/ocr-cli-v4    (v4 模型内嵌, ~16MB)
#   dist/ocr-cli-v5    (v5 模型内嵌, ~27MB)
#   dist/ocr-cli-v6    (v6 模型内嵌, ~35MB)

set -euo pipefail

# ── 配置 ──

RELEASE=""
TARGET=""
FEATURES_STATIC=""
VERSIONS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release)
            RELEASE="--release"
            shift
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --target=*)
            TARGET="${1#*=}"
            shift
            ;;
        --static)
            FEATURES_STATIC=",static-libstdcpp"
            shift
            ;;
        v4|v5|v6)
            VERSIONS+=("$1")
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--release] [--target <triple>] [--static] [v4] [v5] [v6]"
            exit 1
            ;;
    esac
done

# 默认构建全部版本
if [[ ${#VERSIONS[@]} -eq 0 ]]; then
    VERSIONS=("v4" "v5" "v6")
fi

# ── 环境检查 ──

if ! command -v cargo &>/dev/null; then
    echo "错误: 找不到 cargo, 请先安装 Rust"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="$PROJECT_ROOT/dist"
BUILD_TYPE="${RELEASE:+release}"
BUILD_TYPE="${BUILD_TYPE:-debug}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║    OCR CLI 构建脚本 — 独立分发自包含产物             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  目标:    ${TARGET:-native}"
echo "  模式:    ${BUILD_TYPE}"
echo "  版本:    ${VERSIONS[*]}"
echo "  静态链接: ${FEATURES_STATIC:+是}"
echo ""

# ── 检查模型文件 ──

check_model() {
    local ver="$1"
    local det rec keys
    case "$ver" in
        v4) det="models/ch_PP-OCRv4_det_infer.mnn"
            rec="models/ch_PP-OCRv4_rec_infer.mnn"
            keys="models/ppocr_keys_v4.txt" ;;
        v5) det="models/PP-OCRv5_mobile_det.mnn"
            rec="models/PP-OCRv5_mobile_rec.mnn"
            keys="models/ppocr_keys_v5.txt" ;;
        v6) det="models/PP-OCRv6_small_det.mnn"
            rec="models/PP-OCRv6_small_rec.mnn"
            keys="models/ppocr_keys_v6.txt" ;;
    esac
    local ok=true
    for f in "$det" "$rec" "$keys"; do
        if [[ ! -f "$PROJECT_ROOT/$f" ]]; then
            echo "  ✗ 缺少: $f"
            ok=false
        fi
    done
    if $ok; then
        echo "  ✓ $ver — 模型文件齐全"
    else
        echo "  ✗ $ver — 模型文件缺失，跳过"
        return 1
    fi
}

echo "── 检查模型文件 ──"
VALID_VERSIONS=()
for ver in "${VERSIONS[@]}"; do
    if check_model "$ver"; then
        VALID_VERSIONS+=("$ver")
    fi
done
echo ""

if [[ ${#VALID_VERSIONS[@]} -eq 0 ]]; then
    echo "错误: 没有任何版本的模型文件齐全，无法构建"
    exit 1
fi

# ── 构建 ──

mkdir -p "$DIST_DIR"

TARGET_FLAG=""
if [[ -n "$TARGET" ]]; then
    TARGET_FLAG="--target $TARGET"
fi

RELEASE_FLAG=""
if [[ -n "$RELEASE" ]]; then
    RELEASE_FLAG="--release"
fi

for ver in "${VALID_VERSIONS[@]}"; do
    echo "── 构建 ocr-cli-$ver ──"

    FEATURE="bundle-models-${ver}${FEATURES_STATIC}"

    cargo build $RELEASE_FLAG $TARGET_FLAG --features "$FEATURE" 2>&1 | tail -3

    # 确定产物路径
    if [[ -n "$TARGET" ]]; then
        SRC="target/$TARGET/$BUILD_TYPE/ocr-cli"
    else
        SRC="target/$BUILD_TYPE/ocr-cli"
    fi

    DST="$DIST_DIR/ocr-cli-$ver"
    if [[ -n "$TARGET" ]]; then
        DST="$DIST_DIR/ocr-cli-$ver-${TARGET}"
    fi

    cp "$SRC" "$DST"
    chmod +x "$DST"

    SIZE=$(ls -lh "$DST" | awk '{print $5}')
    echo "  ✓ $DST ($SIZE)"
    echo ""
done

echo "── 构建完成 ──"
echo ""
ls -lh "$DIST_DIR"/ocr-cli-* 2>/dev/null || true
echo ""
echo "用法:"
for ver in "${VALID_VERSIONS[@]}"; do
    echo "  $DIST_DIR/ocr-cli-$ver photo.jpg"
done
