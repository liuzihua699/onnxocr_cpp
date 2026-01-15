#!/bin/bash

# OnnxOCR - ONNX Runtime 安装脚本
# 适用于 CentOS 7 / Ubuntu 等 Linux 系统

set -e

VERSION="1.16.3"
INSTALL_DIR="/usr/local/onnxruntime"
USE_GPU=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            USE_GPU=true
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --gpu              Install GPU version (requires CUDA)"
            echo "  --version VERSION  ONNX Runtime version (default: $VERSION)"
            echo "  --install-dir DIR  Installation directory (default: $INSTALL_DIR)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "ONNX Runtime Installer"
echo "========================================"
echo "Version: $VERSION"
echo "GPU: $USE_GPU"
echo "Install directory: $INSTALL_DIR"
echo ""

# 确定下载链接
if [ "$USE_GPU" = true ]; then
    FILENAME="onnxruntime-linux-x64-gpu-${VERSION}.tgz"
else
    FILENAME="onnxruntime-linux-x64-${VERSION}.tgz"
fi

URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${FILENAME}"

# 创建临时目录
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "Downloading from: $URL"
wget -q --show-progress "$URL" -O "$FILENAME"

echo "Extracting..."
tar -xzf "$FILENAME"

# 获取解压后的目录名
EXTRACTED_DIR=$(ls -d onnxruntime-linux-* | head -1)

echo "Installing to $INSTALL_DIR..."
sudo rm -rf "$INSTALL_DIR"
sudo mv "$EXTRACTED_DIR" "$INSTALL_DIR"

# 清理
cd /
rm -rf "$TEMP_DIR"

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "Add the following to your ~/.bashrc:"
echo ""
echo "  export ONNXRUNTIME_ROOT=$INSTALL_DIR"
echo "  export LD_LIBRARY_PATH=\$ONNXRUNTIME_ROOT/lib:\$LD_LIBRARY_PATH"
echo ""
echo "Then run: source ~/.bashrc"
echo ""
