#!/bin/bash

# OpenPose TensorRT Conversion Demo
# This script demonstrates the conversion pipeline with different configurations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../../tensorrt_models_demo"

echo "🚀 OpenPose TensorRT Conversion Demo"
echo "===================================="
echo

# Check if we're on a system that can run the conversion
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "ℹ️  Note: This demo is designed for Linux systems with NVIDIA GPUs."
    echo "   On macOS, the scripts can be reviewed but not executed."
    echo "   Please run this on a Linux system with TensorRT for actual conversion."
    echo
fi

echo "📁 Demo will create TensorRT engines in: $OUTPUT_DIR"
echo

# Demo 1: Convert single model with FP16 (most common use case)
echo "🎯 Demo 1: Convert BODY_25 model with FP16 precision"
echo "Command: ./convert_all_models.sh -m body_25 -p fp16 -o $OUTPUT_DIR"
echo "Expected output: openpose_body_25_fp16.trt (~200MB)"
echo "Performance: ~2-3x faster than Caffe"
echo

# Demo 2: Convert all models with INT8 for maximum performance
echo "🎯 Demo 2: Convert all models with INT8 precision"
echo "Command: ./convert_all_models.sh -p int8 -w 2048 -o $OUTPUT_DIR"
echo "Expected output: 3 TensorRT engines (~100MB each)"
echo "Performance: ~3-4x faster than Caffe"
echo

# Demo 3: Debug conversion with verbose output
echo "🎯 Demo 3: Debug conversion with verbose output and temp files"
echo "Command: ./convert_all_models.sh -m coco_18 -v --keep-temp -o $OUTPUT_DIR"
echo "Expected output: ONNX intermediate files + TensorRT engine"
echo "Use case: Debugging conversion issues"
echo

# Demo 4: Production deployment scenario
echo "🎯 Demo 4: Production deployment (multiple precisions)"
echo "Commands:"
echo "  ./convert_all_models.sh -p fp16 -o $OUTPUT_DIR/production"
echo "  ./convert_all_models.sh -p int8 -o $OUTPUT_DIR/production"
echo "Expected output: Both FP16 and INT8 engines for runtime selection"
echo "Use case: A/B testing different precision modes"
echo

echo "💡 Tips:"
echo "   - Use FP16 for best balance of speed and accuracy"
echo "   - Use INT8 for maximum performance (requires calibration)"
echo "   - Use FP32 only for accuracy validation"
echo "   - Larger workspace sizes can improve optimization"
echo

echo "🔧 Integration Example:"
cat << 'EOF'

// C++ Integration Example
#include <openpose/pose/poseExtractorTensorRT.hpp>

// Create TensorRT pose extractor
const auto poseModel = op::PoseModel::BODY_25;
const auto engineFolder = "models/tensorrt/";
const auto gpuId = 0;

auto poseExtractorPtr = std::make_shared<op::PoseExtractorTensorRT>(
    poseModel, engineFolder, gpuId,
    std::vector<op::HeatMapType>{}, // heatMapTypes
    op::ScaleMode::ZeroToOneFixedAspect, // heatMapScaleMode
    false, // addPartCandidates
    false, // maximizePositives
    "", // enginePath (auto-detect)
    "fp16", // precision
    true, // enableNet
    false // enableProfiling
);

// Use in OpenPose wrapper
auto poseExtractor = std::make_shared<op::PoseExtractor>(poseExtractorPtr);

EOF

echo
echo "📊 Expected Performance Improvements:"
echo
printf "%-12s %-15s %-15s %-15s %-10s\n" "Model" "Caffe (ms)" "TRT FP16 (ms)" "TRT INT8 (ms)" "Speedup"
printf "%-12s %-15s %-15s %-15s %-10s\n" "------------" "----------" "----------" "----------" "--------"
printf "%-12s %-15s %-15s %-15s %-10s\n" "BODY_25" "~35" "~12" "~8" "3-4x"
printf "%-12s %-15s %-15s %-15s %-10s\n" "COCO_18" "~30" "~10" "~7" "3-4x"
printf "%-12s %-15s %-15s %-15s %-10s\n" "MPI_15" "~25" "~8" "~6" "3-4x"

echo
echo "✅ Demo completed. To run actual conversion:"
echo "   1. Set up TensorRT environment (see README.md)"
echo "   2. Run: ./convert_all_models.sh"
echo "   3. Integrate TensorRT engines into your OpenPose application"
echo 