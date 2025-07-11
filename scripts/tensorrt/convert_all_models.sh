#!/bin/bash

# OpenPose TensorRT Conversion Pipeline
# This script converts all OpenPose Caffe models to TensorRT engines
# 
# Usage: ./convert_all_models.sh [options]
# 
# Prerequisites:
# - NVIDIA GPU with CUDA support
# - TensorRT installed (version 8.5+)
# - Python environment with required packages (see requirements.txt)

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPOSE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$OPENPOSE_ROOT/models"
OUTPUT_DIR="$OPENPOSE_ROOT/tensorrt_models"
TEMP_DIR="$OUTPUT_DIR/temp"

# Default settings
PRECISION="fp16"
WORKSPACE_SIZE=1024  # MB
BENCHMARK=true
CLEANUP_TEMP=true
VERBOSE=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print usage information
print_usage() {
    cat << EOF
OpenPose TensorRT Conversion Pipeline

Usage: $0 [OPTIONS]

OPTIONS:
    -p, --precision PRECISION    Precision mode: fp32, fp16, int8 (default: fp16)
    -w, --workspace SIZE         Workspace size in MB (default: 1024)
    -o, --output-dir DIR         Output directory for TensorRT engines (default: ./tensorrt_models)
    -m, --model MODEL           Convert specific model only: body_25, coco_18, mpi_15
    --no-benchmark              Skip benchmarking after conversion
    --keep-temp                 Keep temporary ONNX files
    -v, --verbose               Enable verbose output
    -h, --help                  Show this help message

EXAMPLES:
    $0                          # Convert all models with default settings
    $0 -p int8 -w 2048         # Use INT8 precision with 2GB workspace
    $0 -m body_25              # Convert only BODY_25 model
    $0 --no-benchmark          # Skip performance benchmarking

REQUIREMENTS:
    - NVIDIA GPU with CUDA
    - TensorRT 8.5+
    - Python packages: tensorrt, onnx, pycuda, torch
    - Caffe models in models/ directory

EOF
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check CUDA
    if ! command -v nvcc &> /dev/null; then
        log_error "CUDA not found. Please install CUDA toolkit."
        exit 1
    fi
    
    # Check GPU
    if ! nvidia-smi &> /dev/null; then
        log_error "NVIDIA GPU not detected or driver not installed."
        exit 1
    fi
    
    # Check Python packages
    local required_packages=("tensorrt" "onnx" "torch" "numpy")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            log_error "Python package '$package' not found. Install with: pip install $package"
            exit 1
        fi
    done
    
    log_success "System requirements check passed"
}

# Check if Caffe models exist
check_caffe_models() {
    log_info "Checking Caffe models..."
    
    local models=("body_25" "coco_18" "mpi_15")
    local missing_models=()
    
    for model in "${models[@]}"; do
        case $model in
            "body_25")
                prototxt="$MODELS_DIR/pose/body_25/pose_deploy.prototxt"
                caffemodel="$MODELS_DIR/pose/body_25/pose_iter_584000.caffemodel"
                ;;
            "coco_18")
                prototxt="$MODELS_DIR/pose/coco/pose_deploy_linevec.prototxt"
                caffemodel="$MODELS_DIR/pose/coco/pose_iter_440000.caffemodel"
                ;;
            "mpi_15")
                prototxt="$MODELS_DIR/pose/mpi/pose_deploy_linevec.prototxt"
                caffemodel="$MODELS_DIR/pose/mpi/pose_iter_160000.caffemodel"
                ;;
        esac
        
        if [[ ! -f "$prototxt" ]] || [[ ! -f "$caffemodel" ]]; then
            missing_models+=("$model")
        fi
    done
    
    if [[ ${#missing_models[@]} -gt 0 ]]; then
        log_warning "Missing Caffe models: ${missing_models[*]}"
        log_warning "Run './models/getModels.sh' to download missing models"
        return 1
    fi
    
    log_success "All Caffe models found"
    return 0
}

# Convert single model through the pipeline
convert_model() {
    local model_name="$1"
    local precision="$2"
    local workspace_size="$3"
    local benchmark="$4"
    
    log_info "Converting $model_name model..."
    
    # Step 1: Caffe to ONNX
    log_info "Step 1/2: Converting Caffe to ONNX..."
    local onnx_path="$TEMP_DIR/openpose_${model_name}.onnx"
    
    if ! python3 "$SCRIPT_DIR/caffe_to_onnx.py" \
        --model "$model_name" \
        --openpose-root "$OPENPOSE_ROOT" \
        --output-dir "$TEMP_DIR"; then
        log_error "Failed to convert $model_name from Caffe to ONNX"
        return 1
    fi
    
    # Step 2: ONNX to TensorRT
    log_info "Step 2/2: Converting ONNX to TensorRT..."
    local engine_path="$OUTPUT_DIR/openpose_${model_name}_${precision}.trt"
    
    local benchmark_flag=""
    if [[ "$benchmark" == true ]]; then
        benchmark_flag="--benchmark"
    fi
    
    if ! python3 "$SCRIPT_DIR/onnx_to_tensorrt.py" \
        --onnx-path "$onnx_path" \
        --output-dir "$OUTPUT_DIR" \
        --model-name "$model_name" \
        --precision "$precision" \
        --workspace-size "$workspace_size" \
        $benchmark_flag; then
        log_error "Failed to convert $model_name from ONNX to TensorRT"
        return 1
    fi
    
    log_success "Successfully converted $model_name to TensorRT: $engine_path"
    return 0
}

# Clean up temporary files
cleanup() {
    if [[ "$CLEANUP_TEMP" == true ]] && [[ -d "$TEMP_DIR" ]]; then
        log_info "Cleaning up temporary files..."
        rm -rf "$TEMP_DIR"
        log_success "Cleanup completed"
    fi
}

# Main conversion function
main() {
    local target_models=()
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--precision)
                PRECISION="$2"
                shift 2
                ;;
            -w|--workspace)
                WORKSPACE_SIZE="$2"
                shift 2
                ;;
            -o|--output-dir)
                OUTPUT_DIR="$2"
                TEMP_DIR="$OUTPUT_DIR/temp"
                shift 2
                ;;
            -m|--model)
                target_models+=("$2")
                shift 2
                ;;
            --no-benchmark)
                BENCHMARK=false
                shift
                ;;
            --keep-temp)
                CLEANUP_TEMP=false
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Validate precision
    if [[ ! "$PRECISION" =~ ^(fp32|fp16|int8)$ ]]; then
        log_error "Invalid precision: $PRECISION. Use fp32, fp16, or int8"
        exit 1
    fi
    
    # Set default models if none specified
    if [[ ${#target_models[@]} -eq 0 ]]; then
        target_models=("body_25" "coco_18" "mpi_15")
    fi
    
    # Print configuration
    log_info "OpenPose TensorRT Conversion Pipeline"
    log_info "===================================="
    log_info "Precision: $PRECISION"
    log_info "Workspace Size: ${WORKSPACE_SIZE}MB"
    log_info "Output Directory: $OUTPUT_DIR"
    log_info "Target Models: ${target_models[*]}"
    log_info "Benchmark: $BENCHMARK"
    echo
    
    # Check requirements
    check_requirements
    
    # Check Caffe models
    if ! check_caffe_models; then
        log_error "Please ensure all required Caffe models are available"
        exit 1
    fi
    
    # Create output directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$TEMP_DIR"
    
    # Setup cleanup trap
    trap cleanup EXIT
    
    # Convert models
    local success_count=0
    local total_count=${#target_models[@]}
    
    for model in "${target_models[@]}"; do
        echo
        log_info "Processing model $((success_count + 1))/$total_count: $model"
        log_info "================================================"
        
        if convert_model "$model" "$PRECISION" "$WORKSPACE_SIZE" "$BENCHMARK"; then
            ((success_count++))
        else
            log_error "Failed to convert $model"
        fi
    done
    
    # Print summary
    echo
    log_info "Conversion Summary"
    log_info "=================="
    log_info "Successfully converted: $success_count/$total_count models"
    
    if [[ $success_count -eq $total_count ]]; then
        log_success "All models converted successfully!"
        log_info "TensorRT engines are available in: $OUTPUT_DIR"
    else
        log_warning "Some models failed to convert. Check the logs above for details."
        exit 1
    fi
}

# Run main function with all arguments
main "$@" 