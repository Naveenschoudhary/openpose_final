#!/usr/bin/env python3
"""
OpenPose ONNX to TensorRT Converter
Converts ONNX models to optimized TensorRT engines
"""

import os
import sys
import argparse
import logging
from pathlib import Path

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    import numpy as np
    import onnx
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install: pip install tensorrt pycuda onnx")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXToTensorRTConverter:
    """Converter class for ONNX to TensorRT engine conversion"""
    
    def __init__(self):
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.optimization_profiles = {
            'body_25': {
                'min_shape': [1, 3, 184, 184],
                'opt_shape': [1, 3, 368, 368], 
                'max_shape': [1, 3, 736, 736]
            },
            'coco_18': {
                'min_shape': [1, 3, 184, 184],
                'opt_shape': [1, 3, 368, 368],
                'max_shape': [1, 3, 736, 736]
            },
            'mpi_15': {
                'min_shape': [1, 3, 184, 184],
                'opt_shape': [1, 3, 368, 368],
                'max_shape': [1, 3, 736, 736]
            }
        }
    
    def build_engine_from_onnx(self, onnx_path, engine_path, precision='fp16', 
                              max_workspace_size=1<<30, max_batch_size=1,
                              model_type='body_25'):
        """Build TensorRT engine from ONNX model"""
        
        logger.info(f"Building TensorRT engine from {onnx_path}")
        logger.info(f"Precision: {precision}, Max workspace: {max_workspace_size} bytes")
        
        # Create builder and network
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.trt_logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    logger.error(f"Error {error}: {parser.get_error(error)}")
                return False
        
        logger.info("ONNX model parsed successfully")
        
        # Create builder config
        config = builder.create_builder_config()
        config.max_workspace_size = max_workspace_size
        
        # Set precision
        if precision == 'fp16':
            if builder.platform_has_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 precision enabled")
            else:
                logger.warning("FP16 not supported on this platform, using FP32")
        elif precision == 'int8':
            if builder.platform_has_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                # Note: INT8 calibration would be needed here
                logger.info("INT8 precision enabled (calibration required)")
            else:
                logger.warning("INT8 not supported on this platform, using FP32")
        
        # Setup optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()
        
        if model_type in self.optimization_profiles:
            shapes = self.optimization_profiles[model_type]
            
            # Assuming input name is 'image' or first input
            input_name = network.get_input(0).name
            logger.info(f"Setting optimization profile for input: {input_name}")
            
            profile.set_shape(
                input_name,
                shapes['min_shape'],
                shapes['opt_shape'], 
                shapes['max_shape']
            )
            
            config.add_optimization_profile(profile)
            logger.info(f"Added optimization profile: min={shapes['min_shape']}, "
                       f"opt={shapes['opt_shape']}, max={shapes['max_shape']}")
        
        # Build engine
        logger.info("Building TensorRT engine... This may take several minutes.")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            logger.error("Failed to build TensorRT engine")
            return False
        
        # Save engine
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        logger.info(f"TensorRT engine saved to: {engine_path}")
        
        # Validate engine
        return self.validate_engine(engine_path)
    
    def validate_engine(self, engine_path):
        """Validate the built TensorRT engine"""
        try:
            runtime = trt.Runtime(self.trt_logger)
            
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            engine = runtime.deserialize_cuda_engine(engine_data)
            
            if engine is None:
                logger.error("Failed to deserialize engine")
                return False
            
            # Get engine info
            logger.info(f"Engine validation successful")
            logger.info(f"Max batch size: {engine.max_batch_size}")
            logger.info(f"Number of bindings: {engine.num_bindings}")
            
            # Print input/output info
            for i in range(engine.num_bindings):
                binding_name = engine.get_binding_name(i)
                binding_shape = engine.get_binding_shape(i)
                binding_dtype = engine.get_binding_dtype(i)
                is_input = engine.binding_is_input(i)
                
                logger.info(f"Binding {i}: {binding_name}, Shape: {binding_shape}, "
                           f"Type: {binding_dtype}, Input: {is_input}")
            
            return True
            
        except Exception as e:
            logger.error(f"Engine validation failed: {e}")
            return False
    
    def benchmark_engine(self, engine_path, num_iterations=100):
        """Benchmark the TensorRT engine performance"""
        logger.info(f"Benchmarking engine: {engine_path}")
        
        try:
            import time
            
            runtime = trt.Runtime(self.trt_logger)
            
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            engine = runtime.deserialize_cuda_engine(engine_data)
            context = engine.create_execution_context()
            
            # Allocate GPU memory for inputs/outputs
            bindings = []
            inputs = []
            outputs = []
            
            for i in range(engine.num_bindings):
                binding_name = engine.get_binding_name(i)
                binding_shape = engine.get_binding_shape(i)
                binding_dtype = engine.get_binding_dtype(i)
                
                # Calculate size
                size = trt.volume(binding_shape) * engine.max_batch_size
                dtype_size = trt.volume([1])  # Simplified
                
                # Allocate GPU memory
                gpu_mem = cuda.mem_alloc(size * 4)  # Assuming float32
                bindings.append(int(gpu_mem))
                
                if engine.binding_is_input(i):
                    inputs.append({'name': binding_name, 'gpu_mem': gpu_mem, 'shape': binding_shape})
                else:
                    outputs.append({'name': binding_name, 'gpu_mem': gpu_mem, 'shape': binding_shape})
            
            # Warm up
            for _ in range(10):
                context.execute_v2(bindings)
            
            # Benchmark
            start_time = time.time()
            
            for _ in range(num_iterations):
                context.execute_v2(bindings)
            
            cuda.Context.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_iterations * 1000  # ms
            fps = 1000 / avg_time
            
            logger.info(f"Average inference time: {avg_time:.2f} ms")
            logger.info(f"Throughput: {fps:.2f} FPS")
            
            # Cleanup
            for binding in bindings:
                cuda.mem_free(binding)
            
            return avg_time, fps
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return None, None
    
    def convert_onnx_to_tensorrt(self, onnx_path, output_dir, model_name, 
                                precision='fp16', benchmark=True):
        """Main conversion function"""
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Engine path
        engine_filename = f"openpose_{model_name}_{precision}.trt"
        engine_path = os.path.join(output_dir, engine_filename)
        
        # Build engine
        success = self.build_engine_from_onnx(
            onnx_path, engine_path, precision, model_type=model_name
        )
        
        if not success:
            logger.error(f"Failed to convert {model_name}")
            return None
        
        # Benchmark if requested
        if benchmark:
            avg_time, fps = self.benchmark_engine(engine_path)
            if avg_time is not None:
                logger.info(f"Performance: {avg_time:.2f}ms ({fps:.2f} FPS)")
        
        return engine_path

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX models to TensorRT engines")
    parser.add_argument("--onnx-path", required=True, help="Path to ONNX model")
    parser.add_argument("--output-dir", default="./tensorrt_engines", 
                       help="Output directory for TensorRT engines")
    parser.add_argument("--model-name", default="body_25", 
                       choices=['body_25', 'coco_18', 'mpi_15'],
                       help="Model name for optimization profiles")
    parser.add_argument("--precision", choices=['fp32', 'fp16', 'int8'], 
                       default='fp16', help="Precision mode")
    parser.add_argument("--workspace-size", type=int, default=1024,
                       help="Max workspace size in MB")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Benchmark the engine after building")
    parser.add_argument("--batch-all", action="store_true",
                       help="Convert all ONNX models in directory")
    
    args = parser.parse_args()
    
    converter = ONNXToTensorRTConverter()
    
    workspace_size = args.workspace_size * 1024 * 1024  # Convert MB to bytes
    
    if args.batch_all:
        # Convert all ONNX files in the directory
        onnx_dir = os.path.dirname(args.onnx_path) if os.path.isfile(args.onnx_path) else args.onnx_path
        
        for filename in os.listdir(onnx_dir):
            if filename.endswith('.onnx'):
                onnx_path = os.path.join(onnx_dir, filename)
                model_name = filename.replace('openpose_', '').replace('.onnx', '')
                
                logger.info(f"Converting {filename}...")
                
                try:
                    engine_path = converter.convert_onnx_to_tensorrt(
                        onnx_path, args.output_dir, model_name, 
                        args.precision, args.benchmark
                    )
                    
                    if engine_path:
                        logger.info(f"Successfully converted {filename} -> {engine_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to convert {filename}: {e}")
    else:
        try:
            engine_path = converter.convert_onnx_to_tensorrt(
                args.onnx_path, args.output_dir, args.model_name,
                args.precision, args.benchmark
            )
            
            if engine_path:
                logger.info(f"Conversion successful: {engine_path}")
            else:
                logger.error("Conversion failed")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main() 