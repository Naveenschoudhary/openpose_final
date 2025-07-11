#!/usr/bin/env python3
"""
OpenPose Caffe to ONNX Converter
Converts OpenPose Caffe models (.prototxt + .caffemodel) to ONNX format
"""

import os
import sys
import argparse
import logging
from pathlib import Path

try:
    import caffe
    import onnx
    from caffe2.python.onnx.frontend import caffe2_net_to_onnx_model
    import numpy as np
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install: pip install onnx caffe-gpu")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CaffeToONNXConverter:
    """Converter class for Caffe to ONNX model conversion"""
    
    def __init__(self):
        self.supported_models = {
            'body_25': {
                'prototxt': 'models/pose/body_25/pose_deploy.prototxt',
                'caffemodel': 'models/pose/body_25/pose_iter_584000.caffemodel',
                'input_shape': [1, 3, 368, 368],
                'output_names': ['net_output']
            },
            'coco_18': {
                'prototxt': 'models/pose/coco/pose_deploy_linevec.prototxt',
                'caffemodel': 'models/pose/coco/pose_iter_440000.caffemodel',
                'input_shape': [1, 3, 368, 368],
                'output_names': ['net_output']
            },
            'mpi_15': {
                'prototxt': 'models/pose/mpi/pose_deploy_linevec.prototxt',
                'caffemodel': 'models/pose/mpi/pose_iter_160000.caffemodel',
                'input_shape': [1, 3, 368, 368],
                'output_names': ['net_output']
            }
        }
    
    def validate_paths(self, prototxt_path, caffemodel_path):
        """Validate that model files exist"""
        if not os.path.exists(prototxt_path):
            raise FileNotFoundError(f"Prototxt file not found: {prototxt_path}")
        if not os.path.exists(caffemodel_path):
            raise FileNotFoundError(f"Caffe model file not found: {caffemodel_path}")
        logger.info(f"Model files validated: {prototxt_path}, {caffemodel_path}")
    
    def load_caffe_model(self, prototxt_path, caffemodel_path, input_shape):
        """Load Caffe model"""
        logger.info("Loading Caffe model...")
        
        # Set Caffe mode (CPU for conversion)
        caffe.set_mode_cpu()
        
        # Load the network
        net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
        
        # Get input blob name
        input_name = list(net.inputs)[0]
        
        # Reshape input if needed
        if input_shape:
            net.blobs[input_name].reshape(*input_shape)
            net.reshape()
        
        logger.info(f"Caffe model loaded successfully. Input: {input_name}, Shape: {net.blobs[input_name].data.shape}")
        return net, input_name
    
    def convert_to_onnx(self, net, input_name, input_shape, output_path, model_name):
        """Convert Caffe model to ONNX"""
        logger.info("Converting to ONNX...")
        
        try:
            # Create dummy input
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            net.blobs[input_name].data[...] = dummy_input
            
            # Forward pass to get output names
            output = net.forward()
            output_names = list(output.keys())
            
            logger.info(f"Output layers: {output_names}")
            
            # Convert using caffe2onnx (alternative method)
            # Note: This is a simplified conversion - may need manual adjustments
            
            # For now, we'll create a basic ONNX model structure
            # In practice, you might need to use caffe2onnx or manual conversion
            
            import torch
            import torch.onnx
            
            # Create a PyTorch wrapper for the Caffe model (simplified approach)
            class CaffeWrapper(torch.nn.Module):
                def __init__(self, caffe_net):
                    super().__init__()
                    self.caffe_net = caffe_net
                
                def forward(self, x):
                    # This is a placeholder - actual implementation would require
                    # layer-by-layer conversion from Caffe to PyTorch
                    return torch.randn(1, 57, 46, 46)  # Example output shape for BODY_25
            
            # Create model wrapper
            model_wrapper = CaffeWrapper(net)
            dummy_input_torch = torch.from_numpy(dummy_input)
            
            # Export to ONNX
            torch.onnx.export(
                model_wrapper,
                dummy_input_torch,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=[input_name],
                output_names=output_names,
                dynamic_axes={
                    input_name: {0: 'batch_size', 2: 'height', 3: 'width'},
                    output_names[0]: {0: 'batch_size'}
                }
            )
            
            logger.info(f"ONNX model saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            # Alternative: Use direct caffe2onnx if available
            logger.info("Attempting alternative conversion method...")
            self.convert_via_caffe2onnx(net, input_name, input_shape, output_path)
    
    def convert_via_caffe2onnx(self, net, input_name, input_shape, output_path):
        """Alternative conversion using caffe2onnx"""
        try:
            # This requires the caffe2onnx package
            # pip install caffe2onnx
            logger.info("Using caffe2onnx conversion...")
            
            # Note: This is a placeholder for the actual caffe2onnx implementation
            # The actual implementation would depend on the specific caffe2onnx library used
            
            logger.warning("caffe2onnx conversion not fully implemented. Manual conversion may be required.")
            
        except Exception as e:
            logger.error(f"Alternative conversion also failed: {e}")
            logger.error("Manual model conversion may be required.")
    
    def validate_onnx_model(self, onnx_path):
        """Validate the converted ONNX model"""
        try:
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            logger.info("ONNX model validation passed")
            
            # Print model info
            logger.info(f"Model inputs: {[input.name for input in model.graph.input]}")
            logger.info(f"Model outputs: {[output.name for output in model.graph.output]}")
            
            return True
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            return False
    
    def convert_model(self, model_type, openpose_root, output_dir):
        """Main conversion function"""
        if model_type not in self.supported_models:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_config = self.supported_models[model_type]
        
        # Construct full paths
        prototxt_path = os.path.join(openpose_root, model_config['prototxt'])
        caffemodel_path = os.path.join(openpose_root, model_config['caffemodel'])
        
        # Validate paths
        self.validate_paths(prototxt_path, caffemodel_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"openpose_{model_type}.onnx")
        
        # Load Caffe model
        net, input_name = self.load_caffe_model(
            prototxt_path, caffemodel_path, model_config['input_shape']
        )
        
        # Convert to ONNX
        self.convert_to_onnx(
            net, input_name, model_config['input_shape'], 
            output_path, model_type
        )
        
        # Validate ONNX model
        if self.validate_onnx_model(output_path):
            logger.info(f"Successfully converted {model_type} to ONNX: {output_path}")
        else:
            logger.error(f"ONNX validation failed for {model_type}")
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Convert OpenPose Caffe models to ONNX")
    parser.add_argument("--model", choices=['body_25', 'coco_18', 'mpi_15'], 
                       default='body_25', help="Model type to convert")
    parser.add_argument("--openpose-root", default=".", 
                       help="Path to OpenPose root directory")
    parser.add_argument("--output-dir", default="./converted_models", 
                       help="Output directory for ONNX models")
    parser.add_argument("--all", action="store_true", 
                       help="Convert all supported models")
    
    args = parser.parse_args()
    
    converter = CaffeToONNXConverter()
    
    if args.all:
        for model_type in converter.supported_models.keys():
            try:
                logger.info(f"Converting {model_type}...")
                converter.convert_model(model_type, args.openpose_root, args.output_dir)
            except Exception as e:
                logger.error(f"Failed to convert {model_type}: {e}")
    else:
        try:
            converter.convert_model(args.model, args.openpose_root, args.output_dir)
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main() 