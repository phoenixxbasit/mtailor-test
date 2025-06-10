import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import sys
import os

from pytorch_model import Classifier, BasicBlock

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ONNXConverter:
    """Handles PyTorch to ONNX conversion with preprocessing integration."""
    
    def __init__(self, model_weights_path="models/pytorch_model_weights.pth", output_path="models/model.onnx"):
        self.model_weights_path = model_weights_path
        self.output_path = output_path
        self.model = None
        
    def load_pytorch_model(self):
        try:
            self.model = Classifier(BasicBlock, [2, 2, 2, 2])
            
            # Check if weights file exists
            if not os.path.exists(self.model_weights_path):
                logger.error(f"Model weights not found at {self.model_weights_path}")
                sys.exit(1)
                
            self.model.load_state_dict(torch.load(self.model_weights_path, map_location='cpu'))
            self.model.eval()
            logger.info("PyTorch model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            return False
    
    def create_enhanced_model_with_preprocessing(self):
        """Create model with integrated preprocessing steps."""
        
        class EnhancedModel(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                
                self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
                
            def forward(self, x):
                x = (x - self.mean) / self.std
                return self.base_model(x)
        
        enhanced_model = EnhancedModel(self.model)
        enhanced_model.eval()
        return enhanced_model
    
    def convert_to_onnx(self):
        """Convert PyTorch model to ONNX format."""
        try:
            logger.info("Converting PyTorch model to ONNX...")
            
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            model_to_convert = self.create_enhanced_model_with_preprocessing()
            
            # Create dummy input (batch_size=1, channels=3, height=224, width=224)
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # Define input and output names
            input_names = ['input_image']
            output_names = ['class_probabilities']
            
            # Dynamic axes for batch size
            dynamic_axes = {
                'input_image': {0: 'batch_size'},
                'class_probabilities': {0: 'batch_size'}
            }
            
            # Export to ONNX
            torch.onnx.export(
                model_to_convert,
                dummy_input,
                self.output_path,
                export_params=True,
                opset_version=11,  # Use opset 11 for better compatibility
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            logger.info(f"ONNX model saved to: {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error during ONNX conversion: {e}")
            return False
    
    def validate_onnx_model(self):
        """Validate the converted ONNX model."""
        try:
            logger.info("Validating ONNX model...")
            
            # Check ONNX model structure
            onnx_model = onnx.load(self.output_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model structure is valid")
            
            # Test with ONNX Runtime
            ort_session = onnxruntime.InferenceSession(self.output_path)
            
            # Get input/output info
            input_info = ort_session.get_inputs()[0]
            output_info = ort_session.get_outputs()[0]
            
            logger.info(f"Input name: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
            logger.info(f"Output name: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")
            
            # Test inference
            dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            ort_inputs = {input_info.name: dummy_input}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            logger.info(f"ONNX model output shape: {ort_outputs[0].shape}")
            logger.info("ONNX model validation successful")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating ONNX model: {e}")
            return False
    
    def compare_outputs(self):
        """Compare PyTorch and ONNX model outputs."""
        try:
            logger.info("Comparing PyTorch and ONNX outputs...")
            
            # Create test input
            test_input = torch.randn(1, 3, 224, 224)
            
            # PyTorch inference
            with torch.no_grad():
                pytorch_output = self.model(test_input)
            
            # ONNX inference
            ort_session = onnxruntime.InferenceSession(self.output_path)
            ort_inputs = {'input_image': test_input.numpy()}
            onnx_output = ort_session.run(None, ort_inputs)[0]
            
            # Compare outputs (note: enhanced model includes preprocessing)
            # For fair comparison, we need to apply preprocessing to PyTorch model
            normalized_input = test_input.clone()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            normalized_input = (normalized_input - mean) / std
            
            with torch.no_grad():
                pytorch_normalized_output = self.model(normalized_input)
            
            # Calculate difference
            diff = np.abs(pytorch_normalized_output.numpy() - onnx_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            logger.info(f"Max difference: {max_diff:.6f}")
            logger.info(f"Mean difference: {mean_diff:.6f}")
            
            if max_diff < 1e-5:
                logger.info("✓ Models produce nearly identical outputs")
            else:
                logger.warning("⚠ Significant difference detected between models")
            
            return max_diff < 1e-3  # Allow small numerical differences
            
        except Exception as e:
            logger.error(f"Error comparing outputs: {e}")
            return False


def main():
    """Main conversion function."""
    logger.info("Starting PyTorch to ONNX conversion...")
    
    converter = ONNXConverter()
    
    # Load PyTorch model
    if not converter.load_pytorch_model():
        logger.error("Failed to load PyTorch model")
        sys.exit(1)
    
    # Convert to ONNX
    if not converter.convert_to_onnx():
        logger.error("Failed to convert to ONNX")
        sys.exit(1)
    
    # Validate ONNX model
    if not converter.validate_onnx_model():
        logger.error("ONNX model validation failed")
        sys.exit(1)
    
    # Compare outputs
    if not converter.compare_outputs():
        logger.warning("Output comparison showed significant differences")
    
    logger.info("✓ Conversion completed successfully!")
    logger.info(f"ONNX model saved at: {converter.output_path}")
    
    # Print model info
    try:
        onnx_model = onnx.load(converter.output_path)
        logger.info(f"Model IR version: {onnx_model.ir_version}")
        logger.info(f"Model opset version: {onnx_model.opset_import[0].version}")
        logger.info(f"Model size: {os.path.getsize(converter.output_path) / (1024*1024):.2f} MB")
    except Exception as e:
        logger.warning(f"Could not get model info: {e}")


if __name__ == "__main__":
    main()