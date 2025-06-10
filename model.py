import onnxruntime
import numpy as np
from PIL import Image
import io
import base64
import logging
from typing import Union, Tuple, List, Dict
from pathlib import Path
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing for ImageNet models."""
    
    def __init__(self):
        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.target_size = (224, 224)
        
    def load_image(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> Image.Image:
        """Load image from various input formats."""
        try:
            if isinstance(image_input, str):
                # File path
                if not Path(image_input).exists():
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                return Image.open(image_input)
                
            elif isinstance(image_input, bytes):
                # Raw bytes
                return Image.open(io.BytesIO(image_input))
                
            elif isinstance(image_input, np.ndarray):
                # NumPy array
                if image_input.dtype != np.uint8:
                    image_input = (image_input * 255).astype(np.uint8)
                return Image.fromarray(image_input)
                
            elif isinstance(image_input, Image.Image):
                # PIL Image
                return image_input
                
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
                
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    
    def load_image_from_base64(self, base64_string: str) -> Image.Image:
        """Load image from base64 string."""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',', 1)[1]
            
            image_bytes = base64.b64decode(base64_string)
            return Image.open(io.BytesIO(image_bytes))
            
        except Exception as e:
            logger.error(f"Error loading image from base64: {e}")
            raise
    
    def convert_to_rgb(self, image: Image.Image) -> Image.Image:
        """Convert image to RGB format if needed."""
        if image.mode != 'RGB':
            logger.debug(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        return image
    
    def resize_image(self, image: Image.Image, size: Tuple[int, int] = None) -> Image.Image:
        """Resize image using bilinear interpolation."""
        if size is None:
            size = self.target_size
            
        # Use LANCZOS for better quality (equivalent to bilinear for downscaling)
        return image.resize(size, Image.LANCZOS)
    
    def normalize_image(self, image_array: np.ndarray) -> np.ndarray:
        """Apply ImageNet normalization."""
        # Ensure image is in [0, 1] range
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
            
        # Apply normalization per channel
        normalized = (image_array - self.mean) / self.std
        return normalized.astype(np.float32)
    
    def preprocess(self, image_input: Union[str, bytes, np.ndarray, Image.Image], 
                   normalize: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        
        Args:
            image_input: Input image in various formats
            normalize: Whether to apply ImageNet normalization (set to False if using enhanced ONNX model)
            
        Returns:
            Preprocessed image array with shape (1, 3, 224, 224)
        """
        try:
            # Load and convert image
            image = self.load_image(image_input)
            image = self.convert_to_rgb(image)
            
            # Resize to target size
            image = self.resize_image(image)
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32)
            
            # Convert from HWC to CHW format
            image_array = np.transpose(image_array, (2, 0, 1))
            
            # Normalize to [0, 1] range
            if image_array.max() > 1.0:
                image_array = image_array / 255.0
            
            # Apply ImageNet normalization if requested
            if normalize:
                # Expand dimensions for broadcasting
                mean = self.mean.reshape(3, 1, 1)
                std = self.std.reshape(3, 1, 1)
                image_array = (image_array - mean) / std
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise


class ONNXModel:
    """Handles ONNX model loading and inference."""
    
    def __init__(self, model_path: str = "models/model.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        self._load_model()
        
    def _load_model(self):
        """Load ONNX model and initialize session."""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
            
            logger.info(f"Loading ONNX model from: {self.model_path}")
            
            # Configure ONNX Runtime session
            providers = ['CPUExecutionProvider']
            if onnxruntime.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = onnxruntime.InferenceSession(
                self.model_path,
                providers=providers
            )
            
            # Get input/output information
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_shape = self.session.get_outputs()[0].shape
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Input: {self.input_name} {self.input_shape}")
            logger.info(f"Output: {self.output_name} {self.output_shape}")
            logger.info(f"Providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            raise
    
    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference on input array."""
        try:
            if self.session is None:
                raise RuntimeError("Model not loaded")
            
            # Validate input shape
            expected_shape = tuple(dim if isinstance(dim, int) else 1 for dim in self.input_shape)
            if input_array.shape[1:] != expected_shape[1:]:
                raise ValueError(f"Input shape mismatch. Expected: {expected_shape}, Got: {input_array.shape}")
            
            # Run inference
            start_time = time.time()
            ort_inputs = {self.input_name: input_array}
            outputs = self.session.run([self.output_name], ort_inputs)
            inference_time = time.time() - start_time
            
            logger.debug(f"Inference completed in {inference_time:.3f}s")
            
            return outputs[0]
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_path': self.model_path,
            'input_name': self.input_name,
            'output_name': self.output_name,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'providers': self.session.get_providers() if self.session else None
        }


class ModelService:
    """Main service orchestrating preprocessing and inference."""
    
    def __init__(self, model_path: str = "models/model.onnx", use_enhanced_model: bool = True):
        self.preprocessor = ImagePreprocessor()
        self.model = ONNXModel(model_path)
        self.use_enhanced_model = use_enhanced_model  # Enhanced model includes preprocessing
        
        # Load ImageNet class labels
        self.class_labels = self._load_imagenet_labels()
        
    def _load_imagenet_labels(self) -> List[str]:
        """Load ImageNet class labels."""
        return [f"class_{i}" for i in range(1000)]
    
    def predict_image(self, image_input: Union[str, bytes, np.ndarray, Image.Image],
                     return_probabilities: bool = False,
                     top_k: int = 5) -> Dict:
        """
        Predict class for input image.
        
        Args:
            image_input: Input image in various formats
            return_probabilities: Whether to return probability scores
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        try:
            start_time = time.time()
            
            # Preprocess image
            # Don't normalize if using enhanced model (it handles normalization internally)
            preprocessed = self.preprocessor.preprocess(
                image_input, 
                normalize=not self.use_enhanced_model
            )
            
            preprocess_time = time.time() - start_time
            
            # Run inference
            inference_start = time.time()
            predictions = self.model.predict(preprocessed)
            inference_time = time.time() - inference_start
            
            # Process results
            probabilities = self._softmax(predictions[0])
            top_indices = np.argsort(probabilities)[::-1][:top_k]
            
            # Prepare results
            results = {
                'predicted_class_id': int(top_indices[0]),
                'predicted_class_name': self.class_labels[top_indices[0]] if top_indices[0] < len(self.class_labels) else f"class_{top_indices[0]}",
                'confidence': float(probabilities[top_indices[0]]),
                'top_predictions': [
                    {
                        'class_id': int(idx),
                        'class_name': self.class_labels[idx] if idx < len(self.class_labels) else f"class_{idx}",
                        'probability': float(probabilities[idx])
                    }
                    for idx in top_indices
                ],
                'processing_time': {
                    'preprocessing': preprocess_time,
                    'inference': inference_time,
                    'total': time.time() - start_time
                }
            }
            
            if return_probabilities:
                results['all_probabilities'] = probabilities.tolist()
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting image: {e}")
            raise
    
    def predict_batch(self, images: List[Union[str, bytes, np.ndarray, Image.Image]],
                     batch_size: int = 4) -> List[Dict]:
        """Predict classes for multiple images."""
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            for image in batch:
                try:
                    result = self.predict_image(image)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing image {i}: {e}")
                    results.append({'error': str(e)})
        
        return results
    
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def get_service_info(self) -> Dict:
        """Get service information."""
        return {
            'model_info': self.model.get_model_info(),
            'use_enhanced_model': self.use_enhanced_model,
            'num_classes': len(self.class_labels)
        }


# Convenience functions for easy usage
def load_model_service(model_path: str = "models/model.onnx") -> ModelService:
    """Load and return a ModelService instance."""
    return ModelService(model_path)


def predict_image_file(image_path: str, model_path: str = "models/model.onnx") -> Dict:
    """Quick prediction for image file."""
    service = ModelService(model_path)
    return service.predict_image(image_path)


def predict_from_base64(base64_string: str, model_path: str = "models/model.onnx") -> Dict:
    """Quick prediction from base64 string."""
    service = ModelService(model_path)
    preprocessor = ImagePreprocessor()
    image = preprocessor.load_image_from_base64(base64_string)
    return service.predict_image(image)


if __name__ == "__main__":
    # Example usage
    try:
        logger.info("Testing model service...")
        
        # Initialize service
        service = ModelService()
        info = service.get_service_info()
        logger.info(f"Service info: {info}")
        
        # Test with sample images if available
        test_images = ["n01440764_tench.jpeg", "n01667114_mud_turtle.jpeg"]
        
        for img_path in test_images:
            if Path(img_path).exists():
                logger.info(f"\nTesting with {img_path}...")
                result = service.predict_image(img_path)
                logger.info(f"Predicted class: {result['predicted_class_id']} ({result['predicted_class_name']})")
                logger.info(f"Confidence: {result['confidence']:.4f}")
                logger.info(f"Processing time: {result['processing_time']['total']:.3f}s")
            else:
                logger.warning(f"Test image {img_path} not found")
                
    except Exception as e:
        logger.error(f"Error in main: {e}")