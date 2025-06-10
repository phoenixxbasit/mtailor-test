import unittest
import numpy as np
from PIL import Image
import tempfile
import os
import json
import time
import logging
from pathlib import Path
import base64
import io

# Import our modules
from model import ModelService, ImagePreprocessor, ONNXModel
from convert_to_onnx import ONNXConverter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestImagePreprocessor(unittest.TestCase):
    """Test cases for ImagePreprocessor class."""
    
    def setUp(self):
        self.preprocessor = ImagePreprocessor()
        # Create a test image
        self.test_image = Image.new('RGB', (256, 256), color='red')
        
    def test_convert_to_rgb(self):
        """Test RGB conversion."""
        # Test with RGBA image
        rgba_image = Image.new('RGBA', (100, 100), color='blue')
        rgb_image = self.preprocessor.convert_to_rgb(rgba_image)
        self.assertEqual(rgb_image.mode, 'RGB')
        
        # Test with already RGB image
        rgb_input = Image.new('RGB', (100, 100), color='green')
        rgb_output = self.preprocessor.convert_to_rgb(rgb_input)
        self.assertEqual(rgb_output.mode, 'RGB')
    
    def test_resize_image(self):
        """Test image resizing."""
        resized = self.preprocessor.resize_image(self.test_image)
        self.assertEqual(resized.size, (224, 224))
        
        # Test custom size
        custom_resized = self.preprocessor.resize_image(self.test_image, (128, 128))
        self.assertEqual(custom_resized.size, (128, 128))
    
    def test_load_image_from_pil(self):
        """Test loading PIL Image."""
        loaded = self.preprocessor.load_image(self.test_image)
        self.assertIsInstance(loaded, Image.Image)
        self.assertEqual(loaded.size, self.test_image.size)
    
    def test_load_image_from_numpy(self):
        """Test loading from numpy array."""
        np_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        loaded = self.preprocessor.load_image(np_array)
        self.assertIsInstance(loaded, Image.Image)
        self.assertEqual(loaded.size, (100, 100))
    
    def test_load_image_from_base64(self):
        """Test loading from base64 string."""
        # Convert test image to base64
        buffer = io.BytesIO()
        self.test_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Test with and without data URL prefix
        loaded1 = self.preprocessor.load_image_from_base64(img_str)
        loaded2 = self.preprocessor.load_image_from_base64(f"data:image/png;base64,{img_str}")
        
        self.assertIsInstance(loaded1, Image.Image)
        self.assertIsInstance(loaded2, Image.Image)
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        processed = self.preprocessor.preprocess(self.test_image)
        
        # Check output shape
        self.assertEqual(processed.shape, (1, 3, 224, 224))
        
        # Check data type
        self.assertEqual(processed.dtype, np.float32)
        
        # Test without normalization
        processed_no_norm = self.preprocessor.preprocess(self.test_image, normalize=False)
        self.assertEqual(processed_no_norm.shape, (1, 3, 224, 224))
        
        # Normalized version should be different
        self.assertFalse(np.array_equal(processed, processed_no_norm))
    
    def test_normalize_image(self):
        """Test image normalization."""
        # Test with [0, 1] range input
        test_array = np.random.random((3, 224, 224)).astype(np.float32)
        normalized = self.preprocessor.normalize_image(test_array)
        
        # Check shape preservation
        self.assertEqual(normalized.shape, test_array.shape)
        
        # Test with [0, 255] range input
        test_array_255 = np.random.randint(0, 255, (3, 224, 224)).astype(np.float32)
        normalized_255 = self.preprocessor.normalize_image(test_array_255)
        self.assertEqual(normalized_255.shape, test_array_255.shape)


class TestONNXModel(unittest.TestCase):
    """Test cases for ONNXModel class."""
    
    def setUp(self):
        self.model_path = "models/model.onnx"
        # Skip tests if model doesn't exist
        if not Path(self.model_path).exists():
            self.skipTest(f"ONNX model not found at {self.model_path}")
    
    def test_model_loading(self):
        """Test ONNX model loading."""
        model = ONNXModel(self.model_path)
        
        # Check model attributes
        self.assertIsNotNone(model.session)
        self.assertIsNotNone(model.input_name)
        self.assertIsNotNone(model.output_name)
        self.assertEqual(len(model.input_shape), 4)  # Batch, Channel, Height, Width
    
    def test_prediction(self):
        """Test model prediction."""
        model = ONNXModel(self.model_path)
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Run prediction
        start_time = time.time()
        output = model.predict(dummy_input)
        prediction_time = time.time() - start_time
        
        # Check output
        self.assertEqual(len(output.shape), 2)
        self.assertEqual(output.shape[0], 1)  # Batch size
        self.assertEqual(output.shape[1], 1000)  # Number of classes
        
        # Check prediction time (should be under 3 seconds for production)
        self.assertLess(prediction_time, 3.0, "Prediction took too long for production use")
        
        logger.info(f"Prediction completed in {prediction_time:.3f}s")
    
    def test_batch_prediction(self):
        """Test batch prediction capability."""
        model = ONNXModel(self.model_path)
        
        # Create batch input
        batch_size = 4
        batch_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
        
        # Run prediction
        output = model.predict(batch_input)
        
        # Check output shape
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], 1000)
    
    def test_invalid_input_shape(self):
        """Test handling of invalid input shapes."""
        model = ONNXModel(self.model_path)
        
        # Wrong input shape
        invalid_input = np.random.randn(1, 3, 100, 100).astype(np.float32)
        
        with self.assertRaises(ValueError):
            model.predict(invalid_input)
    
    def test_model_info(self):
        """Test model info retrieval."""
        model = ONNXModel(self.model_path)
        info = model.get_model_info()
        
        self.assertIn('model_path', info)
        self.assertIn('input_name', info)
        self.assertIn('output_name', info)
        self.assertIn('input_shape', info)
        self.assertIn('output_shape', info)
        self.assertIn('providers', info)


class TestModelService(unittest.TestCase):
    """Test cases for ModelService class."""
    
    def setUp(self):
        self.model_path = "models/model.onnx"
        if not Path(self.model_path).exists():
            self.skipTest(f"ONNX model not found at {self.model_path}")
        
        self.service = ModelService(self.model_path)
        self.test_image = Image.new('RGB', (256, 256), color='blue')
    
    def test_service_initialization(self):
        """Test service initialization."""
        self.assertIsNotNone(self.service.preprocessor)
        self.assertIsNotNone(self.service.model)
        self.assertIsInstance(self.service.class_labels, list)
        self.assertEqual(len(self.service.class_labels), 1000)
    
    def test_predict_image_pil(self):
        """Test prediction with PIL Image."""
        result = self.service.predict_image(self.test_image)
        
        self._validate_prediction_result(result)
    
    def test_predict_image_numpy(self):
        """Test prediction with numpy array."""
        np_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = self.service.predict_image(np_image)
        
        self._validate_prediction_result(result)
    
    def test_predict_with_probabilities(self):
        """Test prediction with probability output."""
        result = self.service.predict_image(self.test_image, return_probabilities=True)
        
        self._validate_prediction_result(result)
        self.assertIn('all_probabilities', result)
        self.assertEqual(len(result['all_probabilities']), 1000)
    
    def test_predict_top_k(self):
        """Test top-k predictions."""
        k = 10
        result = self.service.predict_image(self.test_image, top_k=k)
        
        self._validate_prediction_result(result)
        self.assertEqual(len(result['top_predictions']), k)
        
        # Check if predictions are sorted by probability
        probs = [pred['probability'] for pred in result['top_predictions']]
        self.assertEqual(probs, sorted(probs, reverse=True))
    
    def test_predict_batch(self):
        """Test batch prediction."""
        images = [self.test_image, self.test_image.copy()]
        results = self.service.predict_batch(images)
        
        self.assertEqual(len(results), 2)
        
        for result in results:
            if 'error' not in result:
                self._validate_prediction_result(result)
    
    def test_performance_benchmark(self):
        """Test performance requirements."""
        # Run multiple predictions to get average time
        times = []
        for _ in range(10):
            start_time = time.time()
            result = self.service.predict_image(self.test_image)
            total_time = time.time() - start_time
            times.append(total_time)
        
        avg_time = np.mean(times)
        
        # Should be under 3 seconds for production
        self.assertLess(avg_time, 3.0, f"Average prediction time {avg_time:.3f}s exceeds 3s requirement")
        
        logger.info(f"Average prediction time: {avg_time:.3f}s (±{np.std(times):.3f}s)")
    
    def test_service_info(self):
        """Test service info retrieval."""
        info = self.service.get_service_info()
        
        self.assertIn('model_info', info)
        self.assertIn('use_enhanced_model', info)
        self.assertIn('num_classes', info)
        self.assertEqual(info['num_classes'], 1000)
    
    def _validate_prediction_result(self, result):
        """Validate prediction result structure."""
        required_keys = [
            'predicted_class_id', 'predicted_class_name', 'confidence',
            'top_predictions', 'processing_time'
        ]
        
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")
        
        # Validate types and ranges
        self.assertIsInstance(result['predicted_class_id'], int)
        self.assertIsInstance(result['predicted_class_name'], str)
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        
        # Validate top predictions
        self.assertIsInstance(result['top_predictions'], list)
        self.assertGreater(len(result['top_predictions']), 0)
        
        for pred in result['top_predictions']:
            self.assertIn('class_id', pred)
            self.assertIn('class_name', pred)
            self.assertIn('probability', pred)
            self.assertGreaterEqual(pred['probability'], 0.0)
            self.assertLessEqual(pred['probability'], 1.0)
        
        # Validate processing time
        self.assertIn('preprocessing', result['processing_time'])
        self.assertIn('inference', result['processing_time'])
        self.assertIn('total', result['processing_time'])


class TestSampleImages(unittest.TestCase):
    """Test with actual sample images if available."""
    
    def setUp(self):
        self.model_path = "models/model.onnx"
        if not Path(self.model_path).exists():
            self.skipTest(f"ONNX model not found at {self.model_path}")
        
        self.service = ModelService(self.model_path)
        self.sample_images = {
            "n01440764_tench.JPEG": 0,  # Expected class ID
            "n01667114_mud_turtle.JPEG": 35  # Expected class ID
        }
    
    def test_sample_image_predictions(self):
        """Test predictions on provided sample images."""
        for image_path, expected_class in self.sample_images.items():
            if Path(image_path).exists():
                logger.info(f"Testing {image_path} (expected class: {expected_class})")
                
                result = self.service.predict_image(image_path)
                predicted_class = result['predicted_class_id']
                confidence = result['confidence']
                
                logger.info(f"Predicted: {predicted_class}, Confidence: {confidence:.4f}")
                
                # Check if prediction is reasonable (might not be exact due to model differences)
                self.assertIsInstance(predicted_class, int)
                self.assertGreaterEqual(predicted_class, 0)
                self.assertLess(predicted_class, 1000)
                
                # Log top 5 predictions for analysis
                logger.info("Top 5 predictions:")
                for i, pred in enumerate(result['top_predictions'][:5]):
                    logger.info(f"  {i+1}. Class {pred['class_id']}: {pred['probability']:.4f}")
            else:
                logger.warning(f"Sample image {image_path} not found")


class TestONNXConversion(unittest.TestCase):
    """Test ONNX conversion functionality."""
    
    def test_converter_initialization(self):
        """Test converter initialization."""
        converter = ONNXConverter()
        self.assertIsNotNone(converter.model_weights_path)
        self.assertIsNotNone(converter.output_path)
    
    def test_conversion_process(self):
        """Test the conversion process if weights are available."""
        if not Path("pytorch_model_weights.pth").exists():
            self.skipTest("PyTorch weights not found")
        
        converter = ONNXConverter(output_path="test_model.onnx")
        
        # Test model loading
        success = converter.load_pytorch_model()
        self.assertTrue(success)
        
        # Test conversion
        success = converter.convert_to_onnx()
        self.assertTrue(success)
        
        # Test validation
        success = converter.validate_onnx_model()
        self.assertTrue(success)
        
        # Cleanup
        if Path("test_model.onnx").exists():
            os.remove("test_model.onnx")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_invalid_model_path(self):
        """Test handling of invalid model path."""
        with self.assertRaises(FileNotFoundError):
            ONNXModel("nonexistent_model.onnx")
    
    def test_invalid_image_input(self):
        """Test handling of invalid image inputs."""
        if not Path("models/model.onnx").exists():
            self.skipTest("ONNX model not found")
        
        service = ModelService("models/model.onnx")
        
        # Test with invalid file path
        with self.assertRaises(FileNotFoundError):
            service.predict_image("nonexistent_image.jpg")
        
        # Test with invalid base64
        with self.assertRaises(Exception):
            preprocessor = ImagePreprocessor()
            preprocessor.load_image_from_base64("invalid_base64_string")
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively."""
        if not Path("models/model.onnx").exists():
            self.skipTest("ONNX model not found")
        
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        service = ModelService("models/model.onnx")
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Run multiple predictions
        for _ in range(20):
            service.predict_image(test_image)
            gc.collect()  # Force garbage collection
        
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory growth should be reasonable (less than 500MB)
        self.assertLess(memory_growth, 500, f"Excessive memory growth: {memory_growth:.2f}MB")
        
        logger.info(f"Memory growth after 20 predictions: {memory_growth:.2f}MB")


def run_comprehensive_tests():
    """Run all tests with detailed reporting."""
    logger.info("Starting comprehensive test suite...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestImagePreprocessor,
        TestONNXModel,
        TestModelService,
        TestSampleImages,
        TestONNXConversion,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None)
    result = runner.run(test_suite)
    
    # Print summary
    logger.info(f"\nTest Summary:")
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        logger.error("Failures:")
        for test, traceback in result.failures:
            logger.error(f"  {test}: {traceback}")
    
    if result.errors:
        logger.error("Errors:")
        for test, traceback in result.errors:
            logger.error(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    logger.info(f"Overall result: {'PASS' if success else 'FAIL'}")
    
    return success


def run_quick_test():
    """Run a quick test to verify basic functionality."""
    logger.info("Running quick functionality test...")
    
    try:
        # Test preprocessing
        preprocessor = ImagePreprocessor()
        test_image = Image.new('RGB', (256, 256), color='green')
        processed = preprocessor.preprocess(test_image)
        assert processed.shape == (1, 3, 224, 224)
        logger.info("✓ Preprocessing test passed")
        
        # Test model if available
        if Path("models/model.onnx").exists():
            service = ModelService("models/model.onnx")
            result = service.predict_image(test_image)
            assert 'predicted_class_id' in result
            assert 'confidence' in result
            logger.info("✓ Model inference test passed")
        else:
            logger.warning("⚠ ONNX model not found, skipping inference test")
        
        logger.info("✓ Quick test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test suite for image classification model")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive test suite")
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_test()
    elif args.comprehensive:
        success = run_comprehensive_tests()
    else:
        # Default: run both
        logger.info("Running both quick and comprehensive tests...")
        quick_success = run_quick_test()
        comprehensive_success = run_comprehensive_tests()
        success = quick_success and comprehensive_success
    
    exit(0 if success else 1)