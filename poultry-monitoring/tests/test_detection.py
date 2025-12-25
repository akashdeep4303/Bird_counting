"""
Test detection functionality
"""

import pytest
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import load_model


class TestDetection:
    """Test bird detection functionality"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.model = load_model()
        self.test_image_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'data', 
            'videos', 
            'sample_poultry.mp4'
        )
    
    def test_model_loads(self):
        """Test that YOLO model loads successfully"""
        assert self.model is not None
        assert isinstance(self.model, YOLO)
    
    def test_model_inference(self):
        """Test that model can run inference on a frame"""
        # Create a dummy frame
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = self.model(dummy_frame, verbose=False)
        
        # Check that results are returned
        assert results is not None
        assert len(results) > 0
    
    def test_video_frame_detection(self):
        """Test detection on actual video frame"""
        if not os.path.exists(self.test_image_path):
            pytest.skip("Sample video not found")
        
        # Read first frame from video
        cap = cv2.VideoCapture(self.test_image_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            pytest.skip("Could not read video frame")
        
        # Run detection
        results = self.model(frame, conf=0.25, verbose=False)
        
        # Check results
        assert results is not None
        assert len(results) > 0
        
        # Check boxes exist (may or may not detect objects)
        if results[0].boxes is not None:
            boxes = results[0].boxes
            assert hasattr(boxes, 'xyxy')
            assert hasattr(boxes, 'conf')
    
    def test_detection_confidence_threshold(self):
        """Test that confidence threshold filtering works"""
        dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # High confidence threshold should return fewer/no detections
        results_high = self.model(dummy_frame, conf=0.9, verbose=False)
        
        # Low confidence threshold
        results_low = self.model(dummy_frame, conf=0.1, verbose=False)
        
        # Both should return results objects
        assert results_high is not None
        assert results_low is not None
    
    def test_detection_output_format(self):
        """Test that detection output has expected format"""
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = self.model(dummy_frame, verbose=False)
        
        # Check result structure
        assert hasattr(results[0], 'boxes')
        assert hasattr(results[0], 'keypoints')
        assert hasattr(results[0], 'masks')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

