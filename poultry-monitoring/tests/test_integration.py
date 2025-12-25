"""
Integration tests for the complete bird counting system
"""

import pytest
import requests
import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

API_URL = "http://localhost:8000"
TEST_VIDEO = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'data', 'videos', 'sample_poultry.mp4'
)


class TestIntegration:
    """Integration tests for the full system"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        # Check if API is running
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code != 200:
                pytest.skip("API server is not running")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server is not running. Start with: python main.py")
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{API_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert data['status'] == 'OK'
        assert 'timestamp' in data
    
    def test_analyze_video_endpoint(self):
        """Test video analysis endpoint with sample video"""
        if not os.path.exists(TEST_VIDEO):
            pytest.skip("Sample video not found")
        
        # Prepare request
        files = {
            'file': ('test_video.mp4', open(TEST_VIDEO, 'rb'), 'video/mp4')
        }
        data = {
            'fps_sample': 5,
            'conf_thresh': 0.25,
            'iou_thresh': 0.45
        }
        
        # Send request
        response = requests.post(
            f"{API_URL}/analyze_video",
            files=files,
            data=data,
            timeout=120
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Check response structure
        assert 'counts' in result
        assert 'tracks_sample' in result
        assert 'weight_estimates' in result
        assert 'artifacts' in result
        assert 'metadata' in result
        
        # Check metadata
        assert result['metadata']['total_frames'] > 0
        assert result['metadata']['fps'] > 0
        
        # Check weight estimates
        assert 'per_bird' in result['weight_estimates']
        assert 'aggregate' in result['weight_estimates']
        assert 'total_birds_tracked' in result['weight_estimates']['aggregate']
        assert 'average_weight_index' in result['weight_estimates']['aggregate']
        
        # Check artifacts
        assert 'annotated_video' in result['artifacts']
        
        # Verify annotated video was created
        output_video = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'outputs',
            result['artifacts']['annotated_video']
        )
        assert os.path.exists(output_video), f"Annotated video not found: {output_video}"
        assert os.path.getsize(output_video) > 0, "Annotated video is empty"
    
    def test_invalid_file_format(self):
        """Test error handling for invalid file format"""
        # Create a dummy text file
        dummy_content = b"This is not a video file"
        files = {
            'file': ('test.txt', dummy_content, 'text/plain')
        }
        
        response = requests.post(
            f"{API_URL}/analyze_video",
            files=files,
            timeout=10
        )
        
        # Should return 400 error
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data
    
    def test_different_parameters(self):
        """Test with different processing parameters"""
        if not os.path.exists(TEST_VIDEO):
            pytest.skip("Sample video not found")
        
        parameters = [
            {'fps_sample': 1, 'conf_thresh': 0.3, 'iou_thresh': 0.5},
            {'fps_sample': 10, 'conf_thresh': 0.2, 'iou_thresh': 0.4},
        ]
        
        for params in parameters:
            files = {
                'file': ('test_video.mp4', open(TEST_VIDEO, 'rb'), 'video/mp4')
            }
            
            response = requests.post(
                f"{API_URL}/analyze_video",
                files=files,
                data=params,
                timeout=120
            )
            
            assert response.status_code == 200
            result = response.json()
            assert 'metadata' in result
            assert result['metadata']['processing_fps'] == params['fps_sample']
            assert result['metadata']['conf_threshold'] == params['conf_thresh']
    
    def test_output_data_validity(self):
        """Test that output data is valid and within expected ranges"""
        if not os.path.exists(TEST_VIDEO):
            pytest.skip("Sample video not found")
        
        files = {
            'file': ('test_video.mp4', open(TEST_VIDEO, 'rb'), 'video/mp4')
        }
        data = {'fps_sample': 5, 'conf_thresh': 0.25, 'iou_thresh': 0.45}
        
        response = requests.post(
            f"{API_URL}/analyze_video",
            files=files,
            data=data,
            timeout=120
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Check counts
        for count_entry in result['counts']:
            assert 'timestamp' in count_entry
            assert 'frame' in count_entry
            assert 'count' in count_entry
            assert count_entry['count'] >= 0
            assert count_entry['frame'] > 0
            assert count_entry['timestamp'] >= 0
        
        # Check weight estimates
        for bird in result['weight_estimates']['per_bird']:
            assert 'bird_id' in bird
            assert 'weight_index' in bird
            assert 'confidence' in bird
            assert 0 <= bird['weight_index'] <= 100
            assert 0 <= bird['confidence'] <= 1
        
        # Check aggregate
        aggregate = result['weight_estimates']['aggregate']
        assert 0 <= aggregate['average_weight_index'] <= 100
        assert aggregate['total_birds_tracked'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

