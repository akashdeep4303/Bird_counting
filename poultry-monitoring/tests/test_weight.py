"""
Test weight estimation functionality
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import estimate_weight_proxy


class TestWeightEstimation:
    """Test weight estimation functionality"""
    
    def test_weight_estimation_output_format(self):
        """Test that weight estimation returns correct format"""
        area = 1000
        aspect_ratio = 1.0
        avg_area = 1000
        
        result = estimate_weight_proxy(area, aspect_ratio, avg_area)
        
        # Check output structure
        assert 'weight_index' in result
        assert 'confidence' in result
        assert 'features' in result
        assert 'area' in result['features']
        assert 'aspect_ratio' in result['features']
        
        # Check data types
        assert isinstance(result['weight_index'], (int, float))
        assert isinstance(result['confidence'], (int, float))
        assert isinstance(result['features']['area'], int)
        assert isinstance(result['features']['aspect_ratio'], (int, float))
    
    def test_weight_index_range(self):
        """Test that weight index is within reasonable range"""
        area = 1000
        aspect_ratio = 1.0
        avg_area = 1000
        
        result = estimate_weight_proxy(area, aspect_ratio, avg_area)
        
        # Weight index should be between 0 and 100
        assert 0 <= result['weight_index'] <= 100
    
    def test_confidence_range(self):
        """Test that confidence is between 0 and 1"""
        area = 1000
        aspect_ratio = 1.0
        avg_area = 1000
        
        result = estimate_weight_proxy(area, aspect_ratio, avg_area)
        
        # Confidence should be between 0 and 1
        assert 0 <= result['confidence'] <= 1
    
    def test_larger_area_higher_weight(self):
        """Test that larger area results in higher weight index"""
        avg_area = 1000
        aspect_ratio = 1.0
        
        result_small = estimate_weight_proxy(500, aspect_ratio, avg_area)
        result_large = estimate_weight_proxy(1500, aspect_ratio, avg_area)
        
        # Larger bird should have higher weight index
        assert result_large['weight_index'] > result_small['weight_index']
    
    def test_aspect_ratio_effect(self):
        """Test that aspect ratio affects weight estimation"""
        area = 1000
        avg_area = 1000
        
        # More square shape (closer to 1.0)
        result_square = estimate_weight_proxy(area, 1.0, avg_area)
        
        # More elongated shape
        result_elongated = estimate_weight_proxy(area, 0.5, avg_area)
        
        # Square shape should have higher weight index
        assert result_square['weight_index'] >= result_elongated['weight_index']
    
    def test_zero_avg_area(self):
        """Test handling of zero average area"""
        area = 1000
        aspect_ratio = 1.0
        avg_area = 0
        
        # Should not crash
        result = estimate_weight_proxy(area, aspect_ratio, avg_area)
        
        assert result is not None
        assert 'weight_index' in result
    
    def test_extreme_values(self):
        """Test with extreme values"""
        # Very large area
        result_large = estimate_weight_proxy(10000, 1.0, 1000)
        assert result_large['weight_index'] <= 100
        
        # Very small area
        result_small = estimate_weight_proxy(10, 1.0, 1000)
        assert result_small['weight_index'] >= 0
        
        # Extreme aspect ratios
        result_wide = estimate_weight_proxy(1000, 0.1, 1000)
        assert 0 <= result_wide['weight_index'] <= 100
        
        result_tall = estimate_weight_proxy(1000, 5.0, 1000)
        assert 0 <= result_tall['weight_index'] <= 100
    
    def test_normalized_bird(self):
        """Test weight estimation for bird with average size"""
        area = 1000
        aspect_ratio = 1.0
        avg_area = 1000
        
        result = estimate_weight_proxy(area, aspect_ratio, avg_area)
        
        # Average bird with perfect aspect ratio should score around 50
        assert 40 <= result['weight_index'] <= 60
    
    def test_features_preservation(self):
        """Test that input features are preserved in output"""
        area = 1234
        aspect_ratio = 0.87
        avg_area = 1000
        
        result = estimate_weight_proxy(area, aspect_ratio, avg_area)
        
        # Features should match input
        assert result['features']['area'] == int(area)
        assert result['features']['aspect_ratio'] == round(aspect_ratio, 2)
    
    def test_different_scenarios(self):
        """Test multiple realistic scenarios"""
        avg_area = 1000
        
        scenarios = [
            # (area, aspect_ratio, description)
            (1500, 1.0, "Large healthy bird"),
            (800, 0.9, "Small bird"),
            (1200, 0.7, "Medium bird, elongated"),
            (2000, 1.2, "Very large bird"),
        ]
        
        for area, aspect_ratio, description in scenarios:
            result = estimate_weight_proxy(area, aspect_ratio, avg_area)
            
            # All should produce valid results
            assert 0 <= result['weight_index'] <= 100
            assert 0 <= result['confidence'] <= 1
            print(f"{description}: weight_index={result['weight_index']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

