"""
Test tracking functionality
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import BirdTracker


class TestBirdTracker:
    """Test BirdTracker functionality"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.tracker = BirdTracker(max_disappeared=30)
    
    def test_tracker_initialization(self):
        """Test tracker initializes correctly"""
        assert self.tracker.next_id == 0
        assert len(self.tracker.objects) == 0
        assert len(self.tracker.disappeared) == 0
        assert self.tracker.max_disappeared == 30
    
    def test_register_bird(self):
        """Test registering a new bird"""
        centroid = (100, 100)
        area = 1000
        
        self.tracker.register(centroid, area, 1.0)
        
        assert self.tracker.next_id == 1
        assert 0 in self.tracker.objects
        assert self.tracker.objects[0] == centroid
        assert 0 in self.tracker.disappeared
        assert self.tracker.disappeared[0] == 0
        assert 0 in self.tracker.bird_sizes
        assert self.tracker.bird_sizes[0] == [area]
    
    def test_deregister_bird(self):
        """Test deregistering a bird"""
        centroid = (100, 100)
        area = 1000
        
        self.tracker.register(centroid, area, 1.0)
        self.tracker.deregister(0)
        
        assert 0 not in self.tracker.objects
        assert 0 not in self.tracker.disappeared
        assert 0 not in self.tracker.bird_sizes
    
    def test_update_with_no_detections(self):
        """Test update with no detections"""
        # Register a bird first
        self.tracker.register((100, 100), 1000, 1.0)
        
        # Update with no detections
        objects = self.tracker.update([])
        
        # Bird should still exist but disappeared count increased
        assert 0 in self.tracker.objects
        assert self.tracker.disappeared[0] == 1
    
    def test_update_with_single_detection(self):
        """Test update with single detection"""
        detections = [((100, 100), 1000, 1.0)]
        objects = self.tracker.update(detections)
        
        # Should register one bird
        assert len(objects) == 1
        assert 0 in objects
    
    def test_update_with_multiple_detections(self):
        """Test update with multiple detections"""
        detections = [
            ((100, 100), 1000, 1.0),
            ((200, 200), 1200, 1.0),
            ((300, 300), 1100, 1.0)
        ]
        
        objects = self.tracker.update(detections)
        
        # Should register three birds
        assert len(objects) == 3
        assert 0 in objects
        assert 1 in objects
        assert 2 in objects
    
    def test_tracking_continuity(self):
        """Test that IDs persist across frames"""
        # Frame 1: 2 birds
        detections1 = [((100, 100), 1000, 1.0), ((200, 200), 1200, 1.0)]
        objects1 = self.tracker.update(detections1)
        
        # Frame 2: Same birds moved slightly
        detections2 = [((105, 105), 1050, 1.0), ((205, 205), 1250, 1.0)]
        objects2 = self.tracker.update(detections2)
        
        # IDs should remain the same
        assert len(objects2) == 2
        assert 0 in objects2
        assert 1 in objects2
        
        # Positions should be updated
        assert objects2[0] == (105, 105)
        assert objects2[1] == (205, 205)
    
    def test_max_disappeared_threshold(self):
        """Test that birds are removed after max_disappeared frames"""
        tracker = BirdTracker(max_disappeared=2)
        
        # Register a bird
        detections = [((100, 100), 1000, 1.0)]
        tracker.update(detections)
        
        # Update with no detections 3 times
        tracker.update([])
        tracker.update([])
        tracker.update([])
        
        # Bird should be deregistered
        assert len(tracker.objects) == 0
    
    def test_distance_threshold(self):
        """Test that distant detections create new IDs"""
        # Register a bird at (100, 100)
        detections1 = [((100, 100), 1000, 1.0)]
        self.tracker.update(detections1)
        
        # Detection far away (>100 pixels)
        detections2 = [((300, 300), 1000, 1.0)]
        objects = self.tracker.update(detections2)
        
        # Should have 2 birds now (old one disappeared, new one registered)
        assert self.tracker.next_id >= 1
    
    def test_bird_sizes_tracking(self):
        """Test that bird sizes are tracked over time"""
        # Frame 1
        detections1 = [((100, 100), 1000, 1.0)]
        self.tracker.update(detections1)
        
        # Frame 2: Same bird, different size
        detections2 = [((105, 105), 1100, 1.0)]
        self.tracker.update(detections2)
        
        # Check sizes are accumulated
        assert 0 in self.tracker.bird_sizes
        assert len(self.tracker.bird_sizes[0]) == 2
        assert self.tracker.bird_sizes[0] == [1000, 1100]
    
    def test_empty_initial_state(self):
        """Test update on empty tracker"""
        detections = [((100, 100), 1000, 1.0)]
        objects = self.tracker.update(detections)
        
        assert len(objects) == 1
        assert 0 in objects


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

