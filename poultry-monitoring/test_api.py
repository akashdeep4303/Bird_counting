"""
Test script for the Bird Counting API
Usage: python test_api.py <video_path>
"""

import requests
import sys
import json
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_analyze_video(video_path: str, fps_sample: int = 2):
    """Test video analysis endpoint"""
    print(f"Testing /analyze_video with {video_path}...")
    
    if not Path(video_path).exists():
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Prepare request
    files = {
        'file': ('video.mp4', open(video_path, 'rb'), 'video/mp4')
    }
    data = {
        'fps_sample': fps_sample,
        'conf_thresh': 0.25,
        'iou_thresh': 0.45
    }
    
    print("Uploading video and processing (this may take a few minutes)...")
    response = requests.post(
        f"{API_URL}/analyze_video",
        files=files,
        data=data
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n=== Analysis Results ===")
        print(f"Total frames: {result['metadata']['total_frames']}")
        print(f"FPS: {result['metadata']['fps']}")
        print(f"Birds tracked: {result['weight_estimates']['aggregate']['total_birds_tracked']}")
        print(f"Average weight index: {result['weight_estimates']['aggregate']['average_weight_index']}")
        print(f"Annotated video: {result['artifacts']['annotated_video']}")
        
        # Save full response
        output_file = "test_response.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nFull response saved to {output_file}")
        
        # Display sample counts
        print("\n=== Sample Counts (first 5) ===")
        for count in result['counts'][:5]:
            print(f"Time: {count['timestamp']}s, Frame: {count['frame']}, Count: {count['count']}")
        
        # Display sample weight estimates
        print("\n=== Sample Weight Estimates (first 3 birds) ===")
        for bird in result['weight_estimates']['per_bird'][:3]:
            print(f"Bird ID: {bird['bird_id']}")
            print(f"  Weight Index: {bird['weight_index']}")
            print(f"  Confidence: {bird['confidence']}")
            print(f"  Area: {bird['features']['area']} pixels")
            print()
    else:
        print(f"Error: {response.text}")

def main():
    """Main test function"""
    # Test health endpoint
    test_health()
    
    # Test video analysis if video path provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        fps_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        test_analyze_video(video_path, fps_sample)
    else:
        print("Usage: python test_api.py <video_path> [fps_sample]")
        print("Example: python test_api.py data/videos/sample.mp4 2")

if __name__ == "__main__":
    main()