#!/usr/bin/env python3
"""
Script to process video with improved bird detection for background birds
"""
import os
import sys
from main import process_video

if __name__ == "__main__":
    # Input video path
    input_video = "data/videos/2025_12_15_15_24_16_4_dQCiGf.MP4"
    
    # Output video path
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, "annotated_bird_detection_improved.mp4")
    
    if not os.path.exists(input_video):
        print(f"Error: Input video not found: {input_video}")
        sys.exit(1)
    
    print(f"Processing video: {input_video}")
    print("Improved detection settings:")
    print("  - Enhanced frame preprocessing for small birds")
    print("  - Multi-scale detection (original + enhanced frames)")
    print("  - Lower confidence threshold: 0.005")
    print("  - Lower minimum area: 10 pixels")
    print("  - More lenient white color filtering")
    print("  - Higher frame sampling rate: 15 FPS")
    print()
    
    try:
        result = process_video(
            video_path=input_video,
            output_path=output_video,
            fps_sample=15,  # Process more frames
            conf_thresh=0.005,  # Very low threshold for small birds
            iou_thresh=0.35,  # Lower IoU
            require_white=True,
            white_ratio_thresh=0.05,  # Lower white threshold
            white_sat_thresh=90,
            white_val_thresh=100
        )
        
        print("\n" + "="*50)
        print("Processing Complete!")
        print("="*50)
        print(f"Output video: {output_video}")
        print(f"Total detections: {result['detection_summary']['total_detections']}")
        print(f"Average per frame: {result['detection_summary']['average_per_frame']:.2f}")
        print(f"Max per frame: {result['detection_summary']['max_per_frame']}")
        print(f"Total birds tracked: {result['detection_summary']['total_birds_tracked']}")
        print(f"Video size: {result['artifacts']['video_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

