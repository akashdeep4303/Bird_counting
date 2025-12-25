from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from typing import Optional
import json
from datetime import datetime
from collections import defaultdict
import logging
import subprocess
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Poultry Monitoring API", version="1.0.0")

# Global model instance
model = None

def load_model():
    """Load YOLO model for bird detection"""
    global model
    if model is None:
        # Using YOLOv8 nano for faster processing
        # For production, use yolov8m or yolov8l for better accuracy
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'yolov8n.pt')
        if not os.path.exists(model_path):
            # Fallback to current directory or download
            model_path = 'yolov8n.pt'
        model = YOLO(model_path)
    return model

def compute_whiteness_ratio(frame, bbox, sat_thresh=60, val_thresh=160):
    """
    Calculate the ratio of white-looking pixels inside a bounding box.
    White regions have low saturation and high value in HSV space.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return 0.0
    
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    white_mask = (sat < sat_thresh) & (val > val_thresh)
    return float(white_mask.mean())

class BirdTracker:
    """Simple centroid-based tracker with ID persistence"""
    
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}  # {id: centroid}
        self.disappeared = {}  # {id: frames_disappeared}
        self.max_disappeared = max_disappeared
        self.bird_sizes = {}    # {id: [areas]}
        self.bird_aspects = {}  # {id: [aspect_ratios]}
        
    def register(self, centroid, area, aspect_ratio):
        """Register new bird"""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.bird_sizes[self.next_id] = [area]
        self.bird_aspects[self.next_id] = [aspect_ratio]
        self.next_id += 1
        
    def deregister(self, object_id):
        """Remove bird that has disappeared"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.bird_sizes:
            del self.bird_sizes[object_id]
        if object_id in self.bird_aspects:
            del self.bird_aspects[object_id]
    
    def update(self, detections):
        """
        Update tracker with new detections
        detections: list of (centroid, area, aspect_ratio) tuples
        """
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        centroids = [d[0] for d in detections]
        areas = [d[1] for d in detections]
        aspects = [d[2] for d in detections]
        
        # If no existing objects, register all
        if len(self.objects) == 0:
            for i in range(len(centroids)):
                self.register(centroids[i], areas[i], aspects[i])
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distance matrix
            D = np.zeros((len(object_centroids), len(centroids)))
            for i, oc in enumerate(object_centroids):
                for j, dc in enumerate(centroids):
                    D[i, j] = np.linalg.norm(np.array(oc) - np.array(dc))
            
            # Find minimum distances
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Update matched objects
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                # Distance threshold (adjust based on video resolution)
                # Increased threshold for high-resolution videos with many birds
                if D[row, col] > 150:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = centroids[col]
                self.disappeared[object_id] = 0
                self.bird_sizes[object_id].append(areas[col])
                self.bird_aspects[object_id].append(aspects[col])
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Mark disappeared objects
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(centroids[col], areas[col], aspects[col])
        
        return self.objects

def convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable types to native Python types"""
    import numpy as np
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

def estimate_weight_proxy(area, aspect_ratio, avg_area):
    """
    Estimate weight proxy based on visual features
    Returns: weight index (0-100 scale)
    
    Assumptions:
    - Larger birds (by pixel area) are heavier
    - More square-shaped birds (aspect_ratio ~1) are fuller/heavier
    - Weight index calibrated to farm-specific data needed for actual grams
    """
    # Normalize area relative to average
    if avg_area > 0:
        area_score = min(100, (area / avg_area) * 50)
    else:
        area_score = 50
    
    # Aspect ratio score (prefer more square shapes)
    # Healthy birds typically have aspect ratio 0.7-1.3
    # Clamp aspect_ratio to reasonable range
    aspect_ratio_clamped = max(0.3, min(3.0, aspect_ratio))
    aspect_score = 50 * (1 - min(1.0, abs(1.0 - aspect_ratio_clamped)))
    
    # Combined weight index
    weight_index = 0.7 * area_score + 0.3 * aspect_score
    
    # Ensure weight_index is in valid range
    weight_index = max(0, min(100, weight_index))
    
    return {
        'weight_index': float(round(weight_index, 2)),
        'confidence': float(0.75 if 30 < area_score < 70 else 0.6),
        'features': {
            'area': int(area),
            'aspect_ratio': float(round(aspect_ratio, 2))
        }
    }

def enhance_frame_for_detection(frame):
    """
    Enhance frame to improve detection of small/background birds
    - Increase contrast and brightness
    - Apply sharpening filter
    - Enhance edges
    """
    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Apply sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Blend original and sharpened (70% sharpened, 30% original)
    result = cv2.addWeighted(sharpened, 0.7, enhanced, 0.3, 0)
    
    return result

def process_video(
    video_path: str,
    output_path: str,
    fps_sample: int = 15,  # Process more frames to catch all birds including background
    conf_thresh: float = 0.005,  # Even lower threshold to detect small/background birds
    iou_thresh: float = 0.35,  # Lower IoU to allow more overlapping detections
    filter_bird_only: bool = False,  # Detect all objects, filter by white color
    require_white: bool = True,
    white_ratio_thresh: float = 0.05,  # Much lower threshold for white detection (background birds)
    white_sat_thresh: int = 90,  # More lenient saturation threshold
    white_val_thresh: int = 100  # Lower value threshold to catch lighter whites and background birds
):
    """
    Process video for bird detection, tracking, and weight estimation
    Counts birds by bounding boxes detected in each frame
    """
    model = load_model()
    tracker = BirdTracker(max_disappeared=30)  # Increased to handle occlusions better
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if width == 0 or height == 0:
        raise ValueError("Invalid video dimensions")
    
    # Clean up any leftover temp files from previous runs
    temp_pattern = output_path.replace('.mp4', '_temp.mp4')
    if os.path.exists(temp_pattern):
        try:
            os.remove(temp_pattern)
            logger.info(f"Cleaned up leftover temp file: {temp_pattern}")
        except Exception as e:
            logger.warning(f"Could not remove temp file {temp_pattern}: {e}")
    
    # Use ffmpeg for better video encoding compatibility
    # Check if ffmpeg is available
    ffmpeg_available = shutil.which('ffmpeg') is not None
    temp_video = None
    used_codec = None
    
    if ffmpeg_available:
        # Use ffmpeg for better compatibility - write to temporary file first
        temp_video = output_path.replace('.mp4', '_temp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        if out.isOpened():
            used_codec = 'mp4v (will re-encode with ffmpeg)'
            logger.info("Using ffmpeg for final video encoding (better compatibility)")
        else:
            ffmpeg_available = False  # Fallback if temp write fails
            out = None
    
    if not ffmpeg_available or out is None:
        # Fallback to OpenCV with best available codec
        fourcc_options = [
            ('avc1', cv2.VideoWriter_fourcc(*'avc1')),  # H.264
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4
        ]
        
        out = None
        for codec_name, fourcc in fourcc_options:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                used_codec = codec_name
                logger.info(f"Video writer initialized with codec: {codec_name}")
                break
            else:
                out = None
        
        if out is None or not out.isOpened():
            raise RuntimeError(f"Failed to initialize video writer for {output_path}")
    
    # Data collection
    counts_over_time = []
    tracks_sample = []
    all_areas = []
    frame_count = 0
    process_every_n = max(1, fps // fps_sample)
    objects = {}  # Initialize objects dictionary
    current_detections = []  # Store current frame detections for drawing
    current_boxes = []  # Store current frame boxes for drawing
    
    logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
    logger.info(f"Output video: {output_path} ({width}x{height})")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps
        
        # Sample frames based on fps_sample
        if frame_count % process_every_n == 0:
            # Enhance frame for better detection of small/background birds
            enhanced_frame = enhance_frame_for_detection(frame)
            
            # Run detection on both original and enhanced frames for multi-scale detection
            # This helps catch birds at different scales and lighting conditions
            results_original = model(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)
            results_enhanced = model(enhanced_frame, conf=conf_thresh, iou=iou_thresh, verbose=False)
            
            # Combine results from both detections
            # Merge boxes from both results
            all_boxes = []
            if len(results_original) > 0 and results_original[0].boxes is not None:
                all_boxes.extend(results_original[0].boxes)
            if len(results_enhanced) > 0 and results_enhanced[0].boxes is not None:
                all_boxes.extend(results_enhanced[0].boxes)
            
            # Create a combined results object for processing
            results = results_original  # Use original for structure, but we'll process all_boxes
            
            # Extract detections
            detections = []
            boxes_data = []
            current_boxes = []  # Reset for this frame
            seen_boxes = set()  # Track processed boxes to avoid duplicates
            
            # Process all boxes from combined detections
            # Get names from original results (both should have same class names)
            names = results_original[0].names if len(results_original) > 0 and hasattr(results_original[0], 'names') else None
            
            if len(all_boxes) > 0:
                boxes = all_boxes
                for box in boxes:
                    # Class filtering: keep only "bird" class when available
                    keep = True
                    class_name = None
                    if hasattr(box, 'cls') and names:
                        class_id = int(box.cls[0].item())
                        class_name = names.get(class_id, None)
                        if filter_bird_only and class_name != 'bird':
                            keep = False
                    if not keep:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Convert to native Python types
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    conf = float(box.conf[0])
                    
                    # Create a unique key for this box to avoid duplicates
                    box_key = (int(x1), int(y1), int(x2), int(y2))
                    if box_key in seen_boxes:
                        continue
                    seen_boxes.add(box_key)
                    
                    # Color-based filtering: birds in this setup are white
                    # Calculate area to filter out very small detections (likely noise)
                    w = (x2 - x1)
                    h = (y2 - y1)
                    area = w * h
                    
                    # Skip very small detections (likely false positives)
                    # Much lower threshold to catch small birds in background
                    min_area = 10  # Minimum area in pixels (lowered for small/background birds)
                    if area < min_area:
                        continue
                    
                    # Check whiteness on both original and enhanced frames
                    whiteness_orig = compute_whiteness_ratio(
                        frame, (x1, y1, x2, y2),
                        sat_thresh=white_sat_thresh,
                        val_thresh=white_val_thresh
                    )
                    whiteness_enh = compute_whiteness_ratio(
                        enhanced_frame, (x1, y1, x2, y2),
                        sat_thresh=white_sat_thresh,
                        val_thresh=white_val_thresh
                    )
                    # Use the maximum whiteness from both frames
                    whiteness = max(whiteness_orig, whiteness_enh)
                    
                    if require_white and whiteness < white_ratio_thresh:
                        # Skip non-white objects such as red buckets
                        continue
                    
                    # Calculate centroid and aspect ratio
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)
                    aspect_ratio = float(w / max(h, 1e-6))
                    area = float(area)  # Ensure area is native float
                    
                    detections.append(((cx, cy), area, aspect_ratio))
                    boxes_data.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf,
                        'area': float(area),
                        'aspect_ratio': float(aspect_ratio),
                        'class': class_name if class_name else 'unknown',
                        'whiteness': round(whiteness, 3)
                    })
                    current_boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
                    all_areas.append(area)
            
            # Update tracker
            objects = tracker.update(detections)
            current_detections = detections
            
            # Count birds by bounding boxes detected (not just tracked)
            detected_count = len(boxes_data)
            tracked_count = len(objects)
            
            # Record count (use detected count for accuracy)
            counts_over_time.append({
                'timestamp': round(timestamp, 2),
                'frame': frame_count,
                'count': detected_count,  # Count by bounding boxes
                'tracked_count': tracked_count  # Also include tracked count
            })
            
            # Store sample tracks (first 100 frames)
            if frame_count < 100 * process_every_n and len(boxes_data) > 0:
                tracks_sample.append({
                    'timestamp': round(timestamp, 2),
                    'frame': frame_count,
                    'detections': boxes_data[:5]  # Limit to 5 per frame
                })
        else:
            # Clear detections for non-processed frames
            current_boxes = []
            current_detections = []
        
        # Annotate frame with tracking
        annotated = frame.copy()
        
        # Draw bounding boxes for ALL detected birds
        detected_count = len(current_boxes)
        for i, (x1, y1, x2, y2, conf) in enumerate(current_boxes):
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            conf_text = f"{conf:.2f}"
            cv2.putText(annotated, conf_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw tracked objects (centroids and IDs)
        for object_id, centroid in objects.items():
            # Draw centroid
            cv2.circle(annotated, (int(centroid[0]), int(centroid[1])), 5, (255, 0, 0), -1)
            
            # Draw ID
            text = f"ID:{object_id}"
            cv2.putText(annotated, text, (int(centroid[0]) - 15, int(centroid[1]) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add count overlay - show detected count (by bounding boxes)
        cv2.rectangle(annotated, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.putText(annotated, f"Birds Detected: {detected_count}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Birds Tracked: {len(objects)}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(annotated, f"Time: {timestamp:.1f}s", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Ensure frame is in correct format (uint8, BGR)
        if annotated.dtype != np.uint8:
            annotated = annotated.astype(np.uint8)
        if annotated.shape[2] != 3:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
        
        # Write frame (on macOS, write() may return False even when successful)
        success = out.write(annotated)
        if not success and frame_count % 1000 == 0:
            logger.warning(f"Frame write returned False at frame {frame_count} (may be normal on macOS)")
        
        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    if out:
        out.release()
        logger.info(f"Video writer released")
    
    # Re-encode with ffmpeg for better compatibility if available
    if ffmpeg_available and temp_video and os.path.exists(temp_video):
        try:
            logger.info("Re-encoding video with ffmpeg (H.264) for better compatibility...")
            cmd = [
                'ffmpeg', '-y', '-i', temp_video,
                '-c:v', 'libx264', 
                '-preset', 'medium',  # Better compression
                '-crf', '22',  # Good quality/compression balance
                '-pix_fmt', 'yuv420p',  # Ensure compatibility with all players
                '-movflags', '+faststart',  # Enable streaming/quick start
                '-max_muxing_queue_size', '1024',  # Prevent muxing errors
                '-r', str(fps),  # Ensure correct frame rate
                '-an',  # Remove audio track (no audio in input)
                '-profile:v', 'high',  # H.264 high profile for better compatibility
                '-level', '4.0',  # H.264 level 4.0
                output_path
            ]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                timeout=300,
                text=True
            )
            if result.returncode == 0:
                # Remove temporary file
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                    logger.info("Video successfully re-encoded with H.264")
                    logger.info("Temporary file cleaned up")
            else:
                logger.warning(f"ffmpeg re-encoding failed: {result.stderr}")
                # Fallback to temp file
                if os.path.exists(temp_video):
                    os.replace(temp_video, output_path)
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg encoding timed out, using original file")
            if os.path.exists(temp_video):
                os.replace(temp_video, output_path)
        except Exception as e:
            logger.warning(f"Could not re-encode video: {e}")
            if os.path.exists(temp_video):
                os.replace(temp_video, output_path)
        finally:
            # Ensure temp file is always cleaned up if it still exists
            if temp_video and os.path.exists(temp_video) and os.path.exists(output_path):
                try:
                    os.remove(temp_video)
                    logger.info("Cleaned up leftover temporary file")
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_video}: {e}")
    
    # Verify output file was created
    if not os.path.exists(output_path):
        raise RuntimeError(f"Output video file was not created: {output_path}")
    
    file_size = os.path.getsize(output_path)
    if file_size == 0:
        raise RuntimeError(f"Output video file is empty: {output_path}")
    
    logger.info(f"Video saved to: {output_path}")
    logger.info(f"Output video size: {file_size / 1024 / 1024:.2f} MB")
    
    # Calculate weight estimates
    # Convert numpy types to native Python types for JSON serialization
    avg_area = float(np.mean(all_areas)) if all_areas else 0.0
    weight_estimates = []
    
    for bird_id, sizes in tracker.bird_sizes.items():
        if len(sizes) > 0:
            avg_size = float(np.mean(sizes))
            aspects = tracker.bird_aspects.get(bird_id, [])
            avg_aspect = float(np.mean(aspects)) if len(aspects) > 0 else 1.0
            
            weight_data = estimate_weight_proxy(avg_size, avg_aspect, avg_area)
            weight_data['bird_id'] = int(bird_id)  # Ensure int
            weight_data['unit'] = 'index'
            # Ensure all numeric values are native Python types
            weight_data['weight_index'] = float(weight_data['weight_index'])
            weight_data['confidence'] = float(weight_data['confidence'])
            weight_data['features']['area'] = int(weight_data['features']['area'])
            weight_data['features']['aspect_ratio'] = float(weight_data['features']['aspect_ratio'])
            weight_estimates.append(weight_data)
    
    # Calculate aggregate statistics
    if weight_estimates:
        avg_weight_index = float(np.mean([w['weight_index'] for w in weight_estimates]))
    else:
        avg_weight_index = 0.0
    
    # Calculate total birds detected (by bounding boxes)
    total_detected = sum([c['count'] for c in counts_over_time])
    avg_detected_per_frame = float(np.mean([c['count'] for c in counts_over_time])) if counts_over_time else 0.0
    max_detected = max([c['count'] for c in counts_over_time]) if counts_over_time else 0
    
    result = {
        'counts': counts_over_time,
        'tracks_sample': tracks_sample[:20],  # Limit sample size
        'weight_estimates': {
            'per_bird': weight_estimates,
            'aggregate': {
                'average_weight_index': float(round(avg_weight_index, 2)),
                'total_birds_tracked': len(weight_estimates),
                'unit': 'index',
                'calibration_needed': 'Requires known weights for conversion to grams'
            }
        },
        'detection_summary': {
            'total_detections': total_detected,
            'average_per_frame': float(round(avg_detected_per_frame, 2)),
            'max_per_frame': max_detected,
            'counting_method': 'Bounding boxes detected in each frame',
            'total_birds_tracked': len(weight_estimates)
        },
        'artifacts': {
            'annotated_video': os.path.basename(output_path),
            'video_path': output_path,
            'video_size_mb': float(round(os.path.getsize(output_path) / 1024 / 1024, 2))
        },
        'metadata': {
            'total_frames': int(total_frames),
            'fps': int(fps),
            'processing_fps': int(fps_sample),
            'conf_threshold': float(conf_thresh),
            'iou_threshold': float(iou_thresh),
            'video_dimensions': f"{width}x{height}",
            'color_filter': 'white_only' if require_white else 'disabled',
            'white_ratio_threshold': float(white_ratio_thresh)
        }
    }
    
    # Convert all numpy types to native Python types for JSON serialization
    return convert_to_json_serializable(result)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "OK", "timestamp": datetime.now().isoformat()}

@app.post("/analyze_video")
async def analyze_video(
    file: UploadFile = File(...),
    fps_sample: Optional[int] = Form(2),
    conf_thresh: Optional[float] = Form(0.25),
    iou_thresh: Optional[float] = Form(0.45),
    require_white: Optional[bool] = Form(True)
):
    """
    Analyze poultry video for bird counting and weight estimation
    
    Parameters:
    - file: Video file (MP4, AVI, etc.)
    - fps_sample: Process N frames per second (default: 2)
    - conf_thresh: Detection confidence threshold (default: 0.25)
    - iou_thresh: IoU threshold for NMS (default: 0.45)
    - require_white: If true, keep only detections whose pixels are mostly white
    
    Returns:
    - JSON with counts, tracks, weight estimates, and artifact paths
    """
    try:
        # Validate file
        if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid file format. Use MP4, AVI, MOV, or MKV"}
            )
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
            content = await file.read()
            tmp_input.write(content)
            input_path = tmp_input.name
        
        # Create output path
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"annotated_{timestamp}.mp4")
        
        # Handle boolean form inputs that may come in as strings
        white_filter_enabled = True
        if isinstance(require_white, str):
            white_filter_enabled = require_white.lower() not in ("false", "0", "no", "off")
        elif require_white is not None:
            white_filter_enabled = bool(require_white)
        
        # Process video
        logger.info(f"Processing video: {file.filename}")
        result = process_video(
            input_path,
            output_path,
            fps_sample=fps_sample,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            require_white=white_filter_enabled
        )
        
        # Clean up temp file
        os.unlink(input_path)
        
        logger.info("Processing complete")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)