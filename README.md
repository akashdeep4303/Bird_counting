# Bird Counting and Weight Estimation System

A computer vision solution for real-time poultry monitoring using CCTV footage. This system provides accurate bird counting through detection and tracking, along with weight estimation capabilities.

## 🎯 Features

- **Enhanced Bird Detection**: YOLOv8-based object detection optimized for bird detection
  - **Background Bird Detection**: Advanced frame preprocessing (CLAHE + sharpening) and multi-scale detection to catch birds with minimal movement in background areas
  - **Small Bird Detection**: Lower thresholds (conf: 0.005, min_area: 10px) and enhanced image processing for detecting smaller birds
  - **Duplicate Prevention**: Automatic removal of duplicate detections from multi-scale processing
- **Stable Tracking**: Centroid-based tracking with ID persistence
- **Weight Estimation**: Feature-based weight proxy calculation
- **Video Annotation**: Output videos with bounding boxes and tracking IDs
- **JSON Output**: Complete detection data exported as JSON for analysis
- **REST API**: FastAPI service for easy integration
- **Scalable Processing**: Frame sampling for efficient processing (default: 15 FPS for enhanced detection)
- **Universal Video Compatibility**: Videos encoded in H.264 format with faststart flag for all players

## 📹 Video Examples

### Input Video (Raw CCTV Footage)
**Location**: `data/videos/2025_12_15_15_24_16_4_dQCiGf.MP4`

The input video contains raw CCTV footage from a fixed-camera poultry farm setup. It shows:
- Multiple birds moving in the frame
- Birds in background with minimal movement
- No annotations or tracking
- Original video quality and resolution

### Output Video (Annotated with Detection & Tracking)
**Location**: `outputs/annotated_bird_detection_improved.mp4` or `outputs/annotated_bird_detection_compatible.mp4`

The output video includes:
- **Green Bounding Boxes**: All detected birds with confidence scores
- **Red Centroids with IDs**: Tracked birds with persistent tracking IDs (e.g., ID:0, ID:1, ID:2)
- **Count Overlay**: Real-time display showing:
  - "Birds Detected": Count of all bounding boxes in current frame
  - "Birds Tracked": Count of birds with stable tracking IDs
  - Timestamp information

### Key Differences

| Feature | Input Video | Output Video |
|---------|------------|-------------|
| **Detection** | None | Green bounding boxes on all birds |
| **Tracking** | None | Red centroids with unique IDs |
| **Counting** | Manual | Automatic count overlay |
| **Confidence** | N/A | Shown on each detection |
| **Metadata** | None | Timestamp and counts displayed |

**Visual Comparison:**
- **Input**: Raw footage with birds moving naturally, including background birds
- **Output**: Same footage with visual annotations showing detection, tracking, and counting in real-time

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster processing)
- 8GB+ RAM recommended
- FFmpeg (for video processing)

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd bird_counting/poultry-monitoring
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Sample Dataset

For testing, download poultry farm videos from Kaggle:
- Search for "poultry farm CCTV" or "chicken farm monitoring"
- Recommended: Videos with fixed camera angle, good lighting
- Place videos in `data/videos/` directory

Example datasets:
- [Poultry Farm Monitoring Dataset](https://www.kaggle.com/search?q=poultry+farm)
- Any fixed-camera livestock monitoring footage

### 5. Create Required Directories

```bash
mkdir -p outputs data/videos
```

## 🏃 Running the Application

### Option 1: Process Video Directly (Standalone Script)

For processing videos with enhanced background bird detection:

```bash
python process_video.py
```

This script uses optimized settings for detecting birds in the background:
- Enhanced frame preprocessing (CLAHE + sharpening)
- Multi-scale detection (original + enhanced frames)
- Lower detection thresholds (conf: 0.005, min_area: 10 pixels)
- Higher frame sampling rate (15 FPS)

**Output:** `outputs/annotated_bird_detection_improved.mp4`

### Option 2: Start the FastAPI Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📡 API Endpoints

### 1. Health Check

**GET** `/health`

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "OK",
  "timestamp": "2025-12-25T12:00:00"
}
```

### 2. Analyze Video

**POST** `/analyze_video`

Upload a video file for bird detection, tracking, and weight estimation.

**Parameters:**
- `file` (file): Video file (MP4, AVI, MOV, MKV)
- `fps_sample` (int, optional): Frames per second to process (default: 2)
- `conf_thresh` (float, optional): Detection confidence threshold (default: 0.25)
- `iou_thresh` (float, optional): IoU threshold for NMS (default: 0.45)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/analyze_video" \
  -F "file=@data/videos/sample.mp4" \
  -F "fps_sample=10" \
  -F "conf_thresh=0.005"
```

**Response:**
See `sample_response.json` for complete response structure.

## 🔧 Implementation Details

### System Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO INPUT PROCESSING                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FRAME EXTRACTION                             │
│  • Read video frame by frame                                    │
│  • Sample frames based on fps_sample (default: 15 FPS)          │
│  • Extract frame metadata (timestamp, frame number)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              FRAME PREPROCESSING & ENHANCEMENT                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Convert BGR → LAB color space                         │   │
│  │ 2. Extract L (luminance) channel                         │   │
│  │ 3. Apply CLAHE (Contrast Limited Adaptive Histogram      │   │
│  │    Equalization) with clipLimit=2.0, tileGridSize=(8,8)  │   │
│  │ 4. Merge enhanced L channel with original A, B channels  │   │
│  │ 5. Convert LAB → BGR                                     │   │
│  │ 6. Apply sharpening kernel:                              │   │
│  │    [[-1, -1, -1],                                        │   │
│  │     [-1,  9, -1],                                        │   │
│  │     [-1, -1, -1]]                                        │   │
│  │ 7. Blend: 70% sharpened + 30% enhanced                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              MULTI-SCALE OBJECT DETECTION                       │
│  ┌──────────────────────┐      ┌──────────────────────┐         │
│  │ Original Frame       │      │ Enhanced Frame       │         │
│  │ Detection            │      │ Detection            │         │
│  │ • YOLOv8 Model       │      │ • YOLOv8 Model       │         │
│  │ • conf_thresh: 0.005 │      │ • conf_thresh: 0.005 │         │
│  │ • iou_thresh: 0.35   │      │ • iou_thresh: 0.35   │         │
│  └──────────┬───────────┘      └──────────┬───────────┘         │
│             │                             │                     │
│             └──────────┬──────────────────┘                     │
│                        │                                        │
│                        ▼                                        │
│              Merge Detection Results                            │
│              • Combine bounding boxes from both detections      │
│              • Remove duplicate detections                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION FILTERING                          │
│  For each detected bounding box:                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Extract coordinates: (x1, y1, x2, y2)                 │   │
│  │ 2. Calculate area: (x2-x1) × (y2-y1)                     │   │
│  │ 3. Filter by minimum area: area >= 10 pixels             │   │
│  │ 4. Extract confidence score                              │   │
│  │ 5. Calculate centroid: ((x1+x2)/2, (y1+y2)/2)            │   │
│  │ 6. Calculate aspect ratio: width/height                  │   │
│  │ 7. Check for duplicates (same bbox coordinates)          │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BIRD TRACKING                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ BirdTracker.update(detections)                           │   │
│  │                                                          │   │
│  │ 1. If no existing tracked objects:                       │   │
│  │    → Register all detections as new birds                │   │
│  │                                                          │   │
│  │ 2. If existing tracked objects:                          │   │
│  │    a. Compute distance matrix:                           │   │
│  │       D[i,j] = ||centroid_i - detection_j||              │   │
│  │    b. Find minimum distance matches                      │   │
│  │    c. Match if distance < 150 pixels                     │   │
│  │    d. Update matched objects:                            │   │
│  │       - Update centroid position                         │   │
│  │       - Reset disappeared counter                        │   │
│  │       - Append area and aspect ratio to history          │   │
│  │    e. Mark unmatched tracked objects as disappeared      │   │
│  │    f. Register unmatched detections as new birds         │   │
│  │                                                          │   │
│  │ 3. Remove objects disappeared > 30 frames                │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WEIGHT ESTIMATION                            │
│  For each tracked bird:                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Calculate average area from tracking history          │   │
│  │ 2. Calculate average aspect ratio from history           │   │
│  │ 3. Normalize area: area_score = (area / avg_area) × 50   │   │
│  │ 4. Calculate aspect score:                               │   │
│  │    aspect_score = 50 × (1 - |1 - aspect_ratio|)          │   │
│  │ 5. Weight index:                                         │   │
│  │    weight_index = 0.7 × area_score + 0.3 × aspect_score  │   │
│  │ 6. Clamp to [0, 100] range                               │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO ANNOTATION                             │
│  For each frame:                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Draw green bounding boxes for all detections          │   │
│  │ 2. Draw confidence scores above boxes                    │   │
│  │ 3. Draw red centroids for tracked birds                  │   │
│  │ 4. Draw tracking IDs (ID:0, ID:1, etc.)                  │   │
│  │ 5. Draw count overlay:                                   │   │
│  │    - Birds Detected: count of bounding boxes             │   │
│  │    - Birds Tracked: count of tracked objects             │   │
│  │    - Timestamp                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT GENERATION                            │
│  ┌──────────────────────┐      ┌──────────────────────┐         │
│  │ Annotated Video      │      │ JSON Results         │         │
│  │ • H.264 MP4 format   │      │ • Detection counts   │         │
│  │ • Faststart enabled  │      │ • Tracking data      │         │
│  │ • Frame-by-frame     │      │ • Weight estimates   │         │
│  │   annotations        │      │ • Metadata           │         │
│  └──────────────────────┘      └──────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Bird Counting Method

**Detection:**
- Uses YOLOv8 (You Only Look Once) for real-time object detection
- Pretrained on COCO dataset, fine-tunable on poultry data
- **Enhanced Detection for Background Birds:**
  - Frame preprocessing with CLAHE (Contrast Limited Adaptive Histogram Equalization) and sharpening
  - Multi-scale detection: runs detection on both original and enhanced frames
  - Very low confidence threshold (0.005) for detecting small/background birds
  - Lower minimum area filtering (10 pixels) to catch smaller birds
  - Duplicate detection removal to prevent counting the same bird twice
- Optimized for detecting birds with minimal movement in background areas

**Tracking:**
- Centroid-based tracking algorithm
- Stable ID assignment across frames
- Handles temporary occlusions (birds disappear for up to 30 frames)
- Distance-based matching to prevent ID switches

**Occlusion Handling:**
- Maximum disappearance threshold (30 frames default)
- Re-identification based on position and size consistency
- ID switch prevention through temporal consistency checks
- Increased distance threshold (150 pixels) for high-resolution videos

**Double-Counting Prevention:**
- Unique tracking IDs maintained throughout video
- Distance threshold for matching (prevents duplicate IDs)
- Disappeared object management (removes stale tracks)
- Duplicate detection removal from multi-scale processing

### Deep Implementation Details

#### 1. Frame Preprocessing Pipeline

**CLAHE (Contrast Limited Adaptive Histogram Equalization):**
- **Purpose**: Enhance local contrast while preventing over-amplification
- **Process**:
  1. Convert BGR → LAB color space (separates luminance from color)
  2. Extract L (luminance) channel
  3. Divide image into 8×8 tiles
  4. Apply histogram equalization to each tile with clipLimit=2.0
  5. Interpolate between tiles to avoid blocking artifacts
  6. Merge enhanced L with original A, B channels
  7. Convert LAB → BGR

**Sharpening Filter:**
- **Kernel**: 3×3 Laplacian-based sharpening kernel
  ```
  [[-1, -1, -1],
   [-1,  9, -1],
   [-1, -1, -1]]
  ```
- **Effect**: Enhances edges and fine details
- **Blending**: 70% sharpened + 30% CLAHE-enhanced to avoid over-sharpening

#### 2. Multi-Scale Detection Strategy

**Dual Detection Approach:**
1. **Original Frame Detection**:
   - Preserves natural lighting and contrast
   - Better for well-lit, prominent birds
   - Captures birds at natural scale

2. **Enhanced Frame Detection**:
   - Improved contrast reveals subtle features
   - Better for background/low-contrast birds
   - May detect birds missed in original

**Merge Strategy:**
- Combine all bounding boxes from both detections
- Remove duplicates using coordinate-based matching
- Use union of detections for maximum coverage

#### 3. YOLOv8 Detection Process

**Model Architecture:**
- **Backbone**: CSPDarknet53 (Cross Stage Partial Darknet)
- **Neck**: PANet (Path Aggregation Network)
- **Head**: Decoupled head with classification and regression branches

**Detection Pipeline:**
1. **Input**: Frame resized to model input size (640×640 default)
2. **Feature Extraction**: Multi-scale feature maps from backbone
3. **Feature Fusion**: PANet combines features at different scales
4. **Prediction**: Head outputs:
   - Bounding box coordinates (x, y, w, h)
   - Confidence scores
   - Class probabilities
5. **Post-processing**:
   - Non-Maximum Suppression (NMS) with IoU threshold
   - Confidence filtering
   - Coordinate scaling back to original frame size

**Parameters:**
- `conf_thresh=0.005`: Very low threshold to catch all possible birds
- `iou_thresh=0.35`: Lower IoU allows more overlapping detections

#### 4. Tracking Algorithm Deep Dive

**Centroid-Based Tracking:**

**Data Structures:**
```python
objects = {id: (cx, cy)}           # Current centroid positions
disappeared = {id: count}            # Frames since last seen
bird_sizes = {id: [area1, area2...]} # Area history
bird_aspects = {id: [ar1, ar2...]}   # Aspect ratio history
```

**Matching Algorithm:**
1. **Distance Matrix Calculation**:
   ```
   D[i,j] = sqrt((cx_i - cx_j)² + (cy_i - cy_j)²)
   ```
   - Euclidean distance between tracked centroid i and detection j

2. **Greedy Matching**:
   - Sort rows by minimum distance
   - Match closest pairs first
   - Skip if distance > 150 pixels (max movement threshold)
   - Mark matched pairs to prevent double-matching

3. **State Updates**:
   - **Matched**: Update centroid, reset disappeared counter, append history
   - **Unmatched Tracked**: Increment disappeared counter
   - **Unmatched Detection**: Register as new bird

4. **Occlusion Handling**:
   - Track disappears for up to 30 frames
   - Re-identify if bird reappears within distance threshold
   - Maintain ID consistency across occlusions

#### 5. Weight Estimation Approach

**Method: Feature-Based Regression**

**Features Used:**
1. **Bounding Box Area**: Pixel area of detected bird
   - Directly correlates with bird size
   - Normalized relative to average bird size in video

2. **Aspect Ratio**: Width-to-height ratio
   - Healthy birds: ~0.7-1.3 (more square)
   - Underweight birds: Often more elongated
   - Overweight birds: More circular/square

3. **Relative Size**: Comparison to average bird size in frame
   - Accounts for camera distance and perspective
   - Normalizes for different video resolutions

4. **Temporal Consistency**: Average area over tracking history
   - Reduces noise from single-frame measurements
   - Uses median/mean of last N detections

**Weight Proxy Calculation:**
```
area_score = min(100, (area / avg_area) × 50)
aspect_score = 50 × (1 - min(1.0, |1.0 - aspect_ratio|))
weight_index = 0.7 × area_score + 0.3 × aspect_score
weight_index = clamp(weight_index, 0, 100)
```

**Scoring Details:**
- `area_score`: Normalized to 0-100, centered around 50 (average size)
- `aspect_score`: Maximum at aspect_ratio=1.0 (square), decreases with deviation
- `weight_index`: Weighted combination, clamped to [0, 100]

**Output Unit: Weight Index (0-100 scale)**
- 0-30: Underweight
- 30-70: Normal weight
- 70-100: Overweight

#### 6. Video Encoding Pipeline

**Frame Writing:**
1. Validate frame format (uint8, BGR, 3 channels)
2. Write to temporary MP4 file using OpenCV
3. Re-encode with FFmpeg for compatibility

**FFmpeg Encoding:**
- **Codec**: H.264 (libx264)
- **Preset**: Medium (balance between speed and compression)
- **CRF**: 22 (constant rate factor, quality setting)
- **Pixel Format**: yuv420p (universal compatibility)
- **Faststart**: Enabled for streaming/quick playback
- **Profile**: High profile, Level 4.0

#### 7. Performance Optimizations

**Frame Sampling:**
- Process every Nth frame based on `fps_sample`
- Reduces computation while maintaining accuracy
- Default: 15 FPS (processes 15 frames per second of video)

**Memory Management:**
- Release frames after processing
- Batch processing for large videos
- Efficient numpy array operations

**GPU Acceleration:**
- YOLOv8 automatically uses GPU if available
- CUDA support for faster inference
- Falls back to CPU if GPU unavailable

### Converting to Actual Weight (Grams)

**Required Calibration Data:**
1. **Ground Truth Weights**: Manual weighing of 50+ birds
2. **Synchronized Video**: Record birds before/after weighing
3. **Camera Calibration**: 
   - Fixed camera height and angle
   - Known reference object for scale (e.g., feeder dimensions)
4. **Environmental Factors**: Lighting conditions, floor markers

**Calibration Process:**
```
weight_grams = α × weight_index + β × camera_height_factor + γ
```

Where α, β, γ are learned from linear regression on calibration data.

**Recommended Calibration Setup:**
- 100+ bird samples with known weights
- Multiple camera angles if using multiple cameras
- Reference markers on floor (e.g., 30cm × 30cm grid)
- Consistent lighting conditions

### Assumptions

1. **Fixed Camera**: Camera position and angle remain constant
2. **Single Plane**: Birds mostly at floor level (minimal height variation)
3. **Adequate Lighting**: Sufficient contrast for detection
4. **Homogeneous Population**: Similar bird types/breeds
5. **Ground Contact**: Weight estimation assumes birds on ground

## 📊 Output Files

### Video Output

**Location**: `outputs/annotated_<timestamp>.mp4` or `outputs/annotated_bird_detection_improved.mp4`

**Format**: H.264 MP4 with faststart flag for universal compatibility

**Features**:
- Green bounding boxes around detected birds
- Red centroids with tracking IDs
- Real-time count overlay
- Confidence scores
  - Timestamp information

### JSON Output

**Location**: `outputs/annotated_bird_detection_result.json`

**Structure**:
```json
{
  "counts": [
    {
      "timestamp": 0.0,
      "frame": 1,
      "count": 45,
      "tracked_count": 42
    }
  ],
  "tracks_sample": [...],
  "weight_estimates": {
    "per_bird": [...],
    "aggregate": {
      "average_weight_index": 65.5,
      "total_birds_tracked": 1855,
      "unit": "index"
    }
  },
  "detection_summary": {
    "total_detections": 2398405,
    "average_per_frame": 432.53,
    "max_per_frame": 598,
    "counting_method": "Bounding boxes detected in each frame",
    "total_birds_tracked": 1855
  },
  "artifacts": {
    "annotated_video": "annotated_bird_detection_improved.mp4",
    "video_path": "outputs/annotated_bird_detection_improved.mp4",
    "video_size_mb": 1562.91
  },
  "metadata": {
    "total_frames": 5545,
    "fps": 19,
    "processing_fps": 15,
    "conf_threshold": 0.005,
    "iou_threshold": 0.35,
    "video_dimensions": "2560x1440",
    "min_area": 10
  }
}
```

**Key Fields:**
- `counts`: Per-frame bird counts with timestamps
- `tracks_sample`: Sample detection data from first frames
- `weight_estimates`: Per-bird and aggregate weight statistics
- `detection_summary`: Overall detection statistics
- `artifacts`: Output file information
- `metadata`: Processing parameters and video information

### Sample Files
- **Sample Response**: `sample_response.json`
  - Example API response with all fields
  - Reference structure for integration

### Visual Comparison

**Before Processing (Input)**:
- Raw video with birds moving naturally
- No annotations or tracking
- Manual counting required

**After Processing (Output)**:
- Green bounding boxes around all detected birds
- Red centroids with unique tracking IDs (ID:0, ID:1, etc.)
- Real-time count display (Birds Detected / Birds Tracked)
- Confidence scores on each detection
- Timestamp information

## 🎨 Customization

### Adjusting Detection Parameters

```python
# In main.py, modify process_video function:
fps_sample = 15        # Frames per second to process (higher = more accurate, slower)
                       # Default: 15 for improved background bird detection
conf_thresh = 0.005    # Lower for more detections, higher for precision
                       # Default: 0.005 for detecting small/background birds
iou_thresh = 0.35      # IoU threshold for Non-Maximum Suppression
                       # Default: 0.35 for improved detection
min_area = 10          # Minimum bounding box area in pixels
                       # Default: 10 for small/background birds
```

**Enhanced Detection Features:**
- Frame preprocessing automatically applied (CLAHE + sharpening)
- Multi-scale detection enabled by default (original + enhanced frames)
- Duplicate detection removal to prevent double-counting

### Changing Tracking Sensitivity

```python
# In BirdTracker class:
max_disappeared = 30  # Frames to keep disappeared objects
distance_threshold = 150  # Max centroid distance for matching
```

### Weight Estimation Tuning

```python
# In estimate_weight_proxy function:
# Adjust weights in formula:
weight_index = 0.7 * area_score + 0.3 * aspect_score
```

## 🔬 Testing

### Run Unit Tests

```bash
pytest tests/
```

### Test API Locally

```bash
# Terminal 1: Start server
python main.py

# Terminal 2: Test endpoints
python test_api.py
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t poultry-monitoring .
```

### Run with Docker Compose

```bash
docker-compose up
```

The API will be available at `http://localhost:8000`

## 📚 Documentation

Additional documentation available in `docs/`:
- **API_GUIDE.md**: Detailed API usage and examples
- **CALIBRATION_GUIDE.md**: Weight calibration procedures
- **DEPLOYMENT_GUIDE.md**: Production deployment instructions

## 🛠️ Troubleshooting

### Video Not Playing
- Use `outputs/annotated_bird_detection_compatible.mp4` for maximum compatibility
- Ensure video player supports H.264 codec
- Check file size (large videos may need more RAM)

### Low Detection Accuracy
- Lower `conf_thresh` to 0.005 for more detections
- Increase `fps_sample` to process more frames
- Lower `min_area` threshold for smaller birds
- Check camera angle and lighting quality

### Performance Issues
- Reduce `fps_sample` for faster processing
- Use GPU acceleration if available
- Process shorter video segments
- Lower video resolution if possible

## 📄 License

This project is developed for the Kuppismart Solutions (Livestockify) ML Engineer assessment.

## 👤 Author

Akash Deep
- GitHub: [akashdeep4303](https://github.com/akashdeep4303)
- Email: akashgsdeep12@gmail.com

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- FastAPI framework
- OpenCV library

---

**Submission Date**: December 20, 2025
**Assessment**: ML/AI Engineer Internship - Kuppismart Solutions


