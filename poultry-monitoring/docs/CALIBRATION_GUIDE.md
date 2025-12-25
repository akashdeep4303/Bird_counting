# Weight Calibration Guide

## Converting Weight Index to Actual Grams

The current system outputs a **Weight Index** (0-100 scale) based on visual features. To convert this to actual weight in grams, calibration is required.

## Required Data for Calibration

### 1. Ground Truth Weight Dataset

**Minimum Requirements:**
- 100+ birds with known weights
- Weight range covering your flock (e.g., 500g - 3000g)
- Measurements taken within 24 hours of video recording

**Data Collection Process:**
```
For each bird:
1. Weigh bird on calibrated scale (±5g accuracy)
2. Record weight in grams
3. Mark bird with non-toxic, visible marker (colored leg band)
4. Video record bird for 30-60 seconds
5. Note: timestamp, bird_id, weight_grams
```

**Sample Data Format:**
```csv
bird_id,weight_grams,video_file,timestamp,age_days,breed
001,1250,calibration_day1.mp4,00:15:23,35,Broiler
002,1180,calibration_day1.mp4,00:16:45,35,Broiler
003,1420,calibration_day1.mp4,00:18:12,38,Broiler
```

### 2. Camera Calibration

**Fixed Reference Markers:**
Place known-size objects in the camera view:
- Floor grid: 30cm × 30cm squares
- Reference pole: 1 meter height marker
- Calibration plate: 40cm × 40cm checkerboard

**Camera Parameters to Record:**
- Camera height from floor (cm)
- Camera angle (degrees from vertical)
- Field of view (degrees)
- Resolution (pixels)
- Lens type (standard, wide-angle, etc.)

### 3. Environmental Factors

Record consistent conditions:
- Time of day (for lighting)
- Weather conditions (if applicable)
- Floor type (concrete, bedding, etc.)
- Crowd density (birds per m²)

## Calibration Process

### Step 1: Extract Features from Calibration Videos

Run the API on calibration videos to get weight indices:

```bash
curl -X POST "http://localhost:8000/analyze_video" \
  -F "file=@calibration_day1.mp4" \
  -F "fps_sample=5" > calibration_results.json
```

### Step 2: Match Weight Indices to Ground Truth

Create a dataset pairing:
- `weight_index` (from API)
- `weight_grams` (from scale)
- `pixel_area` (bounding box area)
- `camera_height` (from setup)

**Example Data:**
```python
calibration_data = [
    {'weight_index': 65.8, 'weight_grams': 1250, 'pixel_area': 3500, 'camera_height': 250},
    {'weight_index': 58.2, 'weight_grams': 1180, 'pixel_area': 3200, 'camera_height': 250},
    {'weight_index': 72.1, 'weight_grams': 1420, 'pixel_area': 3850, 'camera_height': 250},
    # ... 100+ samples
]
```

### Step 3: Train Regression Model

Use linear regression or polynomial regression:

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Prepare features
X = np.array([[d['weight_index'], d['pixel_area'], d['camera_height']] 
              for d in calibration_data])
y = np.array([d['weight_grams'] for d in calibration_data])

# Option 1: Linear Regression
model = LinearRegression()
model.fit(X, y)

# Option 2: Polynomial Regression (better accuracy)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# Save coefficients
coefficients = {
    'alpha': model.coef_[0],  # weight_index coefficient
    'beta': model.coef_[1],   # pixel_area coefficient
    'gamma': model.coef_[2],  # camera_height coefficient
    'intercept': model.intercept_
}
```

### Step 4: Validation

Split data 80/20 for training/validation:

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f} grams")
print(f"R² Score: {r2:.3f}")

# Target: MAE < 100g, R² > 0.85
```

## Integration into API

### Update `estimate_weight_proxy` function:

```python
def estimate_weight_grams(weight_index, pixel_area, camera_height, coefficients):
    """
    Convert weight index to actual grams using calibration
    
    Args:
        weight_index: Weight proxy (0-100)
        pixel_area: Bounding box area in pixels
        camera_height: Camera height in cm
        coefficients: Dict with alpha, beta, gamma, intercept
    
    Returns:
        weight_grams: Estimated weight in grams
    """
    weight_grams = (
        coefficients['alpha'] * weight_index +
        coefficients['beta'] * pixel_area +
        coefficients['gamma'] * camera_height +
        coefficients['intercept']
    )
    
    return max(0, weight_grams)  # Ensure non-negative

# Load calibration coefficients (from JSON file)
with open('calibration_coefficients.json', 'r') as f:
    calibration = json.load(f)

# Use in weight estimation
weight_grams = estimate_weight_grams(
    weight_index=65.8,
    pixel_area=3500,
    camera_height=250,
    coefficients=calibration
)
```

## Expected Accuracy

With proper calibration:
- **Mean Absolute Error**: 50-100 grams
- **R² Score**: 0.85-0.95
- **Confidence Interval**: ±150 grams (95% confidence)

### Factors Affecting Accuracy:
1. **Camera Quality**: Higher resolution = better accuracy
2. **Calibration Dataset Size**: More samples = better fit
3. **Bird Posture**: Standing vs. sitting affects apparent size
4. **Occlusion**: Partially hidden birds reduce accuracy
5. **Feather Condition**: Ruffled feathers increase apparent size

## Recommended Calibration Schedule

- **Initial Calibration**: 100+ birds, comprehensive dataset
- **Monthly Recalibration**: 20-30 birds to adjust for growth patterns
- **Camera Change**: Full recalibration required
- **Seasonal Adjustment**: Consider temperature effects on bird behavior

## Alternative: Depth-Based Weight Estimation

For higher accuracy, consider adding depth cameras:

**Intel RealSense or similar:**
- Provides actual 3D measurements
- Eliminates perspective distortion
- Achieves MAE < 50 grams
- Requires additional hardware (~$200-$500)

## Troubleshooting

**Issue: High MAE (>150g)**
- Solution: Increase calibration dataset size
- Check camera stability (ensure fixed position)
- Verify ground truth weights are accurate

**Issue: Poor R² Score (<0.75)**
- Solution: Add polynomial features
- Consider non-linear regression (Random Forest, XGBoost)
- Check for outliers in calibration data

**Issue: Inconsistent Predictions**
- Solution: Ensure consistent lighting conditions
- Verify camera settings haven't changed
- Recalibrate with recent data

## Contact for Calibration Support

For assistance with calibration or to discuss advanced weight estimation methods, please contact the development team.