"""
Configuration file for Dual-Video RF-DETR Object Detection and Tracking

Modify these settings to customize the application behavior.
"""

import os

# Server Settings
HOST = '0.0.0.0'
PORT = int(os.environ.get('PORT', 5000))
DEBUG = False

# Streaming Settings
JPEG_QUALITY = 85
STREAM_FPS = 30

# Video Settings
VIDEO_SOURCES = ["4791734-hd_1920_1080_30fps.mp4", "5021555-hd_1920_1080_30fps.mp4"]
CAMERA_WIDTH = 640           # Target processing width (auto-scaled for performance)
CAMERA_HEIGHT = 360           # Target processing height
CAMERA_FPS = 30               # Target FPS

# RF-DETR Model Settings
RFDETR_MODEL_SIZE = 'RFDETRBase'  # RFDETRBase or RFDETRLarge
CONFIDENCE_THRESHOLD = 0.5    # Minimum confidence for detections (0.0 to 1.0)

# Deep SORT Tracking Settings
# Tracking Parameters
MAX_AGE = 50                  # Frames to keep a lost track
N_INIT = 3                    # Consecutive frames to confirm a track
DATASET_MODE = 'COCO'      # 'VISDRONE', 'UAVDT', or 'COCO'
TRACKER_TYPE = 'bytetrack'    # 'bytetrack' or 'botsort'
REID_ENABLED = True           # Enable re-identification after occlusion

# VisDrone Class Mapping
VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van', 
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

# Target Priority Mapping (Lower integer = Higher priority)
PRIORITY_MAP = {
    'pedestrian': 1,
    'people': 1,
    'car': 2,
    'van': 2,
    'bus': 3,
    'truck': 4,
    'bicycle': 5,
    'motor': 5,
    'others': 99
}

FOLLOW_HEIGHT_PCT  = 0.6
DESIRED_BBOX_HEIGHT = 150

# HUD Settings
SHOW_PREDICTIONS = True       # Show Kalman predicted boxes
SHOW_TELEMETRY = True         # Show HUD dashboard
EXPLAIN_MODE = True           # Add reasoning text to HUD
HUD_COLOR_THEME = (0, 255, 0) # Green dashboard

# Military HUD Layers
HUD_MIL_STYLE = True          # Use operator-grade minimalist style
HUD_SHOW_CROSSHAIR = True     # Center drone aim point
HUD_SHOW_TRAJECTORY = True    # 2s motion prediction vectors
HUD_SHOW_ALERTS = True        # Locking/Loss notifications
HUD_SHOW_CORRECTIONS = True   # Directional arrows for control
HUD_SHOW_TAPES = True         # Altitude/Speed tapes on edges

# MIL-STD Colors (BGR)
COLOR_LOCKED = (0, 255, 0)    # Green
COLOR_SEARCHING = (0, 191, 255)# Amber
COLOR_LOST = (0, 0, 255)      # Red
COLOR_HUD_BASE = (0, 255, 0)  # Primary HUD color
