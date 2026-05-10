"""
Configuration file for YOLOv8 Live Object Detection and Tracking

Modify these settings to customize the application behavior.
"""

# Server Settings
HOST = '0.0.0.0'
PORT = 5000
DEBUG = False

# Streaming Settings
JPEG_QUALITY = 85
STREAM_FPS = 30

# Video/Camera Settings
USE_WEBCAM = False            # Set to False to use a video file
VIDEO_SOURCE = 'drone_video.mp4' # Path to 4K or UAV surveillance video
CAMERA_INDEX = 0              # Camera index (used only if USE_WEBCAM is True)
CAMERA_WIDTH = 1280           # Target processing width (auto-scaled for performance)
CAMERA_HEIGHT = 720           # Target processing height
CAMERA_FPS = 30               # Target FPS

# YOLOv8 Model Settings
YOLO_MODEL = 'yolov8n.pt'     # Model size: yolov8n.pt (fastest) to yolov8x.pt (most accurate)
CONFIDENCE_THRESHOLD = 0.5    # Minimum confidence for detections (0.0 to 1.0)

# Deep SORT Tracking Settings
# Tracking Parameters
MAX_AGE = 50                  # Frames to keep a lost track
N_INIT = 3                    # Consecutive frames to confirm a track
DATASET_MODE = 'VISDRONE'      # 'VISDRONE', 'UAVDT', or 'COCO'
TRACKER_TYPE = 'bytetrack'    # 'bytetrack' or 'botsort'
ACTIVATE_AUTONOMOUS = False   # Enable autonomous control signal generation
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

# AirSim Control Settings
UNREAL_EXECUTABLE_PATH = r"C:\Users\Admin\Desktop\AirSimNH\WindowsNoEditor\AirSimNH.exe"   # e.g., "C:\\Path\\To\\Environment.exe"
AUTO_START_SIMULATION = True  # Automatically launch the environment on start
USE_AIRSIM_CAMERA = True      # Use the drone's simulated camera instead of webcam
AIRSIM_CAMERA_NAME = "0"      # "0" for front-facing, "3" for bottom-facing
AIRSIM_IP = '127.0.0.1'       # AirSim host IP
AIRSIM_PORT = 41451           # AirSim API port
FOLLOW_HEIGHT_PCT = 0.6       # Vertical setpoint (below center, 0.6 from top)
DESIRED_BBOX_HEIGHT = 150     # Target bbox height for distance control (pixels)
CONTROL_GAIN_P = 0.5          # P-gain for velocity control

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
