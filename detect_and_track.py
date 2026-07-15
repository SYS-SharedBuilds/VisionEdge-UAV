"""
Object Detection and Tracking Module using YOLOv8 and Deep SORT

This module handles:
1. YOLOv8 object detection
2. Deep SORT multi-object tracking
3. Frame processing and annotation
4. Webcam capture and management
"""

import os
# Fix OpenMP runtime conflict - must be set before importing other libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import threading
import time
from collections import defaultdict
import math

# Global state for tracking toggle
tracking_enabled = threading.Event()
tracking_enabled.set()
active_stream = "A"

def set_active_stream(stream_id):
    global active_stream
    active_stream = stream_id

class TargetSelector:
    """Selects the best target from active tracks based on priority and size."""
    def __init__(self, priority_map=None):
        self.priority_map = priority_map or PRIORITY_MAP
        self.active_category = "all"

    def get_priority(self, class_name):
        return self.priority_map.get(class_name, self.priority_map.get('others', 99))

    def select(self, tracks):
        if not tracks:
            return None
        filtered = []
        for t in tracks:
            cls = t['class_name'].lower()
            cat = self.active_category
            if cat and cat != "all":
                if cat == 'person'  and cls not in ['pedestrian','people','person']:
                    continue
                if cat == 'vehicle' and cls not in ['car','van','truck','bus']:
                    continue
                if cat == 'bicycle' and cls not in ['bicycle','motorcycle','motor','tricycle']:
                    continue
            filtered.append(t)
        if not filtered:
            return None
        ranked = [(self.get_priority(t['class_name']),
                   -((t['bbox'][2]-t['bbox'][0])*(t['bbox'][3]-t['bbox'][1])),
                   t) for t in filtered]
        ranked.sort()
        return ranked[0][2]

class AdaptiveDetector:
    def __init__(self, model_size='RFDETRBase', confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        try:
            from rfdetr import RFDETR
            self.model = RFDETR(model_size)
            self.is_rfdetr_pkg = True
        except ImportError:
            print("[WARN] rfdetr package not found, falling back to Ultralytics YOLOv8n for real-time performance")
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            self.is_rfdetr_pkg = False

    def track(self, frame, persist=True, tracker="bytetrack.yaml"):
        if not self.is_rfdetr_pkg:
            return self.model.track(frame, persist=persist, conf=self.confidence_threshold, tracker=tracker, verbose=False)
        else:
            # Fallback wrapper if using the real rfdetr package but ultralytics tracker is expected downstream
            # In a real implementation this would format results to match ultralytics or use SimpleTracker.
            # Here we assume the rfdetr package outputs something similar or we fallback to ultralytics.
            try:
                # Try assuming ultralytics-like interface for the sake of demo
                return self.model.track(frame, persist=persist, conf=self.confidence_threshold, tracker=tracker, verbose=False)
            except AttributeError:
                # Mock a return structure if it lacks .track
                class MockBox:
                    def __init__(self):
                        self.id = None
                        self.xyxy = None
                        self.conf = None
                        self.cls = None
                class MockResult:
                    def __init__(self):
                        self.boxes = MockBox()
                return [MockResult()]
try:
    from config import *
except ImportError:
    # Default values if config.py is not available
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    YOLO_MODEL = 'yolov8n.pt'
    CONFIDENCE_THRESHOLD = 0.5
    MAX_AGE = 50
    N_INIT = 3
    NMS_MAX_OVERLAP = 1.0
    MAX_COSINE_DISTANCE = 0.4
    TRAJECTORY_LENGTH = 30
    BBOX_THICKNESS = 2
    TEXT_SCALE = 0.6
    TEXT_THICKNESS = 2

class MetricsSystem:
    """System to track and calculate real-time performance metrics"""
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.prediction_errors = []
        self.track_longevity = defaultdict(int)
        self.last_update = time.time()

    def update(self, frame_processed=True):
        if frame_processed:
            self.frame_count += 1
        
        current_time = time.time()
        elapsed = current_time - self.last_update
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_update = current_time

    def log_prediction_error(self, error):
        self.prediction_errors.append(error)
        if len(self.prediction_errors) > 100:
            self.prediction_errors.pop(0)

    def get_avg_error(self):
        if not self.prediction_errors: return 0
        return sum(self.prediction_errors) / len(self.prediction_errors)

def draw_dashed_rect(img, pt1, pt2, color, thickness=1, style='dotted', gap=10):
    """Draws a dashed rectangle using line segments"""
    x1, y1 = pt1
    x2, y2 = pt2
    # Top
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness)
    # Bottom
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness)
    # Left
    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, thickness)
    # Right
    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, thickness)

class HUDPainter:
    """Military-grade HUD rendering engine for drone operators"""
    def __init__(self, theme_color=COLOR_HUD_BASE):
        self.theme_color = theme_color
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def draw_dashboard(self, frame, metrics, system_status="IDLE", primary_target=None, drone_connected=False):
        h, w = frame.shape[:2]
        color = self.theme_color if system_status != "ALERT" else COLOR_LOST
        
        # 1. Crosshair (Center)
        if HUD_SHOW_CROSSHAIR:
            cx, cy = w // 2, h // 2
            gap = 15
            length = 40
            cv2.line(frame, (cx - length, cy), (cx - gap, cy), color, 1, self.line_type)
            cv2.line(frame, (cx + gap, cy), (cx + length, cy), color, 1, self.line_type)
            cv2.line(frame, (cx, cy - length), (cx, cy - gap), color, 1, self.line_type)
            cv2.line(frame, (cx, cy + gap), (cx, cy + length), color, 1, self.line_type)
            cv2.circle(frame, (cx, cy), 2, color, -1)

        # 2. Telemetry Panel (Situational Awareness)
        if HUD_SHOW_TAPES:
            # Simple tapes/blocks
            cv2.rectangle(frame, (20, 20), (160, 80), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, 20), (160, 80), color, 1)
            cv2.putText(frame, f"FPS: {metrics.fps:.1f}", (30, 40), self.font, 0.4, color, 1)
            
            # Alt/Speed - Use real values only if connected
            alt_str = f"ALT: N/A"
            spd_str = f"SPD: N/A"
            if drone_connected:
                alt_str = f"ALT: 12.4m" # placeholder for actual drone data
                spd_str = f"SPD: 4.2m/s"
            
            cv2.putText(frame, alt_str, (30, 55), self.font, 0.4, color, 1)
            cv2.putText(frame, spd_str, (30, 70), self.font, 0.4, color, 1)
            
            # Status Banner
            status_color = COLOR_LOCKED if system_status == "LOCKED" else COLOR_SEARCHING
            cv2.rectangle(frame, (w//2-70, 20), (w//2+70, 50), (0, 0, 0), -1)
            cv2.rectangle(frame, (w//2-70, 20), (w//2+70, 50), status_color, 1)
            cv2.putText(frame, system_status, (w//2-50, 40), self.font, 0.5, status_color, 2)

        # 3. Control Corrections
        if HUD_SHOW_CORRECTIONS and primary_target:
            tx, ty = (primary_target['bbox'][0] + primary_target['bbox'][2]) / 2, \
                     (primary_target['bbox'][1] + primary_target['bbox'][3]) / 2
            cx, cy = w // 2, h // 2
            err_x = tx - cx
            err_y = ty - (h * FOLLOW_HEIGHT_PCT)
            if abs(err_x) > 20 or abs(err_y) > 20:
                # Draw arrow towards target
                end_pt = (int(cx + np.clip(err_x, -50, 50)), int(cy + np.clip(err_y, -50, 50)))
                cv2.arrowedLine(frame, (cx, cy), end_pt, COLOR_SEARCHING, 2, tipLength=0.3)

    def draw_target(self, frame, track_data, is_primary=False):
        x1, y1, x2, y2 = map(int, track_data['bbox'])
        tid = track_data['id']
        conf = track_data['confidence']
        cls = track_data['class_name']
        
        color = COLOR_LOCKED if is_primary else (150, 150, 150)
        thickness = 2 if is_primary else 1
        
        # Operator cornered box
        length = 20
        # Corners
        cv2.line(frame, (x1, y1), (x1+length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1+length), color, thickness)
        cv2.line(frame, (x2, y1), (x2-length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1+length), color, thickness)
        cv2.line(frame, (x1, y2), (x1+length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2-length), color, thickness)
        cv2.line(frame, (x2, y2), (x2-length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2-length), color, thickness)
        
        # Label
        label = f"TRK {tid} [{cls.upper()}]" if is_primary else f"#{tid}"
        cv2.putText(frame, label, (x1, y1 - 10), self.font, 0.4, color, 1)
        if is_primary:
            cv2.putText(frame, f"CONF: {conf:.2f}", (x1, y2 + 15), self.font, 0.4, color, 1)

    def draw_prediction(self, frame, pred_bbox, track_id):
        x1, y1, x2, y2 = map(int, pred_bbox)
        draw_dashed_rect(frame, (x1, y1), (x2, y2), COLOR_SEARCHING, 1, gap=5)
        cv2.putText(frame, f"LOST-REACQ #{track_id}", (x1, y1 - 10), self.font, 0.4, COLOR_SEARCHING, 1)

    def draw_trajectory(self, frame, tinfo, color):
        if len(tinfo.history) < 2: return
        points = np.array(tinfo.history, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], False, color, 1)
        
        # Velocity Vector
        if HUD_SHOW_TRAJECTORY and tinfo.velocity:
            vx, vy = tinfo.velocity
            last_pt = tinfo.history[-1]
            future_pt = (int(last_pt[0] + vx * 10), int(last_pt[1] + vy * 10))
            cv2.arrowedLine(frame, (int(last_pt[0]), int(last_pt[1])), future_pt, color, 1, tipLength=0.2)

class SimpleTracker:
    """
    A simple object tracker using IoU (Intersection over Union) matching
    Compatible with Python 3.13.5
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_count = 0
    
    def update(self, detections):
        """
        Update tracks with new detections
        detections: list of [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        # Predict new locations of existing tracks
        for track in self.tracks:
            track['age'] += 1
            track['hits_since_update'] += 1
        
        # Match detections to existing tracks
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        if len(self.tracks) > 0 and len(detections) > 0:
            # Calculate IoU matrix
            iou_matrix = self._calculate_iou_matrix(detections)
            
            # Find matches using Hungarian algorithm (simplified)
            matches = self._associate_detections_to_tracks(iou_matrix)
            
            # Update matched tracks
            for match in matches:
                det_idx, track_idx = match
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)
                
                track = self.tracks[track_idx]
                det = detections[det_idx]
                
                # Update track with detection
                track['bbox'] = det[:4]
                track['confidence'] = det[4]
                track['class_id'] = det[5]
                track['class_name'] = det[6]
                track['hits_since_update'] = 0
                track['hit_streak'] += 1
                matched_tracks.append(track)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_track = {
                'id': self.track_id_count,
                'bbox': det[:4],
                'confidence': det[4],
                'class_id': det[5],
                'class_name': det[6],
                'age': 0,
                'hit_streak': 1,
                'hits_since_update': 0
            }
            self.tracks.append(new_track)
            self.track_id_count += 1
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks 
                      if track['hits_since_update'] < self.max_age]
        
        # Return confirmed tracks
        return [track for track in self.tracks 
                if track['hit_streak'] >= self.min_hits or track['age'] < self.min_hits]
    
    def _calculate_iou_matrix(self, detections):
        """Calculate IoU between detections and existing tracks"""
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        
        for d, det in enumerate(detections):
            for t, track in enumerate(self.tracks):
                iou_matrix[d, t] = self._calculate_iou(det[:4], track['bbox'])
        
        return iou_matrix
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _associate_detections_to_tracks(self, iou_matrix):
        """Simple greedy matching based on IoU"""
        matches = []
        used_detections = set()
        used_tracks = set()
        
        # Sort by IoU score (highest first)
        indices = np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape)
        
        for det_idx, track_idx in zip(indices[0], indices[1]):
            if (det_idx not in used_detections and 
                track_idx not in used_tracks and 
                iou_matrix[det_idx, track_idx] > self.iou_threshold):
                
                matches.append((det_idx, track_idx))
                used_detections.add(det_idx)
                used_tracks.add(track_idx)
        
        return matches

class TrackInfo:
    """Stores persistent state for a track for behavior analysis and Re-ID"""
    def __init__(self, track_id, class_name):
        self.track_id = track_id
        self.class_name = class_name
        self.history = []
        self.state = "DETECTED" # DETECTED, OCCLUDED, LOST
        self.last_seen = time.time()
        self.velocity = (0, 0)
        self.predicted_pos = None

    def update(self, center):
        if len(self.history) > 0:
            last_center = self.history[-1]
            self.velocity = (center[0] - last_center[0], center[1] - last_center[1])
        self.history.append(center)
        if len(self.history) > 30: self.history.pop(0)
        self.state = "DETECTED"
        self.last_seen = time.time()
        self.predicted_pos = (center[0] + self.velocity[0], center[1] + self.velocity[1])

class ObjectDetectorTracker:
    def __init__(self, video_source, stream_id="A"):
        """
        Initialize the object detector and tracker
        
        Args:
            video_source (str): Path to video file
            stream_id (str): Identifier for this stream
        """
        self.video_source = video_source
        self.stream_id = stream_id
        
        # Use config values as defaults
        confidence_threshold = CONFIDENCE_THRESHOLD
        
        # Initialize Dual Backbone / RF-DETR model
        print(f"[{self.stream_id}] Loading RF-DETR model...")
        self.model = AdaptiveDetector(model_size=RFDETR_MODEL_SIZE, confidence_threshold=confidence_threshold)
        self.confidence_threshold = confidence_threshold
        
        # Initialize Advanced Components
        self.metrics = MetricsSystem()
        self.hud = HUDPainter(theme_color=HUD_COLOR_THEME)
        self.selector = TargetSelector()
        
        # Track management
        self.persistent_tracks = {} # track_id -> TrackInfo
        self.primary_target_id = None
        self.frame_lock = threading.Lock()
        self.current_frame = None
        
        # Load Dataset-Specific Classes
        if DATASET_MODE == 'VISDRONE':
            self.class_names = VISDRONE_CLASSES
            print(f"Loaded {len(self.class_names)} VisDrone classes.")
        elif DATASET_MODE == 'UAVDT':
            self.class_names = ['car', 'truck', 'bus']
            print(f"Loaded {len(self.class_names)} UAVDT vehicle classes.")
        else:
            # COCO class names for YOLOv8
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
    
    def generate_color(self, track_id):
        """Generate a unique color for each track ID"""
        if track_id not in self.track_colors:
            # Generate a random color based on track ID
            np.random.seed(track_id)
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            self.track_colors[track_id] = color
        return self.track_colors[track_id]
    
    def start_video(self):
        """
        Start video source
        """
        print(f"[{self.stream_id}] Starting video acquisition from: {self.video_source}")
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            print(f"[{self.stream_id}] ERROR: Could not open source {self.video_source}")
            self.current_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            cv2.putText(self.current_frame, "SIGNAL LOST: NO VIDEO SOURCE", (CAMERA_WIDTH//2-200, CAMERA_HEIGHT//2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.is_running = True
            return
        
        self.is_running = True
        print(f"[{self.stream_id}] Video source initialized successfully!")
    
    def stop_video(self):
        """Stop video capture and release resources"""
        print(f"[{self.stream_id}] Stopping video...")
        self.is_running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        print(f"[{self.stream_id}] Video stopped successfully!")
    
    def detect_and_track(self, frame):
        """
        Perform object detection and tracking on a single frame
        """
        h, w = frame.shape[:2]
        
        global active_stream
        if not tracking_enabled.is_set() or self.stream_id != active_stream:
            # If tracking is disabled, return raw frame without running RF-DETR
            return frame.copy()

        # Run RF-DETR tracking
        results = self.model.track(frame, persist=True, tracker=TRACKER_TYPE + ".yaml")
        
        annotated_frame = frame.copy()
        current_tracks_data = []
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, conf, cls in zip(boxes, ids, confs, clss):
                class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                track_data = {
                    'id': track_id,
                    'bbox': box,
                    'confidence': conf,
                    'class_name': class_name
                }
                current_tracks_data.append(track_data)
                
                # Update persistent state
                if track_id not in self.persistent_tracks:
                    self.persistent_tracks[track_id] = TrackInfo(track_id, class_name)
                
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                self.persistent_tracks[track_id].update(center)
                
                # Use modular HUD components
                self.hud.draw_target(annotated_frame, track_data, is_primary=False)
                if HUD_SHOW_TRAJECTORY:
                    self.hud.draw_trajectory(annotated_frame, self.persistent_tracks[track_id], (150, 150, 150))
        
        # Handle Occlusions (Tracks lost this frame but still in memory)
        current_ids = [t['id'] for t in current_tracks_data]
        for tid, tinfo in self.persistent_tracks.items():
            if tid not in current_ids:
                if time.time() - tinfo.last_seen < 3.0: # Keep alive for 3 seconds (Extended for Military Re-id)
                    tinfo.state = "OCCLUDED"
                    if tinfo.predicted_pos and SHOW_PREDICTIONS:
                        self.hud.draw_prediction(annotated_frame, 
                                               (tinfo.predicted_pos[0]-25, tinfo.predicted_pos[1]-40, 
                                                tinfo.predicted_pos[0]+25, tinfo.predicted_pos[1]+40), tid)
                else:
                    tinfo.state = "LOST"

        # Adaptive Target Selection
        self.primary_target_id = self.selector.select(current_tracks_data)
        primary_track = None
        if self.primary_target_id:
            tid = self.primary_target_id['id']
            primary_track = self.primary_target_id
            # Redraw primary target with operator highlight
            self.hud.draw_target(annotated_frame, primary_track, is_primary=True)
        
        self.metrics.update()
        hud_status = "LOCKED" if primary_track else "SEARCHING"
        if any(t.state == "OCCLUDED" for t in self.persistent_tracks.values()):
            hud_status = "REACQUIRING"
            
        self.hud.draw_dashboard(annotated_frame, self.metrics, 
                              system_status=hud_status, primary_target=primary_track,
                              drone_connected=False)
        
        return annotated_frame
    
    def get_frame(self):
        """
        Get the current processed frame (thread-safe)
        
        Returns:
            processed frame or None if no frame available
        """
        if not hasattr(self, 'frame_lock'):
            self.frame_lock = threading.Lock()
        with self.frame_lock:
            if not hasattr(self, 'current_frame'):
                return None
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def process_video_stream(self):
        """
        Main processing loop for continuous surveillance
        """
        print(f"[{self.stream_id}] Entering surveillance processing mode...")

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_time = 1.0 / fps

        while self.is_running:
            if self.cap is None:
                time.sleep(0.1)
                continue
            
            start_t = time.time()
            
            ret, frame = self.cap.read()
            
            # Auto-loop for video files
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            
            if not ret:
                time.sleep(0.1)
                continue
            
            try:
                # 4K Resolution Downscaling for Performance
                if frame.shape[1] > CAMERA_WIDTH:
                    frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_AREA)
                
                # Process frame
                processed_frame = self.detect_and_track(frame)
                
                with self.frame_lock:
                    self.current_frame = processed_frame
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Processing Error: {e}")
                with self.frame_lock:
                    # Serve the raw frame anyway if tracking crashes to keep the feed alive
                    if 'frame' in locals() and frame is not None:
                        self.current_frame = frame
            
            # Real-time video playback synchronization
            elapsed = time.time() - start_t
            frames_to_skip = int(elapsed / frame_time)
            
            if frames_to_skip > 0:
                for _ in range(frames_to_skip):
                    self.cap.grab()
            else:
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        print("Video processing loop ended")
    
    def start_processing_thread(self):
        """Start the video processing in a separate thread"""
        processing_thread = threading.Thread(target=self.process_video_stream, daemon=True)
        processing_thread.start()
        return processing_thread


# Global detectors dictionary (will be initialized by Flask app)
detectors = {}

def initialize_detectors(video_sources):
    """Initialize the global detector instances for multiple streams"""
    global detectors
    stream_ids = ["A", "B", "C", "D"]
    for i, source in enumerate(video_sources):
        sid = stream_ids[i]
        if sid not in detectors:
            detectors[sid] = ObjectDetectorTracker(video_source=source, stream_id=sid)
    return detectors

def get_detector(stream_id):
    """Get the global detector instance for a specific stream"""
    return detectors.get(stream_id)
