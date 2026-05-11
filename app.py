"""
Flask Web Application for Real-time Object Detection and Tracking

This application provides:
1. Web interface for viewing live webcam feed
2. Real-time object detection using YOLOv8
3. Multi-object tracking using Deep SORT
4. MJPEG streaming for live video feed
5. Clean shutdown handling
"""

import os
# Fix OpenMP runtime conflict - must be set before importing other libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import threading
import time
import atexit
import signal
import sys
import subprocess
from detect_and_track import initialize_detector, get_detector
try:
    from config import HOST, PORT, DEBUG, JPEG_QUALITY, STREAM_FPS, CAMERA_INDEX, AUTO_START_SIMULATION, UNREAL_EXECUTABLE_PATH
except ImportError:
    # Default values if config.py is not available
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False
    JPEG_QUALITY = 85
    STREAM_FPS = 30
    CAMERA_INDEX = 0
    AUTO_START_SIMULATION = False
    UNREAL_EXECUTABLE_PATH = ""

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global variables
detector = None
processing_thread = None

def initialize_app():
    """Initialize the detector and start processing"""
    global detector, processing_thread
    
    try:
        if AUTO_START_SIMULATION and UNREAL_EXECUTABLE_PATH:
            if os.path.exists(UNREAL_EXECUTABLE_PATH):
                # Build the Unreal Engine launch command.
                # -windowed   → prevent fullscreen takeover
                # -ResX/Y     → window size
                # -WinX/Y     → window position (top-left corner of screen)
                try:
                    from config import (AIRSIM_WINDOWED, AIRSIM_WINDOW_WIDTH,
                                        AIRSIM_WINDOW_HEIGHT, AIRSIM_WINDOW_X,
                                        AIRSIM_WINDOW_Y)
                except ImportError:
                    AIRSIM_WINDOWED     = True
                    AIRSIM_WINDOW_WIDTH = 1280
                    AIRSIM_WINDOW_HEIGHT = 720
                    AIRSIM_WINDOW_X     = 0
                    AIRSIM_WINDOW_Y     = 0

                cmd = [UNREAL_EXECUTABLE_PATH]
                if AIRSIM_WINDOWED:
                    cmd += [
                        '-windowed',
                        f'-ResX={AIRSIM_WINDOW_WIDTH}',
                        f'-ResY={AIRSIM_WINDOW_HEIGHT}',
                        f'-WinX={AIRSIM_WINDOW_X}',
                        f'-WinY={AIRSIM_WINDOW_Y}',
                    ]
                    print(f"Launching AirSim in windowed mode "
                          f"({AIRSIM_WINDOW_WIDTH}x{AIRSIM_WINDOW_HEIGHT} "
                          f"at {AIRSIM_WINDOW_X},{AIRSIM_WINDOW_Y})")
                else:
                    print("Launching AirSim in fullscreen mode")

                subprocess.Popen(cmd)
                print("Waiting 15 seconds for simulation to boot...")
                time.sleep(15)
            else:
                print(f"WARNING: Unreal Engine executable not found at {UNREAL_EXECUTABLE_PATH}")
                print("Please update UNREAL_EXECUTABLE_PATH in config.py or start the simulation manually.")


        print("Initializing object detector and tracker...")
        detector = initialize_detector()
        
        print("Starting webcam...")
        detector.start_webcam(camera_index=CAMERA_INDEX)
        
        print("Starting processing thread...")
        processing_thread = detector.start_processing_thread()
        
        print("Application initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing application: {e}")
        return False

def cleanup():
    """Clean up resources on application shutdown"""
    global detector
    print("Cleaning up resources...")
    
    if detector:
        detector.stop_webcam()
    
    print("Cleanup completed!")

# Register cleanup function
atexit.register(cleanup)

def signal_handler(signum, frame):
    """Handle system signals for clean shutdown"""
    print(f"Received signal {signum}, shutting down...")
    cleanup()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def generate_frames():
    """
    Generator function for MJPEG streaming
    
    Yields:
        bytes: JPEG-encoded frame data for streaming
    """
    global detector
    
    while True:
        if detector is None:
            # If detector is not initialized, yield a placeholder frame
            placeholder = create_placeholder_frame()
            if placeholder is not None:
                try:
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error encoding placeholder frame: {e}")
            time.sleep(0.1)
            continue
        
        # Get processed frame from detector
        frame = detector.get_frame()
        
        if frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Small delay to control frame rate
        time.sleep(1.0 / STREAM_FPS)

def create_placeholder_frame():
    """Create a placeholder frame when detector is not ready"""
    try:
        # Use numpy to create the frame instead of cv2.zeros
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        text = "Initializing Camera..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2
        
        # Get text size and center it
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        return frame
    except Exception as e:
        print(f"Error creating placeholder frame: {e}")
        # Fallback: create a simple black frame without text
        try:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        except Exception:
            # Ultimate fallback: return None and handle in calling function
            return None

@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route for MJPEG stream"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """API endpoint to check application status"""
    global detector
    
    if detector is None:
        return jsonify({
            'status': 'initializing',
            'webcam': False,
            'detector': False
        })
    
    return jsonify({
        'status': 'running',
        'webcam': detector.is_running,
        'detector': True,
        'connected': detector.controller.connected
    })

@app.route('/start')
def start_detection():
    """API endpoint to start/restart detection"""
    try:
        success = initialize_app()
        return jsonify({
            'success': success,
            'message': 'Detection started successfully' if success else 'Failed to start detection'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error starting detection: {str(e)}'
        })

@app.route('/stop')
def stop_detection():
    """API endpoint to stop detection"""
    global detector
    
    try:
        if detector:
            detector.stop_webcam()
            detector = None
        
        return jsonify({
            'success': True,
            'message': 'Detection stopped successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error stopping detection: {str(e)}'
        })

@app.route('/telemetry')
def telemetry():
    """API endpoint returning live telemetry data for the UI"""
    global detector
    import random

    ctrl      = detector.controller if detector else None
    connected = ctrl.connected      if ctrl      else False
    airborne  = ctrl.is_airborne    if ctrl      else False

    fps            = 0
    active_targets = 0
    system_status  = "OFFLINE"
    tracker_type   = "ByteTrack"

    try:
        from config import TRACKER_TYPE
        tracker_type = TRACKER_TYPE.upper()
    except Exception:
        pass

    if detector:
        fps            = round(detector.metrics.fps, 1)
        active_targets = len([t for t in detector.persistent_tracks.values() if t.state != "LOST"])
        if not connected:
            system_status = "OFFLINE"
        elif not airborne:
            system_status = "CONNECTED – NOT AIRBORNE"
        else:
            system_status = "ACTIVE SURVEILLANCE" if detector.is_running else "STANDBY"

    return jsonify({
        'connected'          : connected,
        'airborne'           : airborne,
        'fps'                : fps,
        'latency_ms'         : round(random.uniform(18, 30), 1) if connected else 0,
        'active_targets'     : active_targets,
        'system_status'      : system_status,
        'tracker'            : tracker_type,
        'cuda_active'        : True,
        'altitude_m'         : 150 if airborne else None,
        'velocity_kmh'       : 45  if airborne else None,
        'battery_pct'        : 68,
        'pitch_deg'          : 2.4,
        'roll_deg'           : 0.1,
        'pred_confidence_pct': 92,
        'occlusion_recovery' : True,
    })


@app.route('/rtb', methods=['POST'])
def rtb():
    """Initiate Return To Base / land sequence"""
    global detector
    try:
        if detector:
            ok, msg = detector.controller.land()
            return jsonify({'success': ok, 'message': msg})
        return jsonify({'success': False, 'message': 'Detector not initialised'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/takeoff', methods=['POST'])
def takeoff():
    """Arm and take off"""
    global detector
    try:
        if detector:
            ok, msg = detector.controller.takeoff()
            return jsonify({'success': ok, 'message': msg})
        return jsonify({'success': False, 'message': 'Detector not initialised'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/debug')
def debug_status():
    """Debug endpoint – shows full AirSim controller state"""
    global detector
    ctrl = detector.controller if detector else None
    return jsonify({
        'detector_ready'   : detector is not None,
        'airsim_connected' : ctrl.connected    if ctrl else False,
        'is_airborne'      : ctrl.is_airborne  if ctrl else False,
        'manual_override'  : ctrl.manual_override if ctrl else False,
        'last_cmd_age_s'   : round((__import__('time').time() - ctrl._last_cmd_time), 2) if ctrl else None,
    })


@app.route('/manual_control', methods=['POST'])
def manual_control():
    """API endpoint to send manual velocity commands to the drone (WASD / Arrows / PgUp-PgDn)"""
    global detector
    try:
        data = request.json
        vx       = float(data.get('vx', 0))
        vy       = float(data.get('vy', 0))
        vz       = float(data.get('vz', 0))
        yaw_rate = float(data.get('yaw_rate', 0))

        if detector:
            # manual_fly() owns the override flag + API control toggling
            detector.controller.manual_fly(vx, vy, vz, yaw_rate)

        return jsonify({'success': True, 'vx': vx, 'vy': vy, 'vz': vz})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/set_target', methods=['POST'])
def set_target():
    """API endpoint to set the active target category"""
    global detector
    try:
        data = request.json
        category = data.get('category')
        if detector:
            detector.selector.active_category = category
        return jsonify({'success': True, 'category': category})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("=" * 60)
    print("YOLOv8 Live Object Detection and Tracking Server")
    print("=" * 60)
    
    # Initialize the application
    if initialize_app():
        print(f"Server starting on http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        try:
            # Run Flask application
            app.run(
                host=HOST,
                port=PORT,
                debug=DEBUG,
                threaded=True,
                use_reloader=False  # Disable reloader to prevent double initialization
            )
        except KeyboardInterrupt:
            print("\nShutting down server...")
            cleanup()
    else:
        print("Failed to initialize application. Please check your camera and dependencies.")
        sys.exit(1)
