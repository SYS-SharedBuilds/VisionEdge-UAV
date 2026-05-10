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
                print(f"Launching Unreal Engine Simulation from: {UNREAL_EXECUTABLE_PATH}")
                subprocess.Popen([UNREAL_EXECUTABLE_PATH])
                print("Waiting 10 seconds for simulation to boot...")
                time.sleep(10)
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
