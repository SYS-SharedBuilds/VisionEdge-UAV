import time
import math
import numpy as np
import threading
try:
    import airsim
except ImportError:
    airsim = None
    print("Warning: airsim package not found. Running in simulation-log mode.")

from config import PRIORITY_MAP, FOLLOW_HEIGHT_PCT, DESIRED_BBOX_HEIGHT, CONTROL_GAIN_P, AIRSIM_IP, ACTIVATE_AUTONOMOUS

class TargetSelector:
    """Selects the best target based on priority and size"""
    def __init__(self, priority_map=None):
        self.priority_map = priority_map or PRIORITY_MAP
    
    def get_priority(self, class_name):
        return self.priority_map.get(class_name, self.priority_map.get('others', 99))

    def select(self, tracks):
        if not tracks:
            return None
            
        # Sort tracks by (priority level ASC, area DESC)
        candidate_tracks = []
        for track in tracks:
            priority = self.get_priority(track['class_name'])
            x1, y1, x2, y2 = track['bbox']
            area = (x2 - x1) * (y2 - y1)
            candidate_tracks.append((priority, -area, track))
            
        candidate_tracks.sort() # Sorts by priority (lower is better), then -area (larger is better)
        return candidate_tracks[0][2]

class AirSimController:
    """Handles interaction with Microsoft AirSim and control signal generation"""
    def __init__(self):
        self.client = None
        self.connected = False
        self.last_control_time = time.time()
        self.autonomous_enabled = True
        
        if airsim:
            # Connect in a background thread to avoid blocking initialization
            threading.Thread(target=self._connect_airsim, daemon=True).start()

    def _connect_airsim(self):
        max_retries = 100
        retry_delay = 3
        
        for attempt in range(max_retries):
            try:
                print(f"Connecting to AirSim on {AIRSIM_IP} (Attempt {attempt+1}/{max_retries})...")
                client = airsim.MultirotorClient(ip=AIRSIM_IP)
                client.confirmConnection()
                if ACTIVATE_AUTONOMOUS:
                    print("AirSim Connected. Enabling API Control...")
                    client.enableApiControl(True)
                    print("Arming Drone...")
                    client.armDisarm(True)
                    print("Taking off...")
                    client.takeoffAsync().join()
                    print("Drone is airborne and ready for autonomous control!")
                else:
                    print("AirSim Connected in Monitoring Mode. API Control Disabled for Manual Flight.")
                    client.enableApiControl(False)
                
                self.client = client
                self.connected = True
                return
            except Exception as e:
                print(f"AirSim Connection failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                
        print("Failed to connect to AirSim after multiple attempts.")

                
    def compute_control(self, target, frame_width, frame_height):
        """
        Calculates control errors based on target bbox
        Returns: (vx, vy, vz, yaw_rate)
        """
        if not target:
            return 0, 0, 0, 0
            
        x1, y1, x2, y2 = target['bbox']
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        bbox_height = y2 - y1
        
        # 1. Horizontal Alignment (Yaw/Side movement)
        # Error normalized to [-1, 1]
        error_x = (target_center_x - (frame_width / 2)) / (frame_width / 2)
        
        # 2. Vertical Alignment (Height/Pitch)
        # Setpoint is slightly below center (FOLLOW_HEIGHT_PCT)
        setpoint_y = frame_height * FOLLOW_HEIGHT_PCT
        error_y = (target_center_y - setpoint_y) / (frame_height / 2)
        
        # 3. Distance Alignment (Forward/Backward)
        # Error based on bbox height vs desired height
        error_z = (DESIRED_BBOX_HEIGHT - bbox_height) / DESIRED_BBOX_HEIGHT
        
        # Simple P-control for velocities (clamped)
        vy = np.clip(error_x * 5.0, -2.0, 2.0)  # Sidebar movement / Yaw rate
        vz = np.clip(-error_y * 3.0, -1.5, 1.5) # Vert velocity
        vx = np.clip(error_z * 4.0, -3.0, 3.0)  # Forward velocity
        
        return vx, vy, vz, error_x * 0.5 # Returning vx, vy, vz, yaw_rate

    def send_commands(self, vx, vy, vz, yaw_rate):
        """Sends velocity commands to AirSim"""
        if self.connected and self.client:
            try:
                # moveByVelocityAsync(vx, vy, vz, duration)
                # In AirSim, x is forward, y is right, z is down
                self.client.moveByVelocityAsync(vx, vy, vz, 0.1, 
                                               airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                               airsim.YawMode(True, yaw_rate * 45))
            except Exception as e:
                print(f"AirSim Command Error: {e}")
        else:
            # Simulation-log mode
            pass

class MAVLinkPacker:
    """Utility to format tracking data into MAVLink-compatible structures"""
    @staticmethod
    def pack_telemetry(target_id, vx, vy, vz, state="TRACKING"):
        # Simulated MAVLink SET_POSITION_TARGET_LOCAL_NED structure
        # time_boot_ms, target_system, target_component, coordinate_frame, type_mask, x, y, z, vx, vy, vz...
        return {
            "msg_id": "SET_POSITION_TARGET_LOCAL_NED",
            "target_id": target_id,
            "vx": round(float(vx), 3),
            "vy": round(float(vy), 3),
            "vz": round(float(vz), 3),
            "state": state,
            "timestamp": time.time()
        }
