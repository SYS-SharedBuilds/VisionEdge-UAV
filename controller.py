import time
import numpy as np
import threading

try:
    import airsim
except ImportError:
    airsim = None
    print("Warning: airsim package not found. Running in simulation-log mode.")

from config import PRIORITY_MAP, FOLLOW_HEIGHT_PCT, DESIRED_BBOX_HEIGHT, CONTROL_GAIN_P, AIRSIM_IP, ACTIVATE_AUTONOMOUS

# ── Vehicle name must match settings.json exactly ─────────────────────────────
VEHICLE = "SimpleFlight"


class TargetSelector:
    """Selects the best target based on priority and size"""
    def __init__(self, priority_map=None):
        self.priority_map = priority_map or PRIORITY_MAP
        self.active_category = "all"

    def get_priority(self, class_name):
        return self.priority_map.get(class_name, self.priority_map.get('others', 99))

    def select(self, tracks):
        if not tracks:
            return None

        filtered_tracks = []
        for track in tracks:
            cls_name = track['class_name'].lower()
            if self.active_category and self.active_category != "all":
                if self.active_category == 'person' and cls_name not in ['pedestrian', 'people', 'person']:
                    continue
                if self.active_category == 'vehicle' and cls_name not in ['car', 'van', 'truck', 'bus']:
                    continue
                if self.active_category == 'bicycle' and cls_name not in ['bicycle', 'motorcycle', 'motor', 'tricycle']:
                    continue
            filtered_tracks.append(track)

        if not filtered_tracks:
            return None

        candidate_tracks = []
        for track in filtered_tracks:
            priority = self.get_priority(track['class_name'])
            x1, y1, x2, y2 = track['bbox']
            area = (x2 - x1) * (y2 - y1)
            candidate_tracks.append((priority, -area, track))

        candidate_tracks.sort()
        return candidate_tracks[0][2]


class AirSimController:
    """
    Drone flight controller.

    Design principle
    ────────────────
    API control is enabled ONCE after takeoff and kept ON for the entire
    session.  There is NO toggling of enableApiControl during normal operation.
    A background hover-heartbeat thread keeps the drone stable whenever
    neither manual nor autonomous commands are being sent.

    Priority order (highest → lowest):
      1. manual_fly()          – WASD / arrow / PgUp-PgDn from web dashboard
      2. send_commands()       – autonomous AI tracking
      3. _hover_heartbeat()    – background hover to prevent drift
    """

    # How many seconds after the last key-press before auto-tracking re-engages
    MANUAL_OVERRIDE_DURATION = 1.5   # generous window; reduced after keys released
    HOVER_INTERVAL            = 0.25  # seconds between hover heartbeat pulses

    def __init__(self):
        self.client        = None
        self.connected     = False
        self.is_airborne   = False

        # Manual override flag – set by manual_fly(), auto-expires
        self._manual_override          = False
        self._manual_override_expiry   = 0.0
        self._last_cmd_time            = 0.0   # time of most recent velocity send

        if airsim:
            threading.Thread(target=self._connect_airsim, daemon=True).start()

    # ── Public property ───────────────────────────────────────────────────────

    @property
    def manual_override(self) -> bool:
        if self._manual_override:
            if time.time() < self._manual_override_expiry:
                return True
            self._manual_override = False
        return False

    # ── Connection & takeoff ──────────────────────────────────────────────────

    def _connect_airsim(self):
        max_retries = 100
        retry_delay = 3

        for attempt in range(max_retries):
            try:
                print(f"[AirSim] Connecting on {AIRSIM_IP} (attempt {attempt+1}/{max_retries})...")
                client = airsim.MultirotorClient(ip=AIRSIM_IP)
                client.confirmConnection()
                print("[AirSim] Connection confirmed.")

                # ── Always enable API control first ──────────────────────────
                client.enableApiControl(True, vehicle_name=VEHICLE)
                print("[AirSim] API control enabled.")

                # ── Arm ──────────────────────────────────────────────────────
                client.armDisarm(True, vehicle_name=VEHICLE)
                print("[AirSim] Drone armed.")

                # ── Takeoff only if ACTIVATE_AUTONOMOUS ──────────────────────
                if ACTIVATE_AUTONOMOUS:
                    print("[AirSim] Taking off...")
                    client.takeoffAsync(vehicle_name=VEHICLE).join()
                    print("[AirSim] Airborne!")
                    self.is_airborne = True
                else:
                    print("[AirSim] Monitoring mode – skipping takeoff.")

                # Store client AFTER everything succeeds
                self.client    = client
                self.connected = True

                # Start the hover heartbeat so the drone stays stable
                threading.Thread(target=self._hover_heartbeat, daemon=True).start()
                print("[AirSim] Ready. WASD / Arrow / PgUp-PgDn controls are live.")
                return

            except Exception as e:
                print(f"[AirSim] Connection attempt failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)

        print("[AirSim] Giving up after too many failed attempts.")

    def takeoff(self):
        """Public takeoff – can be called from the /takeoff Flask route."""
        if not (self.connected and self.client):
            return False, "AirSim not connected"
        if self.is_airborne:
            return True, "Already airborne"
        try:
            self.client.enableApiControl(True, vehicle_name=VEHICLE)
            self.client.armDisarm(True, vehicle_name=VEHICLE)
            self.client.takeoffAsync(vehicle_name=VEHICLE).join()
            self.is_airborne = True
            print("[AirSim] Takeoff complete via /takeoff endpoint.")
            return True, "Takeoff successful"
        except Exception as e:
            return False, str(e)

    def land(self):
        """RTB – land and disarm."""
        if not (self.connected and self.client):
            return False, "AirSim not connected"
        try:
            self.client.landAsync(vehicle_name=VEHICLE).join()
            self.client.armDisarm(False, vehicle_name=VEHICLE)
            self.is_airborne = False
            return True, "Landed successfully"
        except Exception as e:
            return False, str(e)

    # ── Hover heartbeat ───────────────────────────────────────────────────────

    def _hover_heartbeat(self):
        """
        Sends a gentle zero-velocity command at regular intervals so SimpleFlight
        keeps the drone stationary when no other commands are in flight.
        Without this, SimpleFlight may slowly drift or fall after a velocity
        command expires.
        """
        while self.connected:
            try:
                now = time.time()
                # Only hover if no command was sent recently
                idle = now - self._last_cmd_time > self.HOVER_INTERVAL * 1.5
                if idle and self.is_airborne and self.client:
                    self.client.moveByVelocityAsync(
                        0, 0, 0, self.HOVER_INTERVAL * 2,
                        airsim.DrivetrainType.MaxDegreeOfFreedom,
                        airsim.YawMode(False, 0),
                        vehicle_name=VEHICLE
                    )
            except Exception:
                pass
            time.sleep(self.HOVER_INTERVAL)

    # ── Manual flight (WASD / arrows / PgUp-PgDn) ────────────────────────────

    def manual_fly(self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0):
        """
        Called by the /manual_control Flask route ~10 Hz while keys are held.
        Sets the manual override flag so autonomous tracking backs off.
        """
        # Refresh override window
        self._manual_override         = True
        self._manual_override_expiry  = time.time() + self.MANUAL_OVERRIDE_DURATION

        if not (self.connected and self.client):
            print("[AirSim] manual_fly: not connected – command dropped")
            return

        if not self.is_airborne:
            print("[AirSim] manual_fly: drone not airborne – ignoring velocity cmd")
            return

        try:
            duration = 0.3   # seconds – longer than the 100 ms call interval for smooth flight
            self._last_cmd_time = time.time()
            self.client.moveByVelocityAsync(
                vx, vy, vz, duration,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate * 45),
                vehicle_name=VEHICLE
            )
            if vx != 0 or vy != 0 or vz != 0:
                print(f"[AirSim] manual_fly: vx={vx:.1f} vy={vy:.1f} vz={vz:.1f}")
        except Exception as e:
            print(f"[AirSim] manual_fly error: {e}")

    # ── Autonomous tracking ───────────────────────────────────────────────────

    def compute_control(self, target, frame_width, frame_height):
        """Returns (vx, vy, vz, yaw_rate). Returns zeros if manual override active."""
        if self.manual_override:
            return 0, 0, 0, 0

        if not target:
            return 0, 0, 0, 0

        x1, y1, x2, y2 = target['bbox']
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        bbox_height     = y2 - y1

        error_x = (target_center_x - frame_width  / 2) / (frame_width  / 2)
        error_y = (target_center_y - frame_height * FOLLOW_HEIGHT_PCT) / (frame_height / 2)
        error_z = (DESIRED_BBOX_HEIGHT - bbox_height) / DESIRED_BBOX_HEIGHT

        vy       = np.clip(error_x * 5.0,  -2.0, 2.0)
        vz       = np.clip(-error_y * 3.0, -1.5, 1.5)
        vx       = np.clip(error_z * 4.0,  -3.0, 3.0)
        yaw_rate = error_x * 0.5

        return vx, vy, vz, yaw_rate

    def send_commands(self, vx, vy, vz, yaw_rate):
        """Sends autonomous velocity commands. Skipped during manual override."""
        if self.manual_override:
            return
        if not (self.connected and self.client and self.is_airborne):
            return
        try:
            self._last_cmd_time = time.time()
            self.client.moveByVelocityAsync(
                vx, vy, vz, 0.15,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate * 45),
                vehicle_name=VEHICLE
            )
        except Exception as e:
            print(f"[AirSim] send_commands error: {e}")


class MAVLinkPacker:
    """Utility to format tracking data into MAVLink-compatible structures"""
    @staticmethod
    def pack_telemetry(target_id, vx, vy, vz, state="TRACKING"):
        return {
            "msg_id"    : "SET_POSITION_TARGET_LOCAL_NED",
            "target_id" : target_id,
            "vx"        : round(float(vx), 3),
            "vy"        : round(float(vy), 3),
            "vz"        : round(float(vz), 3),
            "state"     : state,
            "timestamp" : time.time()
        }
