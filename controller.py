"""
AirSim Drone Controller
=======================

IOLoop-safe: every blocking AirSim call uses time.sleep() instead of .join()
so Tornado's msgpack-rpc IOLoop never conflicts with Flask threads.

Global keyboard input: pynput captures keys at OS level regardless of which
window has focus, so WASD works even while AirSim Unreal window is active.
"""

import time
import numpy as np
import threading

try:
    import airsim
except ImportError:
    airsim = None
    print("[WARN] airsim package not found – running in log-only mode.")

from config import (PRIORITY_MAP, FOLLOW_HEIGHT_PCT, DESIRED_BBOX_HEIGHT,
                    CONTROL_GAIN_P, AIRSIM_IP, ACTIVATE_AUTONOMOUS)

VEHICLE        = "SimpleFlight"   # must match settings.json exactly
TAKEOFF_WAIT_S = 6.0              # seconds after takeoffAsync before we mark airborne
LAND_WAIT_S    = 7.0              # seconds after landAsync before we mark landed


# ─────────────────────────────────────────────────────────────────────────────
class TargetSelector:
    """Selects the best target from active tracks based on priority and size."""

    def __init__(self, priority_map=None):
        self.priority_map    = priority_map or PRIORITY_MAP
        self.active_category = "all"

    def get_priority(self, class_name):
        return self.priority_map.get(class_name,
                                     self.priority_map.get('others', 99))

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


# ─────────────────────────────────────────────────────────────────────────────
class AirSimController:
    """
    Flight controller for AirSim SimpleFlight multirotor.

    Key design decisions
    --------------------
    * NO .join() anywhere — every async call is fire-and-forget followed by
      time.sleep().  This eliminates all "IOLoop is already running" errors.
    * Global keyboard listener via pynput runs in its own daemon thread and
      captures keys independently of which OS window has focus.
    * Web dashboard /manual_control POST also calls manual_fly() — both
      inputs coexist safely because manual_fly() is thread-safe.
    """

    MANUAL_OVERRIDE_DURATION = 2.0   # s — how long manual overrides auto-tracking
    HOVER_INTERVAL            = 0.25  # s — heartbeat period
    CMD_DURATION              = 0.45  # s — velocity command horizon (> 100 ms poll)

    # Flight speeds
    SPEED    = 3.0   # m/s forward/strafe
    VSPEED   = 2.0   # m/s vertical
    YAW_RATE = 1.0   # normalised → ×45 °/s in moveByVelocityAsync

    def __init__(self):
        self.client      = None
        self.connected   = False
        self.is_airborne = False

        self._manual_override        = False
        self._manual_override_expiry = 0.0
        self._last_cmd_time          = 0.0
        self._takeoff_lock           = threading.Lock()

        if airsim:
            threading.Thread(target=self._connect_airsim, daemon=True,
                             name="airsim-connect").start()

    # ── property ──────────────────────────────────────────────────────────────

    @property
    def manual_override(self) -> bool:
        if self._manual_override:
            if time.time() < self._manual_override_expiry:
                return True
            self._manual_override = False
        return False

    # ── connection + auto-takeoff ─────────────────────────────────────────────

    def _connect_airsim(self):
        """Background thread: connect, arm, take off, then start helpers."""
        import socket
        max_retries = 200   # ~10 minutes of attempts at 3 s each
        retry_delay = 3
        airsim_port = 41451  # default AirSim msgpack-rpc port

        for attempt in range(1, max_retries + 1):
            # ── Pre-check: is the AirSim port open yet? ───────────────────────
            # MultirotorClient() blocks indefinitely if AirSim isn't ready.
            # A quick socket probe avoids that hang and lets the retry loop fire.
            try:
                probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                probe.settimeout(2.0)
                probe.connect((AIRSIM_IP, airsim_port))
                probe.close()
            except (socket.timeout, ConnectionRefusedError, OSError):
                print(f"[AirSim] Port {airsim_port} not ready yet (attempt {attempt}) — retrying in {retry_delay}s")
                time.sleep(retry_delay)
                continue

            # ── Port is open — safe to create the full client ─────────────────
            try:
                print(f"[AirSim] Port open — connecting (attempt {attempt})...")
                # Set a 10 s socket timeout so MultirotorClient() / confirmConnection()
                # fails fast instead of hanging indefinitely if AirSim isn't ready.
                socket.setdefaulttimeout(10)
                try:
                    client = airsim.MultirotorClient(ip=AIRSIM_IP)
                    client.confirmConnection()
                finally:
                    socket.setdefaulttimeout(None)  # restore for the rest of the session
                print("[AirSim] ✓ Connected")

                client.enableApiControl(True, vehicle_name=VEHICLE)
                print("[AirSim] ✓ API control enabled")

                client.armDisarm(True, vehicle_name=VEHICLE)
                print("[AirSim] ✓ Armed")

                # Publish client BEFORE takeoff so the web TAKEOFF button
                # and keyboard listener are available immediately.
                self.client    = client
                self.connected = True

                # takeoffAsync fires the command; we sleep instead of .join()
                # to avoid AirSim's Tornado IOLoop conflict with Flask threads.
                print(f"[AirSim] ↑ Taking off (sleeping {TAKEOFF_WAIT_S}s for altitude)…")
                client.takeoffAsync(vehicle_name=VEHICLE)
                time.sleep(TAKEOFF_WAIT_S)
                self.is_airborne = True
                print("[AirSim] ✓ Airborne")

                # Start support threads
                threading.Thread(target=self._hover_heartbeat, daemon=True,
                                 name="airsim-hover").start()
                threading.Thread(target=self._start_kb_listener, daemon=True,
                                 name="airsim-keyboard").start()

                print("[AirSim] ✓ Ready — WASD/QE/RF/PgUp-PgDn active globally.")
                return

            except Exception as exc:
                print(f"[AirSim] Attempt {attempt} failed: {exc}")
                time.sleep(retry_delay)

        print("[AirSim] ✗ Gave up after too many retries.")

    # ── public takeoff (web TAKEOFF button) ───────────────────────────────────

    def takeoff(self):
        if not (self.connected and self.client):
            return False, "AirSim not connected"
        if self.is_airborne:
            return True, "Already airborne"

        def _bg():
            try:
                self.client.enableApiControl(True, vehicle_name=VEHICLE)
                self.client.armDisarm(True, vehicle_name=VEHICLE)
                print(f"[AirSim] ↑ Web-button takeoff (sleeping {TAKEOFF_WAIT_S}s)…")
                self.client.takeoffAsync(vehicle_name=VEHICLE)
                time.sleep(TAKEOFF_WAIT_S)
                self.is_airborne = True
                print("[AirSim] ✓ Airborne (web button)")
            except Exception as e:
                print(f"[AirSim] takeoff() error: {e}")

        threading.Thread(target=_bg, daemon=True).start()
        return True, "Takeoff initiated"

    # ── RTB / land ────────────────────────────────────────────────────────────

    def land(self):
        if not (self.connected and self.client):
            return False, "AirSim not connected"
        try:
            print(f"[AirSim] ↓ Landing (sleeping {LAND_WAIT_S}s)…")
            self.client.landAsync(vehicle_name=VEHICLE)
            time.sleep(LAND_WAIT_S)
            self.client.armDisarm(False, vehicle_name=VEHICLE)
            self.is_airborne = False
            print("[AirSim] ✓ Landed")
            return True, "Landed successfully"
        except Exception as e:
            return False, str(e)

    # ── hover heartbeat ───────────────────────────────────────────────────────

    def _hover_heartbeat(self):
        """
        Sends vx=vy=vz=0 when idle so SimpleFlight holds position.
        Skips if a velocity command was sent recently (< 1.5× interval ago).
        """
        while self.connected:
            try:
                idle = (time.time() - self._last_cmd_time) > self.HOVER_INTERVAL * 1.5
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

    # ── global keyboard listener ──────────────────────────────────────────────

    def _start_kb_listener(self):
        """
        Captures WASD/QE/RF/Arrows/PgUp-PgDn/Space at OS level via pynput.
        Works even when the AirSim Unreal Engine window has focus.

        Key map (same as web dashboard):
          W/↑   forward      S/↓   backward
          D/→   strafe R     A/←   strafe L
          R/PgUp ascend      F/PgDn descend
          Q     yaw CCW      E     yaw CW
          Space takeoff
        """
        try:
            from pynput import keyboard as kb
            print("[KB] pynput listener starting…")
        except ImportError:
            print("[KB] pynput not found — global keys unavailable.")
            print("[KB] Run:  pip install pynput")
            return

        pressed   = set()
        loop_evt  = threading.Event()
        lock      = threading.Lock()

        def _tok(key):
            try:
                return key.char.lower() if key.char else key
            except AttributeError:
                return key

        def _build():
            vx = vy = vz = yr = 0.0
            if kb.Key.up    in pressed or 'w' in pressed: vx += self.SPEED
            if kb.Key.down  in pressed or 's' in pressed: vx -= self.SPEED
            if kb.Key.right in pressed or 'd' in pressed: vy += self.SPEED
            if kb.Key.left  in pressed or 'a' in pressed: vy -= self.SPEED
            if kb.Key.page_up   in pressed or 'r' in pressed: vz -= self.VSPEED
            if kb.Key.page_down in pressed or 'f' in pressed: vz += self.VSPEED
            if 'q' in pressed: yr -= self.YAW_RATE
            if 'e' in pressed: yr += self.YAW_RATE
            return vx, vy, vz, yr

        def _cmd_loop():
            while loop_evt.is_set():
                vx, vy, vz, yr = _build()
                if any([vx, vy, vz, yr]):
                    self.manual_fly(vx, vy, vz, yr)
                time.sleep(0.10)

        _cmd_thread = None

        def on_press(key):
            nonlocal _cmd_thread
            tok = _tok(key)
            if tok == ' ':
                if not self.is_airborne:
                    threading.Thread(target=self.takeoff, daemon=True).start()
                return
            with lock:
                pressed.add(tok)
                if not loop_evt.is_set():
                    loop_evt.set()
                    _cmd_thread = threading.Thread(target=_cmd_loop, daemon=True)
                    _cmd_thread.start()

        def on_release(key):
            tok = _tok(key)
            if tok == ' ':
                return
            with lock:
                pressed.discard(tok)
                if not pressed:
                    loop_evt.clear()
                    self.manual_fly(0, 0, 0, 0)   # stop / hover

        print("[KB] ✓ Global keyboard listener active.")
        with kb.Listener(on_press=on_press, on_release=on_release,
                         suppress=False) as listener:
            listener.join()

    # ── manual flight (web /manual_control + global KB) ──────────────────────

    def manual_fly(self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0):
        """
        Send a velocity command and set the manual override flag.
        Called at 10 Hz by both the web-dashboard key handler and the global
        keyboard listener.

        NOTE: Does NOT auto-takeoff. The drone must be airborne first (use the
        TAKEOFF button or press Space). This prevents the 6-second blocking
        auto-takeoff from conflicting with velocity commands.
        """
        # Refresh override so autonomous tracking backs off
        self._manual_override        = True
        self._manual_override_expiry = time.time() + self.MANUAL_OVERRIDE_DURATION

        if not (self.connected and self.client):
            return

        # Silently ignore velocity commands until airborne
        if not self.is_airborne:
            return

        try:
            self._last_cmd_time = time.time()
            self.client.moveByVelocityAsync(
                vx, vy, vz, self.CMD_DURATION,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate * 45),
                vehicle_name=VEHICLE
            )
            if vx or vy or vz or yaw_rate:
                print(f"[AirSim] ▶ vx={vx:+.1f} vy={vy:+.1f} "
                      f"vz={vz:+.1f} yaw={yaw_rate:+.1f}")
        except Exception as e:
            print(f"[AirSim] manual_fly error: {e}")

    # ── autonomous tracking ───────────────────────────────────────────────────

    def compute_control(self, target, frame_width, frame_height):
        """PID-like control from target bounding box. Returns zeros during override."""
        if self.manual_override or not target:
            return 0, 0, 0, 0
        x1, y1, x2, y2 = target['bbox']
        ex  = ((x1+x2)/2 - frame_width /2) / (frame_width /2)
        ey  = ((y1+y2)/2 - frame_height*FOLLOW_HEIGHT_PCT) / (frame_height/2)
        ez  = (DESIRED_BBOX_HEIGHT - (y2-y1)) / DESIRED_BBOX_HEIGHT
        vy  = float(np.clip(ex * 5.0,  -2.0, 2.0))
        vz  = float(np.clip(-ey * 3.0, -1.5, 1.5))
        vx  = float(np.clip(ez * 4.0,  -3.0, 3.0))
        yr  = ex * 0.5
        return vx, vy, vz, yr

    def send_commands(self, vx, vy, vz, yaw_rate):
        """Autonomous velocity commands; skipped during manual override."""
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


# ─────────────────────────────────────────────────────────────────────────────
class MAVLinkPacker:
    """Formats telemetry into MAVLink-compatible dicts for downstream consumers."""

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
