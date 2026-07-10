"""
utils.py
--------
Core math + logic for the AI Drowsiness Detection system.

Contains:
  - eye_aspect_ratio()      -> detects closed eyes
  - mouth_aspect_ratio()    -> detects yawning
  - estimate_head_pose()    -> detects head nodding / tilting away from road
  - DrowsinessStateMachine  -> turns raw per-frame signals into a stable alert

All landmark indices refer to MediaPipe's 468-point Face Mesh.
"""

import time
import numpy as np
import cv2

# ----------------------------------------------------------------------
# Landmark index groups (MediaPipe Face Mesh topology)
# ----------------------------------------------------------------------
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 13, 14, 78, 308]  # left, right, top, bottom, in-left, in-right

# 6 stable points used for solvePnP head-pose estimation
HEAD_POSE_POINTS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}

# Generic 3D face model (mm) used for solvePnP — does not need to be
# person-specific, only proportionally correct.
MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -63.6, -12.5),      # chin
    (-43.3, 32.7, -26.0),     # left eye corner
    (43.3, 32.7, -26.0),      # right eye corner
    (-28.9, -28.9, -24.1),    # left mouth corner
    (28.9, -28.9, -24.1),     # right mouth corner
], dtype=np.float64)


def _euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """
    Classic EAR formula (Soukupova & Cech, 2016):
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Low EAR (< ~0.21) sustained over several frames => eyes closed.
    """
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks, img_w, img_h):
    """
    MAR = vertical mouth opening / horizontal mouth width.
    High MAR (> ~0.6) sustained => yawning.
    """
    left, right, top, bottom, in_left, in_right = [
        (landmarks[i].x * img_w, landmarks[i].y * img_h) for i in MOUTH
    ]
    vertical = _euclidean(top, bottom)
    horizontal = _euclidean(left, right)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def estimate_head_pose(landmarks, img_w, img_h):
    """
    Solves for the head's pitch/yaw/roll using 6 facial landmarks + solvePnP.
    Returns (pitch, yaw, roll) in degrees.
    A large pitch drop (chin-to-chest) or yaw (looking away) for a sustained
    period flags a distraction / drowsy "head nod" event.
    """
    image_points = np.array([
        (landmarks[idx].x * img_w, landmarks[idx].y * img_h)
        for idx in HEAD_POSE_POINTS.values()
    ], dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_3D_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, np.zeros((3, 1))))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [float(a) for a in euler_angles]
    return pitch, yaw, roll


class DrowsinessStateMachine:
    """
    Converts noisy, per-frame EAR / MAR / head-pose readings into a single,
    stable alert decision. This is the "why it's good: clear flow" part —
    every rule is simple and explainable:

        eyes closed for N consecutive frames  -> DROWSY (eyes)
        yawns detected >= Y times in a rolling window -> DROWSY (yawn)
        head pitch drop beyond threshold for N frames  -> DROWSY (head nod)

    Any one of the three triggers a DROWSY state; all three clearing
    resets to AWAKE.
    """

    def __init__(
        self,
        ear_threshold=0.21,
        ear_consec_frames=15,          # ~0.5s at 30fps
        mar_threshold=0.6,
        yawn_window_seconds=60,
        yawn_count_trigger=2,
        pitch_drop_threshold=15.0,     # degrees, relative to calibrated baseline
        head_consec_frames=20,
    ):
        self.ear_threshold = ear_threshold
        self.ear_consec_frames = ear_consec_frames
        self.mar_threshold = mar_threshold
        self.yawn_window_seconds = yawn_window_seconds
        self.yawn_count_trigger = yawn_count_trigger
        self.pitch_drop_threshold = pitch_drop_threshold
        self.head_consec_frames = head_consec_frames

        self.eye_closed_counter = 0
        self.head_down_counter = 0
        self.yawn_events = []       # timestamps of yawn starts
        self._was_yawning = False
        self.baseline_pitch = None
        self.alert_state = "AWAKE"
        self.alert_reason = None

    def calibrate_baseline(self, pitch):
        """Call for the first ~30 frames while the driver looks at the road."""
        if self.baseline_pitch is None:
            self.baseline_pitch = pitch
        else:
            self.baseline_pitch = 0.9 * self.baseline_pitch + 0.1 * pitch

    def update(self, ear, mar, pitch, now=None):
        now = now or time.time()
        reasons = []

        # --- Eyes ---
        if ear < self.ear_threshold:
            self.eye_closed_counter += 1
        else:
            self.eye_closed_counter = 0
        if self.eye_closed_counter >= self.ear_consec_frames:
            reasons.append("eyes_closed")

        # --- Yawning ---
        is_yawning = mar > self.mar_threshold
        if is_yawning and not self._was_yawning:
            self.yawn_events.append(now)
        self._was_yawning = is_yawning
        self.yawn_events = [t for t in self.yawn_events if now - t <= self.yawn_window_seconds]
        if len(self.yawn_events) >= self.yawn_count_trigger:
            reasons.append("frequent_yawning")

        # --- Head position ---
        if self.baseline_pitch is not None:
            if (self.baseline_pitch - pitch) > self.pitch_drop_threshold:
                self.head_down_counter += 1
            else:
                self.head_down_counter = 0
            if self.head_down_counter >= self.head_consec_frames:
                reasons.append("head_nod")

        self.alert_reason = reasons
        self.alert_state = "DROWSY" if reasons else "AWAKE"
        return self.alert_state, reasons
