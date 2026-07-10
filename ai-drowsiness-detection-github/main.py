"""
main.py
-------
Real-time AI Drowsiness Detection using a laptop/USB webcam.

Pipeline (see report / README for the diagram):
    Camera Frame
        -> MediaPipe Face Mesh (468 landmarks)
        -> Eye Aspect Ratio (EAR)          -> closed-eye detection
        -> Mouth Aspect Ratio (MAR)        -> yawn detection
        -> Head Pose (solvePnP)            -> head-nod / look-away detection
        -> DrowsinessStateMachine          -> AWAKE / DROWSY decision
        -> On-screen overlay + audible alert

Run:
    pip install -r requirements.txt
    python main.py

Press 'q' to quit. Look at the camera for the first ~1 second so the
system can calibrate your normal ("awake") head pitch.
"""

import time
import cv2
import mediapipe as mp

from utils import (
    LEFT_EYE, RIGHT_EYE,
    eye_aspect_ratio, mouth_aspect_ratio, estimate_head_pose,
    DrowsinessStateMachine,
)

CALIBRATION_FRAMES = 30


def beep():
    """Best-effort audible alert; falls back silently if unavailable."""
    try:
        import winsound
        winsound.Beep(1000, 400)
    except ImportError:
        print("\a", end="", flush=True)  # terminal bell on Linux/Mac


def draw_hud(frame, ear, mar, pitch, state, reasons, fps):
    h, w = frame.shape[:2]
    color = (0, 0, 255) if state == "DROWSY" else (0, 200, 0)

    cv2.rectangle(frame, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"MAR: {mar:.2f}", (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Pitch: {pitch:.1f} deg", (290, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    status_text = f"STATUS: {state}" + (f" ({', '.join(reasons)})" if reasons else "")
    cv2.putText(frame, status_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6 if state == "DROWSY" else 2)


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions / index.")
        return

    sm = DrowsinessStateMachine()
    frame_count = 0
    prev_time = time.time()
    last_beep = 0

    print("Starting... look at the camera normally to calibrate.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(landmarks, w, h)
            pitch, yaw, roll = estimate_head_pose(landmarks, w, h)

            frame_count += 1
            if frame_count <= CALIBRATION_FRAMES:
                sm.calibrate_baseline(pitch)
                cv2.putText(frame, "Calibrating...", (w // 2 - 80, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            state, reasons = sm.update(ear, mar, pitch, now=now)
            draw_hud(frame, ear, mar, pitch, state, reasons, fps)

            if state == "DROWSY" and now - last_beep > 2:
                beep()
                last_beep = now
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("AI Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
