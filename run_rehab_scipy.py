import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from Spine_Data import SpineCoordinateAnalyzer
from Pose_Compare_Scipy import PoseComparatorSciPy

def start_rehab_scipy_mode():
    print("🚀 [SciPy 십자 Frame 고정밀 재활 모드] 초기화 시작...")

    MODEL_PATH = 'pose_landmarker_lite.task'
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        print("경량화 포즈 모델(Lite) 다운로드 중...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, MODEL_PATH)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    spine_analyzer = SpineCoordinateAnalyzer()
    scipy_comparator = PoseComparatorSciPy("ref_pose.json")

    PURE_BODY_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
    ]
    KEY_LANDMARK_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    print("🎥 Camo Studio / 카메라 탐색 중...")
    cap = None
    search_indices = [1, 2, 3, 0]
    for idx in search_indices:
        temp_cap = cv2.VideoCapture(idx)
        if temp_cap.isOpened():
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            temp_cap.set(cv2.CAP_PROP_FPS, 60)

            success = False
            for _ in range(5):
                ret, frame = temp_cap.read()
                if ret and frame is not None and frame.size > 0:
                    success = True
                    break
                time.sleep(0.05)

            if success:
                cap = temp_cap
                print(f"✅ [SciPy 재활 모드] 카메라 연결 성공! (인덱스: {idx})")
                break
            temp_cap.release()

    if cap is None:
        print("❌ 카메라 연결 실패: Camo Studio 앱 연결 상태를 확인해 주세요.")
        return

    smooth_landmarks = {}
    ALPHA = 0.85
    start_time = time.time()
    prev_frame_time = 0

    print("🏥 [SciPy 십자 Frame 재활 모드 구동 중...] 종료하려면 'q'를 누르세요.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue

        curr_frame_time = time.time()
        fps = 1 / (curr_frame_time - prev_frame_time) if (curr_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = curr_frame_time

        frame_timestamp_ms = int((curr_frame_time - start_time) * 1000)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        if detection_result.pose_landmarks:
            h, w, _ = frame.shape
            raw_landmarks = detection_result.pose_landmarks[0]
            curr_pts = {}

            for idx in KEY_LANDMARK_INDICES:
                lm = raw_landmarks[idx]
                if hasattr(lm, 'visibility') and lm.visibility < 0.35:
                    continue
                cx, cy = int(lm.x * w), int(lm.y * h)
                if idx in smooth_landmarks:
                    px, py = smooth_landmarks[idx]
                    sm_x = int(ALPHA * cx + (1 - ALPHA) * px)
                    sm_y = int(ALPHA * cy + (1 - ALPHA) * py)
                    curr_pts[idx] = (sm_x, sm_y)
                else:
                    curr_pts[idx] = (cx, cy)

            smooth_landmarks = curr_pts

            for start_idx, end_idx in PURE_BODY_CONNECTIONS:
                if start_idx in curr_pts and end_idx in curr_pts:
                    cv2.line(frame, curr_pts[start_idx], curr_pts[end_idx], (0, 255, 0), 2)

            for idx, pt in curr_pts.items():
                cv2.circle(frame, pt, 5, (255, 0, 0), -1)

            if detection_result.pose_world_landmarks:
                world_landmarks = detection_result.pose_world_landmarks[0]
                raw_coords, ground_fixed_coords = spine_analyzer.extract_coordinates(world_landmarks)
                
                u_spine = ground_fixed_coords["upper_spine"]
                m_spine = ground_fixed_coords["middle_spine"]
                d_spine = ground_fixed_coords["down_spine"]

                # ★ 십자(+) Frame 기반 3D 회전각 연산 (어깨-척추) ★
                pitch, roll, yaw = scipy_comparator.compute_cross_frame_rotation(
                    raw_coords["head_center"],
                    raw_coords["down_spine"],
                    raw_coords["l_shoulder"],
                    raw_coords["r_shoulder"]
                )

                r_elbow_angle = spine_analyzer.calculate_angle_3d(
                    raw_coords["r_shoulder"], raw_coords["r_elbow"], raw_coords["r_wrist"]
                )
                r_shoulder_angle = spine_analyzer.calculate_angle_3d(
                    raw_coords["hip_center"], raw_coords["r_shoulder"], raw_coords["r_elbow"]
                )
                current_angles = {"r_elbow_angle": r_elbow_angle, "r_shoulder_angle": r_shoulder_angle}

                spatial_errors, score, total_dist = scipy_comparator.calculate_scipy_error(
                    current_angles, ground_fixed_coords
                )

                info_texts = [
                    "=== SciPy CROSS-FRAME REHABILITATION ===",
                    f"[Upper Spine  (경추)] X:{u_spine[0]:.2f} Y:{-u_spine[1]:.2f} Z:{u_spine[2]:.2f} (m)",
                    f"[Middle Spine (흉추)] X:{m_spine[0]:.2f} Y:{-m_spine[1]:.2f} Z:{m_spine[2]:.2f} (m)",
                    f"[Down Spine   (요추)] X:{d_spine[0]:.2f} Y:{-d_spine[1]:.2f} Z:{d_spine[2]:.2f} (m)",
                    f"[3D Cross Frame] Pitch:{int(pitch)}° Roll:{int(roll)}° Yaw(Z-Twist):{int(yaw)}°",
                    f"[Pose Match Score] {score:.1f} / 100  (Spatial Err: {total_dist:.2f}m)"
                ]
                
                for i, text in enumerate(info_texts):
                    color = (0, 255, 255) if i == 0 else ((0, 215, 255) if i >= 4 else (0, 255, 0))
                    cv2.putText(frame, text, (30, 90 + i * 32), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"SciPy Rehab FPS: {int(fps)}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('SciPy Advanced Rehabilitation Analyzer', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_rehab_scipy_mode()