import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 별도 모듈 불러오기
from Spine_Data import SpineCoordinateAnalyzer
from Speed_Data import KinematicsSpeedAnalyzer

# 1. 모델 파일 다운로드 (Lite)
MODEL_PATH = 'pose_landmarker_lite.task'
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    print("경량화 포즈 모델(Lite) 다운로드 중...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

# 2. MediaPipe PoseLandmarker 옵션 설정
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
speed_analyzer = KinematicsSpeedAnalyzer()

PURE_BODY_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]
# 발목(27, 28) 및 발(29, 30, 31, 32) 인덱스까지 완전히 포함
KEY_LANDMARK_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# 4. Camo 카메라 연결 (1080p / 60 FPS / MJPG)
cap = None
for idx in range(5):
    temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if temp_cap.isOpened():
        temp_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        temp_cap.set(cv2.CAP_PROP_FPS, 60)
        temp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, frame = temp_cap.read()
        if ret and frame is not None and frame.size > 0:
            print(f"✅ 아이폰 15 Pro 카메라 연결 성공! (인덱스: {idx})")
            cap = temp_cap
            break
        temp_cap.release()

if cap is None:
    print("❌ Camo 카메라를 찾을 수 없습니다. Camo Studio 프로그램이 켜져 있는지 확인하세요.")
    exit()

smooth_landmarks = {}
ALPHA = 0.85

start_time = time.time()
prev_frame_time = 0

# 5. 실시간 추적 및 모션 분석 루프
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
                cv2.line(frame, curr_pts[start_idx], curr_pts[end_idx], (0, 0, 255), 2)

        for idx, pt in curr_pts.items():
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)

        # 3D 분석 연산
        if detection_result.pose_world_landmarks:
            world_landmarks = detection_result.pose_world_landmarks[0]

            # 1) Spine_Data.py: 지면 기준 고정 3D 좌표계 및 각도 산출
            raw_coords, ground_fixed_coords = spine_analyzer.extract_coordinates(world_landmarks)
            r_elbow_angle = spine_analyzer.calculate_angle_3d(
                raw_coords["r_shoulder"], raw_coords["r_elbow"], raw_coords["r_wrist"]
            )
            r_shoulder_angle = spine_analyzer.calculate_angle_3d(
                raw_coords["hip_center"], raw_coords["r_shoulder"], raw_coords["r_elbow"]
            )

            # 2) Speed_Data.py: 속도 및 피크 분석
            speed_data = speed_analyzer.analyze_speed_and_peak(
                raw_coords, ground_fixed_coords, r_elbow_angle, r_shoulder_angle, curr_frame_time
            )

            ground_wrist = ground_fixed_coords["r_wrist"]

            # 지면 기준 라이브 수치 표기 (노란색)
            info_texts = [
                f"[LIVE] Wrist Speed : {speed_data['wrist_speed_kmh']:.1f} km/h",
                f"[LIVE] Elbow Angle  : {int(r_elbow_angle)} deg",
                f"[GROUND 3D] Wrist Ht: Y:{-ground_wrist[1]:.2f}m (Floor Ref)"
            ]
            for i, text in enumerate(info_texts):
                cv2.putText(frame, text, (30, 100 + i * 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)

            # 피크 스냅샷 표기 (주황색)
            snapshot = speed_data["peak_snapshot"]
            if snapshot is not None:
                pw = snapshot["fixed_wrist_3d"]
                peak_texts = [
                    "--- [PEAK RELEASE SNAPSHOT] ---",
                    f"Max Speed      : {snapshot['peak_speed_kmh']:.1f} km/h",
                    f"Elbow Angle    : {int(snapshot['elbow_angle'])} deg",
                    f"Shoulder Angle : {int(snapshot['shoulder_angle'])} deg",
                    f"Ground Wrist Ht: Y:{-pw[1]:.2f}m"
                ]
                for i, text in enumerate(peak_texts):
                    color = (0, 165, 255) if i == 0 else (255, 200, 0)
                    cv2.putText(frame, text, (30, 230 + i * 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"iPhone 15 Pro Camo FPS: {int(fps)}", (30, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('iPhone 15 Pro Ground-Fixed Pose Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()