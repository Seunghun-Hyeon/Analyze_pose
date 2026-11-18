import cv2
import time
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 모듈 임포트
from Spine_Data import SpineCoordinateAnalyzer
from Skeleton_3D_Viewer import Skeleton3DViewer
from Pose_Compare_Scipy import PoseComparatorSciPy

def analyze_video_3d_scipy(video_path):
    if not os.path.exists(video_path):
        print(f"❌ 분석할 영상 파일을 찾을 수 없습니다: {video_path}")
        return

    MODEL_PATH = 'pose_landmarker_lite.task'
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    spine_analyzer = SpineCoordinateAnalyzer()
    viewer_3d = Skeleton3DViewer()
    scipy_comparator = PoseComparatorSciPy("ref_pose.json")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    print(f"🎬 3D SciPy 영상 분석 구동 중: {video_path}")
    print("   (영상 재생 중 'q'를 누르면 종료됩니다.)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int((frame_idx / fps) * 1000)
        frame_idx += 1

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)

        if detection_result.pose_world_landmarks:
            world_landmarks = detection_result.pose_world_landmarks[0]
            
            # 1. 3D 지면 고정 좌표 추출
            raw_coords, ground_fixed_coords = spine_analyzer.extract_coordinates(world_landmarks)
            
            # 2. 체형 차이 보정을 위한 척추 길이 기반 3D 좌표 정규화 (Normalization)
            spine_len = np.linalg.norm(raw_coords["head_center"] - raw_coords["down_spine"]) + 1e-6
            normalized_coords = {
                k: v / spine_len for k, v in ground_fixed_coords.items()
            }

            # 3. 3D 뷰포트 공간에 실시간 스켈레톤 재생
            viewer_3d.update_3d_plot(ground_fixed_coords)

            # 4. 십자(+) Frame 3D 회전각 연산 (Pitch: 숙임, Roll: 기울임, Yaw: Z축 비틀림)
            pitch, roll, yaw = scipy_comparator.compute_cross_frame_rotation(
                raw_coords["head_center"],
                raw_coords["down_spine"],
                raw_coords["l_shoulder"],
                raw_coords["r_shoulder"]
            )

            # 5. 관절 사이각 연산
            r_elbow_angle = spine_analyzer.calculate_angle_3d(
                raw_coords["r_shoulder"], raw_coords["r_elbow"], raw_coords["r_wrist"]
            )
            r_shoulder_angle = spine_analyzer.calculate_angle_3d(
                raw_coords["hip_center"], raw_coords["r_shoulder"], raw_coords["r_elbow"]
            )
            current_angles = {"r_elbow_angle": r_elbow_angle, "r_shoulder_angle": r_shoulder_angle}

            # 6. SciPy 오차 분석 & 체형 보정 일치도 점수 계산
            spatial_errors, score, total_dist = scipy_comparator.calculate_scipy_error(
                current_angles, normalized_coords
            )

            # 2D 비디오 화면 오버레이
            info_texts = [
                f"[SciPy Video Analysis] Score: {score:.1f} / 100",
                f"Spine 3D Rot -> Pitch:{int(pitch)} deg | Roll:{int(roll)} deg | Yaw(Z-Twist):{int(yaw)} deg",
                f"Normalized Spatial Error: {total_dist:.2f}"
            ]
            
            for i, text in enumerate(info_texts):
                color = (0, 255, 0) if i == 0 else (0, 255, 255)
                cv2.putText(frame, text, (30, 40 + i * 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.imshow('2D Video Player (Press Q to quit)', frame)

        # 실제 영상 프레임 속도에 맞춰 대기
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    viewer_3d.close()

if __name__ == "__main__":
    # 분석할 영상 파일명 지정
    analyze_video_3d_scipy("my_swing_Golf2.mp4")