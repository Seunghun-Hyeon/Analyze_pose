import cv2
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from Spine_Data import SpineCoordinateAnalyzer
from Skeleton_3D_Viewer import Skeleton3DViewer
from Pose_Compare import PoseComparator

def analyze_video_in_3d(video_path):
    if not os.path.exists(video_path):
        print(f"❌ 영상 파일을 찾을 수 없습니다: {video_path}")
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
    comparator = PoseComparator("ref_pose.json")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    print(f"🎬 3D 영상 분석 구동 중: {video_path}")

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
            
            raw_coords, ground_fixed_coords = spine_analyzer.extract_coordinates(world_landmarks)
            
            # 3D 공간 뷰어 갱신
            viewer_3d.update_3d_plot(ground_fixed_coords)

            r_elbow_angle = spine_analyzer.calculate_angle_3d(
                raw_coords["r_shoulder"], raw_coords["r_elbow"], raw_coords["r_wrist"]
            )
            r_shoulder_angle = spine_analyzer.calculate_angle_3d(
                raw_coords["hip_center"], raw_coords["r_shoulder"], raw_coords["r_elbow"]
            )
            
            current_angles = {"r_elbow_angle": r_elbow_angle, "r_shoulder_angle": r_shoulder_angle}
            errors, score = comparator.calculate_error(current_angles, ground_fixed_coords)

            cv2.putText(frame, f"Pose Score: {score:.1f} / 100", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('2D Video Player', frame)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    viewer_3d.close()

if __name__ == "__main__":
    # 분석하고자 하는 영상 파일명 지정
    analyze_video_in_3d("test_video.mp4")