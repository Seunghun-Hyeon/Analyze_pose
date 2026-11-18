import os
import numpy as np
import threading
from datetime import datetime

class KinematicsSpeedAnalyzer:
    """
    SpineCoordinateAnalyzer에서 산출된 3D 좌표와 시간 변화량(dt)을 바탕으로
    선속도(km/h, m/s), 각속도(deg/s), 최고 속도(Release) 순간 피크 스냅샷을 관리하고
    최고 속도 달성 시 Data.txt 파일로 비동기(Async Thread) 자동 저장하는 모듈
    """
    def __init__(self, *args, **kwargs):
        self.prev_time = None
        self.prev_r_wrist = None
        self.prev_r_elbow_angle = None

        self.max_wrist_speed_kmh = 0.0
        self.peak_snapshot = None
        self.hold_start_time = 0.0
        self.HOLD_DURATION = 3.0  # 스냅샷 화면 유지 시간 (3초)
        
        self.log_filename = kwargs.get("log_filename", "Data.txt")

    def _write_file_async(self, log_entry):
        """별도 쓰레드에서 파일 저장을 수행하여 메인 프레임 멈춤 방지"""
        try:
            with open(self.log_filename, "a", encoding="utf-8") as f:
                f.write(log_entry)
            print(f"💾 [{self.log_filename} 비동기 저장 완료]")
        except Exception as e:
            print(f"❌ 파일 저장 중 오류 발생: {e}")

    def _save_peak_to_txt(self, snapshot):
        """최고 속도 스냅샷 데이터를 포맷팅하여 비동기 파일 저장"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pw = snapshot["fixed_wrist_3d"]

        log_entry = (
            f"========================================\n"
            f" TIMESTAMP        : {now_str}\n"
            f" MAX WRIST SPEED  : {snapshot['peak_speed_kmh']:.2f} km/h\n"
            f" ELBOW ANGLE      : {int(snapshot['elbow_angle'])} deg\n"
            f" SHOULDER ANGLE   : {int(snapshot['shoulder_angle'])} deg\n"
            f" ANGULAR VELOCITY : {int(snapshot['angular_velocity'])} deg/s\n"
            f" FIXED WRIST 3D   : X={pw[0]:.3f}m, Y={pw[1]:.3f}m, Z={pw[2]:.3f}m\n"
            f"========================================\n\n"
        )

        thread = threading.Thread(target=self._write_file_async, args=(log_entry,), daemon=True)
        thread.start()

    def analyze_speed_and_peak(self, raw_coords, fixed_coords, r_elbow_angle, r_shoulder_angle, curr_time_sec):
        r_wrist = raw_coords["r_wrist"]

        wrist_speed_ms = 0.0
        wrist_speed_kmh = 0.0
        elbow_angular_velocity = 0.0

        if self.prev_time is not None and self.prev_r_wrist is not None:
            dt = curr_time_sec - self.prev_time
            if dt > 0.001:
                dist_m = np.linalg.norm(r_wrist - self.prev_r_wrist)
                wrist_speed_ms = dist_m / dt
                wrist_speed_kmh = wrist_speed_ms * 3.6

                angle_diff = abs(r_elbow_angle - self.prev_r_elbow_angle)
                elbow_angular_velocity = angle_diff / dt

        if wrist_speed_kmh > 8.0 and wrist_speed_kmh > self.max_wrist_speed_kmh:
            self.max_wrist_speed_kmh = wrist_speed_kmh
            self.hold_start_time = curr_time_sec
            self.peak_snapshot = {
                "peak_speed_kmh": wrist_speed_kmh,
                "elbow_angle": r_elbow_angle,
                "shoulder_angle": r_shoulder_angle,
                "angular_velocity": elbow_angular_velocity,
                "fixed_wrist_3d": fixed_coords["r_wrist"]
            }

            self._save_peak_to_txt(self.peak_snapshot)

        if curr_time_sec - self.hold_start_time > self.HOLD_DURATION:
            if wrist_speed_kmh < 4.0:
                self.max_wrist_speed_kmh = 0.0

        self.prev_time = curr_time_sec
        self.prev_r_wrist = r_wrist
        self.prev_r_elbow_angle = r_elbow_angle

        return {
            "wrist_speed_ms": wrist_speed_ms,
            "wrist_speed_kmh": wrist_speed_kmh,
            "elbow_angular_velocity": elbow_angular_velocity,
            "peak_snapshot": self.peak_snapshot
        }