import numpy as np
import threading
from datetime import datetime
from scipy.signal import savgol_filter

class KinematicsSpeedAnalyzerSciPy:
    """
    SciPy Savitzky-Golay 필터를 적용하여 프레임 간 3D 위치 노이즈를 억제하고
    안정화된 순간 선속도(km/h) 및 각속도(deg/s)를 산출하는 모듈
    """
    def __init__(self, window_size=5, poly_order=2, *args, **kwargs):
        self.window_size = window_size
        self.poly_order = poly_order
        
        # 최근 N개 프레임의 3D 좌표 버퍼 (노이즈 제거용)
        self.wrist_x_buffer = []
        self.wrist_y_buffer = []
        self.wrist_z_buffer = []
        self.time_buffer = []

        self.prev_time = None
        self.prev_r_wrist_filtered = None
        self.prev_r_elbow_angle = None

        self.max_wrist_speed_kmh = 0.0
        self.peak_snapshot = None
        self.hold_start_time = 0.0
        self.HOLD_DURATION = 3.0
        self.log_filename = kwargs.get("log_filename", "Data_SciPy.txt")

    def _write_file_async(self, log_entry):
        try:
            with open(self.log_filename, "a", encoding="utf-8") as f:
                f.write(log_entry)
            print(f"💾 [{self.log_filename} SciPy 비동기 저장 완료]")
        except Exception as e:
            print(f"❌ 파일 저장 중 오류: {e}")

    def _save_peak_to_txt(self, snapshot):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pw = snapshot["fixed_wrist_3d"]

        log_entry = (
            f"========================================\n"
            f" TIMESTAMP (SciPy Filtered) : {now_str}\n"
            f" MAX WRIST SPEED            : {snapshot['peak_speed_kmh']:.2f} km/h\n"
            f" ELBOW ANGLE                : {int(snapshot['elbow_angle'])} deg\n"
            f" ANGULAR VELOCITY           : {int(snapshot['angular_velocity'])} deg/s\n"
            f" FIXED WRIST 3D             : X={pw[0]:.3f}m, Y={pw[1]:.3f}m, Z={pw[2]:.3f}m\n"
            f"========================================\n\n"
        )
        thread = threading.Thread(target=self._write_file_async, args=(log_entry,), daemon=True)
        thread.start()

    def analyze_speed_scipy(self, raw_coords, fixed_coords, r_elbow_angle, r_shoulder_angle, curr_time_sec):
        r_wrist = raw_coords["r_wrist"]

        # 3D 좌표 버퍼링
        self.wrist_x_buffer.append(r_wrist[0])
        self.wrist_y_buffer.append(r_wrist[1])
        self.wrist_z_buffer.append(r_wrist[2])
        self.time_buffer.append(curr_time_sec)

        # 버퍼 크기 유지
        if len(self.wrist_x_buffer) > 15:
            self.wrist_x_buffer.pop(0)
            self.wrist_y_buffer.pop(0)
            self.wrist_z_buffer.pop(0)
            self.time_buffer.pop(0)

        # Savitzky-Golay 필터 적용 (버퍼 개수가 필터 창 크기 이상일 때)
        if len(self.wrist_x_buffer) >= self.window_size:
            filt_x = savgol_filter(self.wrist_x_buffer, self.window_size, self.poly_order)[-1]
            filt_y = savgol_filter(self.wrist_y_buffer, self.window_size, self.poly_order)[-1]
            filt_z = savgol_filter(self.wrist_z_buffer, self.window_size, self.poly_order)[-1]
            curr_wrist_filtered = np.array([filt_x, filt_y, filt_z])
        else:
            curr_wrist_filtered = r_wrist

        wrist_speed_ms = 0.0
        wrist_speed_kmh = 0.0
        elbow_angular_velocity = 0.0

        if self.prev_time is not None and self.prev_r_wrist_filtered is not None:
            dt = curr_time_sec - self.prev_time
            if dt > 0.001:
                # 노이즈가 제거된 3D 이동 거리 기반 속도 계산
                dist_m = np.linalg.norm(curr_wrist_filtered - self.prev_r_wrist_filtered)
                wrist_speed_ms = dist_m / dt
                wrist_speed_kmh = wrist_speed_ms * 3.6

                if self.prev_r_elbow_angle is not None:
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
        self.prev_r_wrist_filtered = curr_wrist_filtered
        self.prev_r_elbow_angle = r_elbow_angle

        return {
            "wrist_speed_ms": wrist_speed_ms,
            "wrist_speed_kmh": wrist_speed_kmh,
            "elbow_angular_velocity": elbow_angular_velocity,
            "peak_snapshot": self.peak_snapshot
        }