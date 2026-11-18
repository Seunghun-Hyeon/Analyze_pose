import cv2
import time

print("🔍 PC 카메라 장치 전수 조사 중...\n")

found_any = False
for idx in range(10):
    # DSHOW와 MSMF 두 백엔드 모두 진단
    for backend_name, backend in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                h, w, _ = frame.shape
                print(f"✅ [찾음] 인덱스: {idx} | 백엔드: {backend_name} | 해상도: {w}x{h}")
                found_any = True
            cap.release()

if not found_any:
    print("❌ 연결된 카메라를 하나도 찾지 못했습니다!")
    print("👉 Camo Studio에서 아이폰 화면이 정상 렌더링 중인지 확인하세요.")