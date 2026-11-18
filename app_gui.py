import os
# macOS Tk 경고 문구 억제
os.environ['TK_SILENCE_DEPRECATION'] = '1'

import sys
import tkinter as tk
from tkinter import ttk, messagebox

# 기존 모듈 함수 임포트
from run_rehab import start_rehab_mode
from run_sports import start_sports_mode

class MotionAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MediaPipe Motion Analysis System")
        # geometry 규격 띄어쓰기 제거 ("450x380")
        self.root.geometry("450x380")
        self.root.resizable(False, False)

        # 스타일 설정
        style = ttk.Style()
        style.theme_use('clam')

        # 메인 프레임
        main_frame = ttk.Frame(root, padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 타이틀
        title_label = ttk.Label(
            main_frame, 
            text="모션 분석 시스템 모드 선택", 
            font=("Pretendard", 16, "bold")
        )
        title_label.pack(pady=(10, 20))

        # 설명 문구
        desc_label = ttk.Label(
            main_frame, 
            text="구동할 분석 모드를 선택하세요.", 
            font=("Pretendard", 10)
        )
        desc_label.pack(pady=(0, 20))

        # 버튼 생성
        btn_rehab = tk.Button(
            main_frame,
            text="🏥 재활 치료 모드 (run_twist)\n[지면 기준 척추 & 관절 가동범위 추적]",
            font=("Pretendard", 11, "bold"),
            bg="#2E7D32",
            fg="white",
            activebackground="#1B5E20",
            activeforeground="white",
            height=3,
            cursor="hand2",
            command=self.launch_rehab
        )
        btn_rehab.pack(fill=tk.X, pady=8)

        btn_sports = tk.Button(
            main_frame,
            text="⚾ 스포츠 모드 (run_test)\n[릴리즈 peak 속도 측정 & Data.txt 자동 저장]",
            font=("Pretendard", 11, "bold"),
            bg="#1565C0",
            fg="white",
            activebackground="#0D47A1",
            activeforeground="white",
            height=3,
            cursor="hand2",
            command=self.launch_sports
        )
        btn_sports.pack(fill=tk.X, pady=8)

        # 종료 버튼
        btn_exit = tk.Button(
            main_frame,
            text="종료",
            font=("Pretendard", 10),
            bg="#757575",
            fg="white",
            command=self.root.quit
        )
        btn_exit.pack(fill=tk.X, pady=(15, 0))

    def launch_rehab(self):
        print("🏥 [재활 모드] 실행중...")
        try:
            start_rehab_mode()
        except Exception as e:
            messagebox.showerror("오류 발생", f"재활 모드 실행 중 오류가 발생했습니다:\n{e}")

    def launch_sports(self):
        print("⚾ [스포츠 모드] 실행중...")
        try:
            start_sports_mode()
        except Exception as e:
            messagebox.showerror("오류 발생", f"스포츠 모드 실행 중 오류가 발생했습니다:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MotionAnalysisApp(root)
    root.mainloop()