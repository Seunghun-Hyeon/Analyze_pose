import sys
from run_rehab import start_rehab_mode
from run_sports import start_sports_mode

def main():
    if len(sys.argv) < 2:
        print("\n[사용법]")
        print("  python cli.py run_twist  --> 지면 기준 Spine 데이터 재활 치료 모드 실행")
        print("  python cli.py run_test   --> 피크 속도 & 자동 로그 저장의 스포츠 모드 실행\n")
        return

    command = sys.argv[1]

    if command == "run_twist":
        print("🏥 [재활 모드 (run_twist)] 실행...")
        start_rehab_mode()
    elif command == "run_test":
        print("⚾ [스포츠 모드 (run_test)] 실행...")
        start_sports_mode()
    else:
        print(f"❌ 알 수 없는 명령어입니다: {command}")
        print("사용 가능 명령어: run_twist, run_test")

if __name__ == "__main__":
    main()