"""
AI 스마트 분리수거 로봇 - Arduino 시리얼 제어 모듈
서보모터 게이트 제어 및 LED 표시

사용법 (단독 테스트):
    python src/arduino_control.py --port COM3
"""

import argparse
import time

import serial
import serial.tools.list_ports


# 분류별 서보 각도 매핑 (2개 서보로 3방향 분류)
# 서보1: 좌(재활용)/우(일반) 방향 전환
# 서보2: 게이트 개폐
SERVO_ANGLES = {
    0: {"servo1": 0,   "servo2": 90},  # 플라스틱 -> 좌측 게이트
    1: {"servo1": 0,   "servo2": 90},  # 캔 -> 좌측 게이트
    2: {"servo1": 90,  "servo2": 90},  # 종이 -> 중앙 게이트
    3: {"servo1": 90,  "servo2": 90},  # 유리 -> 중앙 게이트
    4: {"servo1": 180, "servo2": 90},  # 비닐 -> 우측 게이트
    5: {"servo1": 180, "servo2": 90},  # 일반쓰레기 -> 우측 게이트
}

CLASS_NAMES = {
    0: "플라스틱", 1: "캔", 2: "종이",
    3: "유리병", 4: "비닐", 5: "일반쓰레기",
}


def list_serial_ports():
    """사용 가능한 시리얼 포트 목록"""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("사용 가능한 시리얼 포트가 없습니다.")
        return []
    print("사용 가능한 포트:")
    for p in ports:
        print(f"  {p.device}: {p.description}")
    return [p.device for p in ports]


class ArduinoController:
    """Arduino 시리얼 통신 제어 클래스"""

    def __init__(self, port: str = "COM3", baudrate: int = 9600, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.connected = False
        self.timeout = timeout

    def connect(self) -> bool:
        """Arduino 연결"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)  # Arduino 리셋 대기
            self.connected = True
            print(f"Arduino 연결 성공: {self.port} @ {self.baudrate}bps")

            # Arduino 준비 확인
            if self.ser.in_waiting:
                msg = self.ser.readline().decode("utf-8", errors="ignore").strip()
                print(f"  Arduino: {msg}")

            return True
        except serial.SerialException as e:
            print(f"Arduino 연결 실패: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """연결 종료"""
        if self.ser and self.ser.is_open:
            self.send_command("R")  # 리셋 명령
            time.sleep(0.5)
            self.ser.close()
            self.connected = False
            print("Arduino 연결 종료")

    def send_command(self, cmd: str) -> str:
        """명령 전송 및 응답 수신"""
        if not self.connected or not self.ser:
            return ""

        try:
            self.ser.write(f"{cmd}\n".encode())
            time.sleep(0.1)

            response = ""
            if self.ser.in_waiting:
                response = self.ser.readline().decode("utf-8", errors="ignore").strip()

            return response
        except serial.SerialException as e:
            print(f"통신 오류: {e}")
            return ""

    def sort_waste(self, class_id: int):
        """쓰레기 분류 명령 전송

        프로토콜: 클래스 ID(0~5)를 문자열로 전송
        Arduino가 해당 서보 각도로 게이트 제어
        """
        if class_id not in CLASS_NAMES:
            print(f"잘못된 클래스 ID: {class_id}")
            return

        name = CLASS_NAMES[class_id]
        angles = SERVO_ANGLES[class_id]
        print(f"분류: {name} (class={class_id}) -> 서보1={angles['servo1']}°, 서보2={angles['servo2']}°")

        response = self.send_command(str(class_id))
        if response:
            print(f"  Arduino 응답: {response}")

    def reset_gate(self):
        """게이트 초기 위치로 리셋"""
        response = self.send_command("R")
        print(f"게이트 리셋 {f'-> {response}' if response else ''}")

    def test_servos(self):
        """모든 서보 동작 테스트"""
        print("\n=== 서보모터 테스트 ===")
        for cls_id in range(6):
            name = CLASS_NAMES[cls_id]
            print(f"\n[{cls_id}] {name} 분류 테스트...")
            self.sort_waste(cls_id)
            time.sleep(2)
            self.reset_gate()
            time.sleep(1)
        print("\n테스트 완료!")


def interactive_test(port: str):
    """대화형 테스트 모드"""
    ctrl = ArduinoController(port=port)

    if not ctrl.connect():
        list_serial_ports()
        return

    print("\n=== 대화형 테스트 모드 ===")
    print("명령어:")
    print("  0~5: 해당 클래스로 분류")
    print("  t: 전체 서보 테스트")
    print("  r: 게이트 리셋")
    print("  q: 종료")

    try:
        while True:
            cmd = input("\n명령> ").strip().lower()

            if cmd == "q":
                break
            elif cmd == "t":
                ctrl.test_servos()
            elif cmd == "r":
                ctrl.reset_gate()
            elif cmd.isdigit() and 0 <= int(cmd) <= 5:
                ctrl.sort_waste(int(cmd))
            else:
                print("잘못된 명령입니다.")
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arduino 서보 제어 테스트")
    parser.add_argument("--port", type=str, default="COM3", help="시리얼 포트")
    parser.add_argument("--list", action="store_true", help="포트 목록 표시")
    args = parser.parse_args()

    if args.list:
        list_serial_ports()
    else:
        interactive_test(args.port)
