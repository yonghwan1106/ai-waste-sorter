"""
AI 스마트 분리수거 로봇 - 실시간 웹캠 분류
웹캠으로 쓰레기를 인식하고, 분류 결과를 화면에 표시 + Arduino로 전송

사용법:
    python src/detect.py                          # 기본 (웹캠 0번)
    python src/detect.py --source 1               # 웹캠 1번
    python src/detect.py --model models/best.onnx  # ONNX 모델
    python src/detect.py --no-serial              # Arduino 없이 테스트
"""

import argparse
import time
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# 분류 클래스 정보
CLASS_NAMES_KR = {
    0: "플라스틱",
    1: "캔",
    2: "종이",
    3: "유리병",
    4: "비닐",
    5: "일반쓰레기",
}

CLASS_COLORS = {
    0: (255, 165, 0),    # 플라스틱 - 주황
    1: (192, 192, 192),  # 캔 - 은색
    2: (139, 69, 19),    # 종이 - 갈색
    3: (0, 255, 0),      # 유리 - 초록
    4: (255, 255, 0),    # 비닐 - 노랑
    5: (128, 128, 128),  # 일반 - 회색
}

# 분류 통계 (대시보드 공유용)
stats = {
    "total": 0,
    "counts": {name: 0 for name in CLASS_NAMES_KR.values()},
    "history": [],
    "last_detection": None,
    "fps": 0.0,
}


def load_model(model_path: str) -> YOLO:
    """YOLOv8 모델 로드"""
    model = YOLO(model_path)
    print(f"모델 로드 완료: {model_path}")
    return model


def process_frame(model: YOLO, frame: np.ndarray, conf_threshold: float = 0.5):
    """프레임에서 쓰레기 감지 및 분류"""
    results = model(frame, conf=conf_threshold, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES_KR.get(cls_id, "unknown"),
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
        })

    return detections


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """감지 결과를 프레임에 그리기"""
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))

        # 바운딩 박스
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # 라벨 배경
        label = f'{det["class_name"]} {det["confidence"]:.1%}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 상단 정보 표시
    info = f"FPS: {stats['fps']:.1f} | Total: {stats['total']}"
    cv2.putText(annotated, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return annotated


def update_stats(detections: list):
    """분류 통계 업데이트"""
    for det in detections:
        stats["total"] += 1
        stats["counts"][det["class_name"]] = stats["counts"].get(det["class_name"], 0) + 1
        stats["last_detection"] = {
            "class": det["class_name"],
            "confidence": det["confidence"],
            "time": datetime.now().isoformat(),
        }
        stats["history"].append(stats["last_detection"])
        # 최근 100개만 유지
        if len(stats["history"]) > 100:
            stats["history"] = stats["history"][-100:]


def save_stats(path: str = "static/stats.json"):
    """통계를 JSON으로 저장 (대시보드 연동)"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def run_detection(
    model_path: str = "models/best.pt",
    source: int = 0,
    conf: float = 0.5,
    use_serial: bool = True,
    serial_port: str = "COM3",
    show: bool = True,
):
    """실시간 웹캠 분류 메인 루프"""
    model = load_model(model_path)

    # Arduino 시리얼 연결 (선택)
    ser = None
    if use_serial:
        try:
            import serial
            ser = serial.Serial(serial_port, 9600, timeout=1)
            time.sleep(2)  # Arduino 리셋 대기
            print(f"Arduino 연결 완료: {serial_port}")
        except Exception as e:
            print(f"Arduino 연결 실패 (시리얼 없이 계속): {e}")
            ser = None

    # 웹캠 열기
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"웹캠 열기 실패: source={source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"웹캠 열림: {int(cap.get(3))}x{int(cap.get(4))}")
    print("종료: 'q' 키")

    prev_time = time.time()
    frame_count = 0
    last_sent_class = None
    last_sent_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 감지 실행
            detections = process_frame(model, frame, conf)

            # FPS 계산
            frame_count += 1
            curr_time = time.time()
            if curr_time - prev_time >= 1.0:
                stats["fps"] = frame_count / (curr_time - prev_time)
                frame_count = 0
                prev_time = curr_time

            # 감지 결과 처리
            if detections:
                # 가장 높은 신뢰도의 감지 결과
                best = max(detections, key=lambda d: d["confidence"])
                update_stats([best])

                # Arduino로 분류 결과 전송 (1초 간격)
                if ser and (curr_time - last_sent_time > 1.0):
                    cmd = f"{best['class_id']}\n"
                    ser.write(cmd.encode())
                    last_sent_class = best["class_id"]
                    last_sent_time = curr_time
                    print(f"-> Arduino: class={best['class_id']} ({best['class_name']})")

                # 통계 저장
                save_stats()

            # 화면 표시
            if show:
                annotated = draw_detections(frame, detections)
                cv2.imshow("AI Waste Sorter", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser:
            ser.close()
        save_stats()
        print(f"\n=== 최종 통계 ===")
        print(f"총 분류: {stats['total']}개")
        for name, count in stats["counts"].items():
            if count > 0:
                print(f"  {name}: {count}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI 분리수거 로봇 - 실시간 분류")
    parser.add_argument("--model", type=str, default="models/best.pt", help="모델 경로")
    parser.add_argument("--source", type=int, default=0, help="웹캠 번호")
    parser.add_argument("--conf", type=float, default=0.5, help="신뢰도 임계값")
    parser.add_argument("--serial-port", type=str, default="COM3", help="Arduino 시리얼 포트")
    parser.add_argument("--no-serial", action="store_true", help="Arduino 없이 실행")
    parser.add_argument("--no-show", action="store_true", help="화면 표시 안 함")
    args = parser.parse_args()

    run_detection(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        use_serial=not args.no_serial,
        serial_port=args.serial_port,
        show=not args.no_show,
    )
