"""
AI 스마트 분리수거 로봇 - 모델 간단 테스트
YOLOv8 사전학습 모델이 정상 동작하는지 확인

사용법:
    python src/test_model.py                    # COCO 사전학습 모델 테스트
    python src/test_model.py --model models/best.pt  # 커스텀 모델 테스트
"""

import argparse
import time
import numpy as np


def test_yolo_load(model_path: str = "yolov8n.pt"):
    """YOLOv8 모델 로드 테스트"""
    from ultralytics import YOLO

    print(f"모델 로드 중: {model_path}")
    t0 = time.time()
    model = YOLO(model_path)
    print(f"모델 로드 완료: {time.time() - t0:.2f}초")
    print(f"모델 타입: {model.type}")
    print(f"모델 이름: {model.model_name}")
    return model


def test_inference(model, num_runs: int = 10):
    """더미 이미지로 추론 속도 테스트"""
    # 더미 이미지 생성 (640x640 RGB)
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # 웜업
    print("\n추론 웜업 (3회)...")
    for _ in range(3):
        model(dummy, verbose=False)

    # 벤치마크
    print(f"벤치마크 ({num_runs}회)...")
    times = []
    for _ in range(num_runs):
        t0 = time.time()
        results = model(dummy, verbose=False)
        times.append(time.time() - t0)

    avg_ms = np.mean(times) * 1000
    fps = 1000 / avg_ms
    print(f"\n=== 추론 성능 ===")
    print(f"평균: {avg_ms:.1f}ms")
    print(f"FPS:  {fps:.1f}")
    print(f"최소: {min(times)*1000:.1f}ms")
    print(f"최대: {max(times)*1000:.1f}ms")

    return fps


def test_webcam_single(model):
    """웹캠에서 한 프레임 캡처 + 추론 테스트"""
    import cv2

    print("\n웹캠 테스트...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 건너뜁니다.")
        return False

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("프레임 캡처 실패")
        return False

    print(f"프레임 크기: {frame.shape}")
    results = model(frame, verbose=False)[0]
    n_detections = len(results.boxes)
    print(f"감지 객체: {n_detections}개")

    if n_detections > 0:
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = results.names[cls]
            print(f"  - {name}: {conf:.1%}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 모델 테스트")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="모델 경로")
    parser.add_argument("--webcam", action="store_true", help="웹캠 테스트 포함")
    parser.add_argument("--runs", type=int, default=10, help="벤치마크 반복 횟수")
    args = parser.parse_args()

    model = test_yolo_load(args.model)
    fps = test_inference(model, args.runs)

    if args.webcam:
        test_webcam_single(model)

    print("\n모든 테스트 완료!")
    if fps >= 30:
        print(f"FPS {fps:.1f} >= 30: 실시간 처리 가능")
    else:
        print(f"FPS {fps:.1f} < 30: ONNX 변환 또는 GPU 사용 권장")
