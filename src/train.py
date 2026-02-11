"""
AI 스마트 분리수거 로봇 - YOLOv8 학습 스크립트
Google Colab에서 실행 권장 (무료 T4 GPU)

사용법:
    python src/train.py --data data/dataset.yaml --epochs 50
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(data_yaml: str, epochs: int = 50, imgsz: int = 640, batch: int = 16):
    """YOLOv8n 전이학습 실행"""
    # COCO 사전학습 가중치로 YOLOv8 Nano 로드
    model = YOLO("yolov8n.pt")

    # 학습 실행
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="waste_sorter",
        patience=10,          # Early stopping (10 에포크 개선 없으면 중단)
        save=True,
        save_period=10,       # 10 에포크마다 체크포인트 저장
        device="0",           # GPU 사용 (Colab), CPU: "cpu"
        workers=2,
        # 데이터 증강
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    return results


def evaluate(model_path: str, data_yaml: str):
    """학습된 모델 성능 평가"""
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    print("\n=== 모델 성능 평가 결과 ===")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    return metrics


def export_model(model_path: str, fmt: str = "onnx"):
    """모델을 ONNX/TFLite로 변환 (경량 배포용)"""
    model = YOLO(model_path)
    exported = model.export(format=fmt, imgsz=640, simplify=True)
    print(f"\n모델 변환 완료: {exported}")
    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI 분리수거 로봇 - YOLOv8 학습")
    parser.add_argument("--data", type=str, default="data/dataset.yaml", help="데이터셋 YAML 경로")
    parser.add_argument("--epochs", type=int, default=50, help="학습 에포크 수")
    parser.add_argument("--imgsz", type=int, default=640, help="입력 이미지 크기")
    parser.add_argument("--batch", type=int, default=16, help="배치 크기")
    parser.add_argument("--eval", type=str, default=None, help="평가할 모델 경로 (예: runs/detect/waste_sorter/weights/best.pt)")
    parser.add_argument("--export", type=str, default=None, help="변환할 모델 경로")
    parser.add_argument("--export-format", type=str, default="onnx", choices=["onnx", "tflite"], help="변환 포맷")
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval, args.data)
    elif args.export:
        export_model(args.export, args.export_format)
    else:
        results = train(args.data, args.epochs, args.imgsz, args.batch)
        # 학습 완료 후 자동 평가
        best_model = Path("runs/detect/waste_sorter/weights/best.pt")
        if best_model.exists():
            evaluate(str(best_model), args.data)
            export_model(str(best_model), "onnx")
