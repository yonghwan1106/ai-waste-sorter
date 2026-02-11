"""
AI 스마트 분리수거 로봇 - Flask 웹 대시보드
실시간 카메라 피드 + 분류 통계 모니터링

사용법:
    python src/dashboard.py                    # 기본 (포트 5000)
    python src/dashboard.py --port 8080        # 포트 변경
    python src/dashboard.py --no-camera        # 카메라 없이 (통계만)
"""

import argparse
import json
import time
import threading
from pathlib import Path
from datetime import datetime

import cv2
from flask import Flask, Response, render_template, jsonify

# detect.py와 공유할 통계 파일 경로
STATS_FILE = Path(__file__).parent.parent / "static" / "stats.json"

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent.parent / "templates"),
    static_folder=str(Path(__file__).parent.parent / "static"),
)

# 전역 변수
camera = None
model = None
use_camera = True


def get_stats() -> dict:
    """분류 통계 로드"""
    default = {
        "total": 0,
        "counts": {
            "플라스틱": 0, "캔": 0, "종이": 0,
            "유리병": 0, "비닐": 0, "일반쓰레기": 0,
        },
        "history": [],
        "last_detection": None,
        "fps": 0.0,
    }
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return default
    return default


def generate_frames():
    """카메라 프레임 스트리밍 (MJPEG)"""
    global camera, model

    if not use_camera:
        return

    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # AI 모델이 로드된 경우 추론 실행
        if model is not None:
            results = model(frame, conf=0.5, verbose=False)[0]
            frame = results.plot()  # 바운딩 박스가 그려진 프레임

        # JPEG 인코딩
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

        time.sleep(0.033)  # ~30 FPS


@app.route("/")
def index():
    """메인 대시보드 페이지"""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """실시간 카메라 피드 (MJPEG 스트림)"""
    if not use_camera:
        return "카메라 비활성화", 503
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/stats")
def api_stats():
    """분류 통계 API"""
    return jsonify(get_stats())


@app.route("/api/health")
def api_health():
    """서버 상태 확인"""
    return jsonify({
        "status": "ok",
        "camera": use_camera and camera is not None,
        "model": model is not None,
        "time": datetime.now().isoformat(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI 분리수거 로봇 - 웹 대시보드")
    parser.add_argument("--port", type=int, default=5000, help="서버 포트")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--no-camera", action="store_true", help="카메라 없이 실행")
    parser.add_argument("--model", type=str, default=None, help="YOLOv8 모델 경로")
    args = parser.parse_args()

    use_camera = not args.no_camera

    if args.model:
        from ultralytics import YOLO
        model = YOLO(args.model)
        print(f"AI 모델 로드: {args.model}")

    print(f"대시보드 시작: http://localhost:{args.port}")
    print(f"카메라: {'활성' if use_camera else '비활성'}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
