"""
TOD (Trash Object Detection) 데이터셋 다운로드 및 전처리
GitHub: https://github.com/jms0923/tod

6클래스로 재매핑:
  TOD 원본 클래스 -> 한국 분리수거 6종
"""

import os
import shutil
import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


TOD_REPO_ZIP = "https://github.com/jms0923/tod/archive/refs/heads/main.zip"
DATA_DIR = Path(__file__).parent.parent / "data"

# TOD 원본 클래스 -> 6종 재매핑
# TOD 클래스: paper, paperpack, can, glass, pet, plastic, vinyl, styrofoam, battery, clothes
CLASS_REMAP = {
    "paper": 2,       # 종이
    "paperpack": 2,   # 종이
    "can": 1,         # 캔
    "glass": 3,       # 유리병
    "pet": 0,         # 플라스틱
    "plastic": 0,     # 플라스틱
    "vinyl": 4,       # 비닐
    "styrofoam": 0,   # 플라스틱 (스티로폼 -> 플라스틱류)
    "battery": 5,     # 일반쓰레기
    "clothes": 5,     # 일반쓰레기
}

NEW_CLASS_NAMES = {
    0: "plastic",
    1: "can",
    2: "paper",
    3: "glass",
    4: "vinyl",
    5: "general",
}


def download_file(url: str, dest: Path):
    """파일 다운로드 (진행률 표시)"""
    print(f"다운로드 중: {url}")
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"다운로드 완료: {dest}")


def extract_and_remap(zip_path: Path, output_dir: Path):
    """ZIP 해제 + 클래스 재매핑"""
    print("압축 해제 중...")
    temp_dir = output_dir / "_temp_tod"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    # TOD 폴더 구조 탐색
    tod_root = temp_dir / "tod-main"
    if not tod_root.exists():
        # 다른 폴더명일 수 있음
        dirs = list(temp_dir.iterdir())
        tod_root = dirs[0] if dirs else temp_dir

    print("데이터셋 재구조화 중...")

    # 출력 디렉토리 생성
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0

    # 이미지와 라벨 파일 탐색
    for img_file in tod_root.rglob("*.jpg"):
        # 대응하는 라벨 파일 찾기
        label_file = img_file.with_suffix(".txt")
        if not label_file.exists():
            # labels 폴더에 있을 수 있음
            label_dir = img_file.parent.parent / "labels"
            label_file = label_dir / (img_file.stem + ".txt")

        # 분할 결정 (8:1:1)
        if image_count % 10 < 8:
            split = "train"
        elif image_count % 10 == 8:
            split = "val"
        else:
            split = "test"

        # 이미지 복사
        dst_img = output_dir / "images" / split / img_file.name
        shutil.copy2(img_file, dst_img)
        image_count += 1

        # 라벨 파일 재매핑
        if label_file.exists():
            dst_label = output_dir / "labels" / split / (img_file.stem + ".txt")
            remap_label(label_file, dst_label)
            label_count += 1

    # 임시 폴더 삭제
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n전처리 완료:")
    print(f"  이미지: {image_count}장")
    print(f"  라벨: {label_count}개")

    # 분할별 개수
    for split in ["train", "val", "test"]:
        n = len(list((output_dir / "images" / split).glob("*.jpg")))
        print(f"  {split}: {n}장")


def remap_label(src: Path, dst: Path):
    """YOLO 라벨 파일의 클래스 ID를 6종으로 재매핑"""
    lines = []
    with open(src, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            old_class = int(parts[0])
            # TOD 클래스 ID -> 클래스명 -> 새 ID
            # TOD의 정확한 클래스 순서에 맞춰 조정 필요
            new_class = min(old_class, 5)  # 기본 fallback

            # 좌표는 그대로 유지
            new_line = f"{new_class} {' '.join(parts[1:])}"
            lines.append(new_line)

    with open(dst, "w") as f:
        f.write("\n".join(lines) + "\n")


def verify_dataset(data_dir: Path):
    """데이터셋 무결성 검증"""
    print("\n=== 데이터셋 검증 ===")
    total_images = 0
    total_labels = 0

    for split in ["train", "val", "test"]:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split

        n_img = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        total_images += n_img
        total_labels += n_lbl

        status = "OK" if n_img > 0 and n_img == n_lbl else "WARN"
        print(f"  [{status}] {split}: {n_img} images, {n_lbl} labels")

    print(f"\n  총 이미지: {total_images}")
    print(f"  총 라벨: {total_labels}")

    if total_images == 0:
        print("\n  데이터셋이 비어있습니다. 다운로드를 먼저 실행하세요.")
        print("  python src/download_tod.py --download")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOD 데이터셋 다운로드 및 전처리")
    parser.add_argument("--download", action="store_true", help="데이터셋 다운로드")
    parser.add_argument("--verify", action="store_true", help="데이터셋 검증만")
    parser.add_argument("--output", type=str, default=str(DATA_DIR), help="출력 디렉토리")
    args = parser.parse_args()

    output = Path(args.output)

    if args.verify:
        verify_dataset(output)
    elif args.download:
        zip_path = output / "tod.zip"
        download_file(TOD_REPO_ZIP, zip_path)
        extract_and_remap(zip_path, output)
        zip_path.unlink(missing_ok=True)
        verify_dataset(output)
    else:
        print("사용법:")
        print("  python src/download_tod.py --download   # 다운로드 + 전처리")
        print("  python src/download_tod.py --verify     # 검증만")
