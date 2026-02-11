"""
TOD (Trash Object Detection) 데이터셋 다운로드 및 전처리
Zenodo: https://doi.org/10.5281/zenodo.4607158

COCO 형식 → YOLO 형식 변환 + 10클래스 → 6클래스 재매핑

사용법:
    python src/download_tod.py --download          # 전체 (다운로드+변환)
    python src/download_tod.py --convert-only      # 이미 다운로드한 ZIP 변환만
    python src/download_tod.py --verify            # 검증만
"""

import json
import os
import random
import shutil
import argparse
import zipfile
from pathlib import Path
from collections import defaultdict

import requests
from tqdm import tqdm


# Zenodo 실제 데이터셋 URL (741MB)
TOD_ZENODO_URL = "https://zenodo.org/api/records/4607158/files/TrAsh_DAtaset_v1.1.zip/content"
DATA_DIR = Path(__file__).parent.parent / "data"

# TOD 원본 10 카테고리 (COCO format의 category id 순서)
# paper, paperpack, can, glass, pet, plastic, vinyl, styrofoam, battery, clothes
TOD_CATEGORIES = [
    "paper", "paperpack", "can", "glass", "pet",
    "plastic", "vinyl", "styrofoam", "battery", "clothes"
]

# TOD 원본 카테고리명 -> 우리 6클래스 ID 매핑
CLASS_REMAP = {
    "paper": 2,       # 종이
    "paperpack": 2,   # 종이
    "can": 1,         # 캔
    "glass": 3,       # 유리병
    "pet": 0,         # 플라스틱
    "plastic": 0,     # 플라스틱
    "vinyl": 4,       # 비닐
    "styrofoam": 0,   # 플라스틱류
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
    print(f"대상 경로: {dest}")
    print("(약 741MB, 네트워크에 따라 5~15분 소요)")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                pbar.update(len(chunk))

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"다운로드 완료: {dest} ({size_mb:.1f}MB)")


def coco_to_yolo_bbox(img_w: int, img_h: int, bbox: list) -> tuple:
    """COCO bbox [x,y,w,h] (절대좌표) → YOLO [cx,cy,w,h] (정규화 0~1)"""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # 범위 클리핑
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def convert_coco_to_yolo(coco_json_path: Path, images_dir: Path, output_dir: Path,
                         split_ratio=(0.8, 0.1, 0.1), seed=42):
    """COCO JSON → YOLO 형식 변환 + 6클래스 재매핑 + train/val/test 분할"""
    print(f"\nCOCO→YOLO 변환 중: {coco_json_path}")

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 카테고리 ID → 이름 매핑
    cat_id_to_name = {}
    for cat in coco["categories"]:
        cat_id_to_name[cat["id"]] = cat["name"].lower().strip()

    print(f"  원본 카테고리: {cat_id_to_name}")

    # 이미지 ID → 이미지 정보 매핑
    img_id_to_info = {}
    for img in coco["images"]:
        img_id_to_info[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }

    # 어노테이션을 이미지별로 그룹화
    img_annotations = defaultdict(list)
    skipped = 0
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        cat_name = cat_id_to_name.get(cat_id, "")

        if cat_name not in CLASS_REMAP:
            skipped += 1
            continue

        new_class = CLASS_REMAP[cat_name]
        img_annotations[img_id].append({
            "class_id": new_class,
            "bbox": ann["bbox"],  # COCO: [x, y, w, h]
        })

    print(f"  총 이미지: {len(img_id_to_info)}")
    print(f"  어노테이션 있는 이미지: {len(img_annotations)}")
    if skipped:
        print(f"  매핑 실패 스킵: {skipped}개")

    # train/val/test 분할
    img_ids = sorted(img_annotations.keys())
    random.seed(seed)
    random.shuffle(img_ids)

    n = len(img_ids)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])

    splits = {}
    for i, img_id in enumerate(img_ids):
        if i < n_train:
            splits[img_id] = "train"
        elif i < n_train + n_val:
            splits[img_id] = "val"
        else:
            splits[img_id] = "test"

    # 출력 디렉토리 생성
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 변환 실행
    counts = {"train": 0, "val": 0, "test": 0}
    class_counts = defaultdict(int)

    for img_id, anns in tqdm(img_annotations.items(), desc="변환 중"):
        info = img_id_to_info.get(img_id)
        if not info:
            continue

        split = splits[img_id]
        src_img = images_dir / info["file_name"]

        # 이미지가 하위 폴더에 있을 수 있음
        if not src_img.exists():
            # 재귀 탐색
            candidates = list(images_dir.rglob(info["file_name"]))
            if candidates:
                src_img = candidates[0]
            else:
                continue

        # 이미지 복사
        dst_img = output_dir / "images" / split / info["file_name"]
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # YOLO 라벨 생성
        label_lines = []
        for ann in anns:
            cx, cy, w, h = coco_to_yolo_bbox(info["width"], info["height"], ann["bbox"])
            if w > 0 and h > 0:
                label_lines.append(f"{ann['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                class_counts[ann["class_id"]] += 1

        dst_label = output_dir / "labels" / split / (Path(info["file_name"]).stem + ".txt")
        with open(dst_label, "w") as f:
            f.write("\n".join(label_lines) + "\n")

        counts[split] += 1

    # 결과 출력
    print(f"\n=== 변환 완료 ===")
    print(f"  train: {counts['train']}장")
    print(f"  val:   {counts['val']}장")
    print(f"  test:  {counts['test']}장")
    print(f"  총계:  {sum(counts.values())}장")
    print(f"\n  클래스별 어노테이션 수:")
    for cls_id in sorted(class_counts.keys()):
        name = NEW_CLASS_NAMES[cls_id]
        print(f"    [{cls_id}] {name}: {class_counts[cls_id]}개")


def extract_and_convert(zip_path: Path, output_dir: Path):
    """ZIP 해제 → COCO JSON 찾기 → YOLO 변환"""
    print("압축 해제 중...")
    temp_dir = output_dir / "_temp_tod"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    # COCO JSON 파일 찾기
    json_files = list(temp_dir.rglob("*.json"))
    print(f"  발견된 JSON 파일: {len(json_files)}")
    for jf in json_files:
        print(f"    {jf.relative_to(temp_dir)}")

    # annotations 폴더 내 JSON 찾기
    coco_json = None
    for jf in json_files:
        if "annotation" in str(jf).lower() or "instances" in jf.name.lower():
            coco_json = jf
            break

    if not coco_json and json_files:
        # 가장 큰 JSON 파일 사용 (보통 annotations 파일이 가장 큼)
        coco_json = max(json_files, key=lambda p: p.stat().st_size)

    if not coco_json:
        print("오류: COCO JSON 파일을 찾을 수 없습니다.")
        print("수동으로 경로를 지정하세요:")
        print("  python src/download_tod.py --convert-only --json <경로>")
        return

    print(f"  사용할 어노테이션: {coco_json.relative_to(temp_dir)}")

    # 이미지 폴더 찾기
    images_dir = None
    for candidate in ["train2017", "val2017", "images", "train", "img"]:
        found = list(temp_dir.rglob(candidate))
        if found:
            images_dir = found[0]
            break

    if not images_dir:
        # jpg 파일이 있는 폴더 찾기
        jpg_files = list(temp_dir.rglob("*.jpg"))
        if jpg_files:
            images_dir = jpg_files[0].parent
        else:
            png_files = list(temp_dir.rglob("*.png"))
            if png_files:
                images_dir = png_files[0].parent

    if not images_dir:
        print("오류: 이미지 폴더를 찾을 수 없습니다.")
        return

    n_images = len(list(images_dir.rglob("*.jpg"))) + len(list(images_dir.rglob("*.png")))
    print(f"  이미지 폴더: {images_dir.relative_to(temp_dir)} ({n_images}장)")

    # COCO → YOLO 변환
    convert_coco_to_yolo(coco_json, images_dir, output_dir)

    # 임시 폴더 삭제
    print("\n임시 파일 정리 중...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("정리 완료")


def verify_dataset(data_dir: Path):
    """데이터셋 무결성 검증"""
    print("\n=== 데이터셋 검증 ===")
    total_images = 0
    total_labels = 0

    for split in ["train", "val", "test"]:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split

        img_exts = ["*.jpg", "*.jpeg", "*.png"]
        n_img = sum(len(list(img_dir.glob(ext))) for ext in img_exts) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        total_images += n_img
        total_labels += n_lbl

        status = "OK" if n_img > 0 and n_img == n_lbl else ("WARN" if n_img > 0 else "EMPTY")
        print(f"  [{status}] {split}: {n_img} images, {n_lbl} labels")

    print(f"\n  총 이미지: {total_images}")
    print(f"  총 라벨:  {total_labels}")

    if total_images == 0:
        print("\n  데이터셋이 비어있습니다.")
        print("  다운로드: python src/download_tod.py --download")
        return

    # 라벨 내 클래스 분포 확인
    print("\n  클래스 분포:")
    class_dist = defaultdict(int)
    for split in ["train", "val", "test"]:
        lbl_dir = data_dir / "labels" / split
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        class_dist[cls_id] += 1

    for cls_id in sorted(class_dist.keys()):
        name = NEW_CLASS_NAMES.get(cls_id, f"unknown({cls_id})")
        print(f"    [{cls_id}] {name}: {class_dist[cls_id]}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOD 데이터셋 다운로드 및 YOLO 변환")
    parser.add_argument("--download", action="store_true", help="Zenodo에서 다운로드 + 변환")
    parser.add_argument("--convert-only", action="store_true", help="이미 다운로드한 ZIP 변환만")
    parser.add_argument("--verify", action="store_true", help="데이터셋 검증만")
    parser.add_argument("--zip", type=str, default=None, help="변환할 ZIP 경로 (--convert-only용)")
    parser.add_argument("--output", type=str, default=str(DATA_DIR), help="출력 디렉토리")
    args = parser.parse_args()

    output = Path(args.output)

    if args.verify:
        verify_dataset(output)
    elif args.convert_only:
        zip_path = Path(args.zip) if args.zip else output / "tod.zip"
        if not zip_path.exists():
            print(f"ZIP 파일을 찾을 수 없습니다: {zip_path}")
            print("--zip 옵션으로 경로를 지정하세요.")
        else:
            extract_and_convert(zip_path, output)
            verify_dataset(output)
    elif args.download:
        zip_path = output / "tod.zip"
        if zip_path.exists():
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            if size_mb > 100:
                print(f"기존 ZIP 파일 발견 ({size_mb:.0f}MB). 다운로드 건너뜀.")
                print("재다운로드하려면 파일을 삭제 후 다시 실행하세요.")
            else:
                download_file(TOD_ZENODO_URL, zip_path)
        else:
            download_file(TOD_ZENODO_URL, zip_path)
        extract_and_convert(zip_path, output)
        verify_dataset(output)
    else:
        print("AI 분리수거 로봇 - TOD 데이터셋 다운로드/변환 도구")
        print()
        print("사용법:")
        print("  python src/download_tod.py --download          # Zenodo 다운로드 + YOLO 변환")
        print("  python src/download_tod.py --convert-only      # ZIP 변환만")
        print("  python src/download_tod.py --verify            # 데이터셋 검증")
        print()
        print(f"데이터 디렉토리: {DATA_DIR}")
        print(f"Zenodo URL: {TOD_ZENODO_URL}")
        print(f"예상 크기: ~741MB (ZIP)")
