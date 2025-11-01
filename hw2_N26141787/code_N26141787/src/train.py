import os
import random
import shutil
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import argparse
import yaml


def convert_to_yolo_format(gt_path, label_path, image_w, image_h):
    """
    將作業的 Ground Truth 格式轉換為 YOLO 格式。
    作業格式: <class label>,<Top-left X>,<Top-left Y>,<Bounding box width>,<Bounding box height>
    YOLO 格式: <class-ID> <x-center-norm> <y-center-norm> <width-norm> <height-norm>
    """
    with open(gt_path, 'r') as f_in, open(label_path, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split(',')
            if len(parts) != 5:
                continue

            class_id, x_tl, y_tl, w, h = map(float, parts)
            class_id = int(class_id)

            x_center = x_tl + w / 2
            y_center = y_tl + h / 2

            x_center_norm = x_center / image_w
            y_center_norm = y_center / image_h
            w_norm = w / image_w
            h_norm = h / image_h

            f_out.write(
                f"{class_id} {x_center_norm} {y_center_norm} {w_norm} {h_norm}\n")


def prepare_dataset(base_data_path, output_path, train_split=0.9):
    """
    準備 YOLO 格式的資料集，包含分割訓練/驗證集和格式轉換。
    """
    train_dir = Path(base_data_path) / 'train'

    # 建立輸出資料夾結構
    output_images_train = Path(output_path) / 'images' / 'train'
    output_images_val = Path(output_path) / 'images' / 'val'
    output_labels_train = Path(output_path) / 'labels' / 'train'
    output_labels_val = Path(output_path) / 'labels' / 'val'

    os.makedirs(output_images_train, exist_ok=True)
    os.makedirs(output_images_val, exist_ok=True)
    os.makedirs(output_labels_train, exist_ok=True)
    os.makedirs(output_labels_val, exist_ok=True)

    image_files = sorted([p for p in train_dir.glob('*.png')])
    if not image_files:
        print(f"錯誤：在 '{train_dir}' 中找不到任何 .png 檔案。")
        exit()

    random.shuffle(image_files)

    split_idx = int(len(image_files) * train_split)
    train_files, val_files = image_files[:split_idx], image_files[split_idx:]

    print(
        f"總圖片數: {len(image_files)}, 訓練集: {len(train_files)}, 驗證集: {len(val_files)}")

    print("正在處理訓練集...")
    for img_path in tqdm(train_files):
        shutil.copy(img_path, output_images_train / img_path.name)
        gt_path = img_path.with_suffix('.txt')
        label_path = output_labels_train / f"{img_path.stem}.txt"
        if gt_path.exists():
            h, w, _ = cv2.imread(str(img_path)).shape
            convert_to_yolo_format(gt_path, label_path, w, h)

    print("正在處理驗證集...")
    for img_path in tqdm(val_files):
        shutil.copy(img_path, output_images_val / img_path.name)
        gt_path = img_path.with_suffix('.txt')
        label_path = output_labels_val / f"{img_path.stem}.txt"
        if gt_path.exists():
            h, w, _ = cv2.imread(str(img_path)).shape
            convert_to_yolo_format(gt_path, label_path, w, h)

    print("資料集準備完成！")
    return str(Path(output_path).resolve())


def create_yaml_file(dataset_path, output_path):
    """
    建立 YOLO 訓練所需的 data.yaml 檔案。
    """
    class_names = ['car', 'hov', 'person', 'motorcycle']

    data_yaml = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }

    yaml_path = Path(output_path) / 'data.yaml'

    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"data.yaml 已建立於: {yaml_path}")
    return str(yaml_path.resolve())


def main(args):

    prepared_dataset_path = prepare_dataset(
        args.data_path, args.output_path)  # 1. 準備資料集
    yaml_file_path = create_yaml_file(
        prepared_dataset_path, args.output_path)  # 2. 建立 YAML 檔案
    random.seed(0)  # 固定隨機種子以確保可重現性
    model = YOLO(args.model_yaml)

    print("\n" + "="*30)
    print("開始 YOLOv8 訓練")
    print(f"模型架構: {args.model_yaml}")
    print(f"資料集設定: {yaml_file_path}")
    print(f"訓練週期: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print("重要：將不會使用任何預訓練權重 (pretrained=False)")
    print("="*30 + "\n")

    model.train(
        data=yaml_file_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        pretrained=False,
        project="cvpdl_hw2_runs",
        name=f"{Path(args.model_yaml).stem}_scratch",
        cos_lr=True,
        lrf=0.01,           # 啟用 Cosine Annealing 學習率排程
        weight_decay=0.0006,  # 稍微增加權重衰減
        mixup=0.1,            # 開啟 10% 的 MixUp
        mosaic=1,          # 始終啟用 Mosaic 增強
        close_mosaic=10,  # 在最後 10 個 epoch 關閉 Mosaic
        copy_paste=0.15,     # 20% 的機率啟用 Copy-Paste，增加稀有物件
        # fl_gamma=1.5,        # 使用 Focal Loss 來處理類別不平衡
        degrees=5,          # 增加隨機旋轉
        patience=15            # 早停的耐心值
    )
    print("訓練完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="為 CVPDL HW2 訓練 YOLOv8 模型")
    parser.add_argument('--data-path', type=str,
                        required=True, help="解壓縮後的原始資料夾路徑")
    parser.add_argument('--output-path', type=str,
                        default='./yolo_dataset', help="儲存 YOLO 格式資料集的路徑")
    parser.add_argument('--model-yaml', type=str, default='yolov8n.yaml',
                        help="YOLO 模型架構檔 (e.g., yolov8n.yaml, yolov8s.yaml)")
    parser.add_argument('--epochs', type=int, default=100, help="訓練週期數")
    parser.add_argument('--batch-size', type=int,
                        default=16, help="批次大小 (依 VRAM 調整)")
    parser.add_argument('--img-size', type=int, default=640, help="輸入圖片尺寸")
    parser.add_argument('--seed', type=int, default=0, help="隨機種子")

    args = parser.parse_args()
    main(args)

    """
    python train.py \
    --data-path ./CVPDL_hw2/CVPDL_hw2 \
    --epochs 150 \
    --batch-size 2 \
    --model-yaml yolov8m.yaml \
    --img-size 1920 \
    --seed 0
    """
