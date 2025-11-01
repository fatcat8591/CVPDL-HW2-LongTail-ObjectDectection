import os
import random
import re
from pathlib import Path
import cv2
from ultralytics import YOLO
import argparse
from tqdm import tqdm

# --- 常數設定 ---
CLASS_NAMES = ['car', 'hov', 'person', 'motorcycle']
PRED_COLOR = (255, 165, 0)  # 橘藍色


def draw_prediction_boxes(image, boxes, color):
    """在圖片上僅繪製 Prediction 的 Bounding Box"""
    for box_info in boxes:
        x_tl, y_tl, x_br, y_br, conf, class_id = list(
            map(int, box_info[:4])) + [box_info[4], int(box_info[5])]
        label = f"{CLASS_NAMES[class_id]} {conf:.2f}"
        cv2.rectangle(image, (x_tl, y_tl), (x_br, y_br), color, 2)
        cv2.putText(image, label, (x_tl, y_tl - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image


def visualize_test_predictions(weights_path, data_path, output_dir, num_images, image_ids, conf_thres=0.25):
    """
    視覺化模型在 TEST SET 上的預測結果。
    整合了隨機選取和指定 ID 兩種模式。
    """
    # 1. 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)
    print(f"測試集視覺化結果將儲存至: {output_dir}")

    # 2. 載入模型
    print(f"正在載入模型: {weights_path}")
    model = YOLO(weights_path)

    # 3. 根據是否提供 --image-ids 來決定選取圖片的模式
    test_dir = Path(data_path) / 'test'
    all_image_paths = list(test_dir.glob('*.png'))

    if not all_image_paths:
        print(f"錯誤: 在 '{test_dir}' 中找不到任何圖片。")
        return

    selected_images = []

    if image_ids:
        # --- 指定模式 ---
        print("模式：指定圖片視覺化")
        image_map = {int(re.search(r'\d+', p.name).group())
                         : p for p in all_image_paths}
        for requested_id in image_ids:
            id_to_find = int(requested_id)
            if id_to_find in image_map:
                selected_images.append(image_map[id_to_find])
            else:
                print(f"警告：在測試集中找不到 Image ID 為 {id_to_find} 的圖片。")
    else:
        # --- 隨機模式 ---
        print("模式：隨機圖片視覺化")
        if num_images > len(all_image_paths):
            selected_images = all_image_paths
        else:
            selected_images = random.sample(all_image_paths, num_images)

    if not selected_images:
        print("錯誤：沒有選取到任何圖片進行處理，終止程式。")
        return

    print(
        f"將對以下 {len(selected_images)} 張圖片進行視覺化: {[p.name for p in selected_images]}")

    # 4. 進行預測與繪圖
    for img_path in tqdm(selected_images):
        image = cv2.imread(str(img_path))
        preds = model.predict(img_path, conf=conf_thres, verbose=False)

        pred_boxes = []
        for box in preds[0].boxes:
            xyxy = box.xyxy[0].tolist()
            conf = box.conf.item()
            class_id = box.cls.item()
            pred_boxes.append(xyxy + [conf, class_id])

        image = draw_prediction_boxes(image, pred_boxes, PRED_COLOR)

        output_path = Path(output_dir) / f"test_vis_{img_path.name}"
        cv2.imwrite(str(output_path), image)

    print("\n" + "="*30)
    print("視覺化完成！")
    print(f"請至 '{output_dir}' 資料夾查看結果。")
    print("="*30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="視覺化 YOLOv8 在測試集上的預測結果 (整合模式)")
    parser.add_argument('--weights', type=str,
                        required=True, help="訓練好的模型權重路徑")
    parser.add_argument('--data-path', type=str,
                        required=True, help="原始資料夾的路徑")
    parser.add_argument('--output-dir', type=str,
                        default='test_visualization', help="儲存視覺化圖片的資料夾")
    parser.add_argument('--conf-thres', type=float,
                        default=0.2, help="預測的信心度閾值")

    # --- 核心修改：將 --image-ids 改為可選，並保留 --num-images ---
    parser.add_argument('--image-ids', type=str, nargs='*',
                        default=None, help="[指定模式] 要視覺化的圖片 ID (可指定多個，用空格隔開)")
    parser.add_argument('--num-images', type=int,
                        default=20, help="[隨機模式] 要視覺化的圖片數量")
    # --- 修改結束 ---

    args = parser.parse_args()
    visualize_test_predictions(args.weights, args.data_path, args.output_dir,
                               args.num_images, args.image_ids, args.conf_thres)

"""
python visualize_test.py \
    --weights ./cvpdl_hw2_runs/yolov8m_scratch4/weights/best.pt \
    --data-path ./CVPDL_hw2/CVPDL_hw2 \
    --num-images 50
"""

"""
如果指定要看某一張圖的結果 上面的是隨機選出30張來看結果
python visualize_test.py \
    --weights ./cvpdl_hw2_runs/yolov8m_scratch2/weights/best.pt \
    --data-path ./CVPDL_hw2/CVPDL_hw2 \
    --image-ids 176 412 
"""
