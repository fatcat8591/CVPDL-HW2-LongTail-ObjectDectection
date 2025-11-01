import os
import re
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import argparse
from tqdm import tqdm


def predict_and_generate_submission(weights_path, test_dir, output_csv, img_size=640, conf_thres=0.01):
    """
    使用訓練好的 YOLOv8 模型進行預測，並產生 Kaggle 提交檔案。
    """
    # 1. 載入訓練好的模型
    print(f"正在載入模型權重: {weights_path}")
    model = YOLO(weights_path)

    # 2. 獲取並排序測試圖片
    test_image_paths = sorted(
        [p for p in Path(test_dir).glob('*.png')],
        key=lambda p: int(re.search(r'\d+', p.name).group())  # 確保按數字順序排序
    )

    if not test_image_paths:
        print(f"錯誤: 在 '{test_dir}' 中找不到任何 .png 圖片。")
        return

    print(f"找到 {len(test_image_paths)} 張測試圖片。")

    # 3. 準備儲存結果
    results_list = []

    # 4. 迭代所有測試圖片進行預測
    print("開始進行預測...")
    for img_path in tqdm(test_image_paths):
        # 進行預測
        preds = model.predict(img_path, imgsz=img_size,
                              conf=conf_thres, verbose=False)

        # 提取 Image_ID (從檔名中提取數字)
        image_id = int(re.search(r'\d+', img_path.name).group())

        prediction_strings = []

        # preds[0].boxes 包含了這張圖片所有的偵測結果
        boxes = preds[0].boxes

        # 遍歷每個偵測到的物件
        for i in range(len(boxes)):
            box = boxes[i]

            # 獲取信心度
            conf = box.conf.item()

            # 獲取類別 ID
            class_id = int(box.cls.item())

            # 獲取 Bounding Box 座標 (x_tl, y_tl, x_br, y_br)
            # 這些座標已经是絕對像素座標 (Denormalized)
            xyxy = box.xyxy[0].tolist()
            bb_left = xyxy[0]
            bb_top = xyxy[1]

            # 計算寬和高
            bb_width = xyxy[2] - xyxy[0]
            bb_height = xyxy[3] - xyxy[1]

            # 根據 PDF 第 11 頁的格式要求組合字串
            # 格式: <conf> <bb_left> <bb_top> <bb_width> <bb_height> <class>
            prediction_strings.append(
                f"{conf:.6f} {bb_left:.2f} {bb_top:.2f} {bb_width:.2f} {bb_height:.2f} {class_id}"
            )

        # 將單張圖片的所有預測結果用空格連接起來
        full_prediction_string = " ".join(prediction_strings)

        results_list.append({
            'Image_ID': image_id,
            'PredictionString': full_prediction_string
        })

    # 5. 將結果轉換為 DataFrame 並儲存為 CSV
    submission_df = pd.DataFrame(results_list)

    # 確保 Image_ID 是整數且已排序
    submission_df['Image_ID'] = submission_df['Image_ID'].astype(int)
    submission_df = submission_df.sort_values(
        by='Image_ID').reset_index(drop=True)

    # 儲存 CSV，格式為 Image_ID,PredictionString，中間沒有空格
    submission_df.to_csv(output_csv, index=False, sep=',')

    print("\n" + "="*30)
    print(f"預測完成！提交檔案已儲存至: {output_csv}")
    print("="*30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="為 CVPDL HW2 產生 Kaggle 提交檔案")
    parser.add_argument('--weights', type=str, required=True,
                        help="訓練好的模型權重路徑 (e.g., cvpdl_hw2_runs/yolov8n_scratch/weights/best.pt)")
    parser.add_argument('--data-path', type=str,
                        required=True, help="原始資料夾的路徑 (包含 test/ 子資料夾)")
    parser.add_argument('--output-csv', type=str,
                        default='submission.csv', help="輸出的 CSV 檔案名稱")
    parser.add_argument('--img-size', type=int,
                        default=960, help="輸入圖片尺寸 (應與訓練時相同)")
    parser.add_argument('--conf-thres', type=float,
                        default=0.2, help="信心度閾值，過濾掉低信心度的預測")

    args = parser.parse_args()

    # 組合出 test 資料夾的完整路徑
    test_directory = Path(args.data_path) / 'test'

    predict_and_generate_submission(args.weights, str(
        test_directory), args.output_csv, args.img_size, args.conf_thres)

    """
    python predict.py \
    --weights ./cvpdl_hw2_runs/yolov8m_scratch4/weights/best.pt \
    --data-path ./CVPDL_hw2/CVPDL_hw2 \
    --img-size 1920
    """
