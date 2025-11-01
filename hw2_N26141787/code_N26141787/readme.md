# 以下簡介操作流程以完成復現
### server是Linux環境，使用bash terminal

* Step 1. 將 taica-cvpdl-2025-hw-2.zip 放入server後 執行 "unzip taica-cvpdl-2025-hw-2.zip"
* Step 2. python3 -m venv .venv (建立.venv folder)　-> source .venv/bin/activate (啟動虛擬環境) (python版本為3.11.10)
* Step 3. #####pip install --upgrade pip
          #####pip install opencv-python-headless tqdm ultralytics pyyaml
          #####pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121p
          #####pip install pandas

### 一共會有三個檔案 -> train.py、visualize_test.py、predict.py 
### 其中 train.py是負責主要訓練的，訓練完會儲存為best.pt ；　visualize_test.py是將訓練完後的模型進行object prediction 的 bounding box視覺化 ； predict.py是負責產出Kaggle競賽要求的CSV

* train.py 欲執行可輸入 "python train.py" 如有需更改超參數亦可輸入: 
    python train.py \
    --data-path ./CVPDL_hw2/CVPDL_hw2 \
    --epochs 200 \
    --batch-size 2 \
    --model-yaml yolov8m.yaml \
    --img-size 1920
#### train.py run完之後 會到 .\cvpdl_hw2_runs\

* visualize_test.py 欲執行可輸入: 
    python visualize_test.py \
    --weights ./cvpdl_hw2_runs/yolov8m_scratch4/weights/best.pt \
    --data-path ./CVPDL_hw2/CVPDL_hw2 \
    --num-images 30

* predict.py 欲直行可輸入:
    python predict.py \
    --weights ./cvpdl_hw2_runs/yolov8m_scratch4/weights/best.pt \
    --data-path ./CVPDL_hw2/CVPDL_hw2 \
    --img-size 1920
#### visualize_test.py 和 predict.py 請注意路徑! 因為每一次執行train.py會產生出一個新的folder，請根據欲觀測指定訓練完模型的預測結果的路徑進行修改
#### 除此之外，predict.py還要注意imgsize的部分，要和train.py的imgsize一樣才會產生出對應的結果，否則預測時沒有辦法達到真正的結果


### 如果有任何問題無法復現或是操作上有問題請寄信: N26141787@gs.ncku.edu.tw
