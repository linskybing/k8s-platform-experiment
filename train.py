import time
from ultralytics import YOLO
from multiprocessing import Process
import os
import subprocess
import csv
import threading

# --------------------------
# Config
# --------------------------
MODEL_PATH = "yolo11n.pt"
DATA_DIR = "coco128"
DATA_YAML = os.path.join(DATA_DIR, "coco128.yaml")
EPOCHS = 1
LOG_DIR = "training_logs"
os.makedirs(LOG_DIR, exist_ok=True)

GPU_PRICE = {
    "RTX4070 SUPER": 19900,
    "RTX5070 TI": 26990,
}
CURRENT_GPU = "RTX5070 TI"
CSV_FILE = os.path.join(LOG_DIR, f"{CURRENT_GPU}.csv")

# --------------------------
# GPU 監控
# --------------------------
gpu_vals, mem_vals = [], []
stop_monitor = False

def get_gpu_utilization():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], encoding="utf-8"
        )
        gpu_util, mem_used, mem_total = map(float, result.strip().split(","))
        mem_util = (mem_used / mem_total) * 100 if mem_total > 0 else 0.0
        return gpu_util, mem_util
    except Exception:
        return 0.0, 0.0

def monitor_gpu(interval=0.05):  # 提高監控頻率
    global gpu_vals, mem_vals, stop_monitor
    while not stop_monitor:
        gpu, mem = get_gpu_utilization()
        gpu_vals.append(gpu)
        mem_vals.append(mem)
        time.sleep(interval)

# --------------------------
# 單用戶訓練 (完全不輸出)
# --------------------------
def train_user(user_id):
    model = YOLO(MODEL_PATH)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=1,
        device=0,
        imgsz=640,
        verbose=False,        # 完全不輸出
        save=False,           # 不存任何結果
        exist_ok=True,
        project="/tmp"        # 暫存目錄
    )
    return user_id

def run_user_process(user_id):
    """每個用戶進程的入口函式"""
    train_user(user_id)

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # CSV 標題 (移除 CP)
    with open(CSV_FILE, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "Users", "Total_Time(s)", "Avg_FPS_est",
            "GPU_Max(%)", "GPU_Avg(%)", "GPU_Min(%)",
            "Mem_Max(%)", "Mem_Avg(%)", "Mem_Min(%)",
            "GPU_Price"
        ])

    for NUM_USERS in range(1, 8):
        print(f"\nSimulating {NUM_USERS} users training...")

        # 啟動 GPU 監控 thread
        stop_monitor = False
        gpu_vals, mem_vals = [], []
        monitor_thread = threading.Thread(target=monitor_gpu)
        monitor_thread.start()

        # 使用 Process 啟動多用戶訓練
        start = time.time()
        processes = []
        for user_id in range(1, NUM_USERS + 1):
            p = Process(target=run_user_process, args=(user_id,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        end = time.time()

        # 停止 GPU 監控
        stop_monitor = True
        monitor_thread.join()

        elapsed = end - start
        avg_fps_est = NUM_USERS / elapsed if elapsed > 0 else 0.0

        # GPU 統計
        gpu_max = max(gpu_vals) if gpu_vals else 0.0
        gpu_avg = sum(gpu_vals) / len(gpu_vals) if gpu_vals else 0.0
        gpu_min = min(gpu_vals) if gpu_vals else 0.0
        mem_max = max(mem_vals) if mem_vals else 0.0
        mem_avg = sum(mem_vals) / len(mem_vals) if mem_vals else 0.0
        mem_min = min(mem_vals) if mem_vals else 0.0

        gpu_price = GPU_PRICE.get(CURRENT_GPU, 1.0)

        # 寫入 CSV (不寫 CP)
        with open(CSV_FILE, "a", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                NUM_USERS, round(elapsed, 2), round(avg_fps_est, 2),
                round(gpu_max, 2), round(gpu_avg, 2), round(gpu_min, 2),
                round(mem_max, 2), round(mem_avg, 2), round(mem_min, 2),
                gpu_price
            ])

    print("\nTraining performance measurement finished. CSV saved:", CSV_FILE)
