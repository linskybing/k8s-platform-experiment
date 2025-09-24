import time
from ultralytics import YOLO
import cv2
from multiprocessing import Pool
import os
import subprocess
import csv
import threading

# --------------------------
# Config
# --------------------------
IMAGE_PATH = "coco/images/val2017"
MODEL_PATH = "yolo11n.pt"
NUM_RUNS = 40
DATASET_NAME = "coco128"
LOG_DIR = f"inference_logs/{DATASET_NAME}"
os.makedirs(LOG_DIR, exist_ok=True)

GPU_PRICE = {
    "RTX4070 SUPER": 19990,
    "RTX5070 TI": 1600
}
CURRENT_GPU = "RTX4070 SUPER"
CSV_FILE = os.path.join(LOG_DIR, f"{CURRENT_GPU}.csv")
SILENT = False  # False 打印完整結果

# --------------------------
# GPU Utilization (即時監控)
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

def monitor_gpu(interval=0.01):
    global gpu_vals, mem_vals, stop_monitor
    while not stop_monitor:
        gpu, mem = get_gpu_utilization()
        gpu_vals.append(gpu)
        mem_vals.append(mem)
        time.sleep(interval)

# --------------------------
# Single process inference (專注推論)
# --------------------------
def user_inference(user_id):
    model = YOLO(MODEL_PATH)

    # load image
    if os.path.isdir(IMAGE_PATH):
        images = [os.path.join(IMAGE_PATH, f) for f in os.listdir(IMAGE_PATH)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            raise ValueError(f"No images found in {IMAGE_PATH}")
        img = cv2.imread(images[0])
    else:
        img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise ValueError(f"Failed to read image {IMAGE_PATH}")

    # warmup
    for _ in range(5):
        _ = model(img, verbose=False, show=False, save=False)

    # timed inference
    start = time.time()
    for _ in range(NUM_RUNS):
        _ = model(img, verbose=False, show=False, save=False)
    end = time.time()

    elapsed = end - start
    avg_latency = elapsed / NUM_RUNS
    avg_fps = NUM_RUNS / elapsed if elapsed > 0 else 0.0

    # 只輸出推論耗時
    if not SILENT:
        print(f"[User {user_id}] Total Time: {elapsed:.4f}s | Avg Latency: {avg_latency:.6f}s | Avg FPS: {avg_fps:.2f}")

    return user_id, elapsed, avg_latency, avg_fps

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # CSV header
    with open(CSV_FILE, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "Users", "Avg_Total_Time(s)", "Avg_Latency(s)", "Avg_FPS_per_user",
            "GPU_Max(%)", "GPU_Avg(%)", "GPU_Min(%)",
            "Mem_Max(%)", "Mem_Avg(%)", "Mem_Min(%)",
            "GPU_Price"
        ])

    summary_rows = []

    for NUM_USERS in range(1, 13):
        print(f"\n=== Simulating {NUM_USERS} Users ===")

        # --- 啟動 GPU 監控 thread ---
        stop_monitor = False
        gpu_vals, mem_vals = [], []
        monitor_thread = threading.Thread(target=monitor_gpu)
        monitor_thread.start()

        # --- 推論 ---
        t_start = time.time()
        with Pool(NUM_USERS) as pool:
            results = pool.map(user_inference, range(1, NUM_USERS + 1))
        t_end = time.time()

        # --- 停止 GPU 監控並 join ---
        stop_monitor = True
        monitor_thread.join()

        # --- GPU 統計 ---
        gpu_max = max(gpu_vals) if gpu_vals else 0.0
        gpu_avg = sum(gpu_vals)/len(gpu_vals) if gpu_vals else 0.0
        gpu_min = min(gpu_vals) if gpu_vals else 0.0
        mem_max = max(mem_vals) if mem_vals else 0.0
        mem_avg = sum(mem_vals)/len(mem_vals) if mem_vals else 0.0
        mem_min = min(mem_vals) if mem_vals else 0.0
        gpu_price = GPU_PRICE.get(CURRENT_GPU, 1.0)

        # --- 推論統計 ---
        avg_total_time = sum(r[1] for r in results) / NUM_USERS
        avg_latency = sum(r[2] for r in results) / NUM_USERS
        avg_fps = sum(r[3] for r in results) / NUM_USERS

        # --- CSV 寫入 ---
        with open(CSV_FILE, "a", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([NUM_USERS, avg_total_time, avg_latency, avg_fps,
                             gpu_max, gpu_avg, gpu_min,
                             mem_max, mem_avg, mem_min,
                             gpu_price])

        summary_rows.append((NUM_USERS, avg_latency, avg_fps, gpu_avg))

        # --- 印 summary ---
        if not SILENT:
            print(f"\n[Summary] Avg Total Time: {avg_total_time:.4f}s | Avg Latency: {avg_latency:.6f}s | Avg FPS: {avg_fps:.2f}")
            print(f"GPU Utilization Avg: {gpu_avg:.2f}% | Memory Utilization Avg: {mem_avg:.2f}%")
