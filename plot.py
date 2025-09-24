import time
from ultralytics import YOLO
import cv2
from multiprocessing.dummy import Pool as ThreadPool
import os
import subprocess
import matplotlib.pyplot as plt
import csv
import glob
import numpy as np

# --------------------------
# Config
# --------------------------
IMAGE_PATH = "coco/val2017"   # single image path or directory
MODEL_PATH = "yolo11n.pt"
NUM_RUNS = 40
DATASET_NAME = "coco128"
LOG_DIR = f"inference_logs/{DATASET_NAME}"
os.makedirs(LOG_DIR, exist_ok=True)

GPU_PRICE = {
    "RTX5070TI": 1600,
}
CURRENT_GPU = "RTX5070TI"

SILENT = True
CSV_FILE = os.path.join(LOG_DIR, f"summary_{CURRENT_GPU}.csv")

MODEL = None

# --------------------------
# GPU Utilization
# --------------------------
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

# --------------------------
# Single user inference
# --------------------------
def user_inference(user_id):
    global MODEL
    if MODEL is None:
        raise RuntimeError("MODEL not initialized")

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
        _ = MODEL(img, verbose=False, show=False, save=False)

    # timed inference
    start = time.time()
    for _ in range(NUM_RUNS):
        _ = MODEL(img, verbose=False, show=False, save=False)
    end = time.time()

    elapsed = end - start
    avg_latency = elapsed / NUM_RUNS
    fps = NUM_RUNS / elapsed if elapsed > 0 else 0.0

    return user_id, elapsed, avg_latency, fps

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    MODEL = YOLO(MODEL_PATH)

    # create CSV header
    with open(CSV_FILE, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "Users", "Avg_Total_Time(s)", "Avg_Latency(s)", "Avg_FPS_per_user",
            "Total_Throughput_FPS", "GPU_Util(%)", "Mem_Util(%)", "GPU_Price"
        ])

    summary_rows = []

    # simulate 1~12 users
    for NUM_USERS in range(1, 13):
        log_file = os.path.join(LOG_DIR, f"inference_{NUM_USERS}_users.log")
        if not SILENT:
            print(f"\nSimulating {NUM_USERS} users. Log -> {log_file}")

        t_start = time.time()
        with ThreadPool(NUM_USERS) as pool:
            results = pool.map(user_inference, range(1, NUM_USERS + 1))
        t_end = time.time()
        total_elapsed_global = t_end - t_start

        total_images = NUM_USERS * NUM_RUNS
        total_fps = total_images / total_elapsed_global if total_elapsed_global > 0 else 0.0

        per_user_total_times = [r[1] for r in results]
        per_user_latencies = [r[2] for r in results]
        per_user_fps = [r[3] for r in results]

        avg_total_time = sum(per_user_total_times) / NUM_USERS
        avg_latency = sum(per_user_latencies) / NUM_USERS
        avg_fps = sum(per_user_fps) / NUM_USERS
        max_diff = max(per_user_total_times) - min(per_user_total_times)

        gpu_util, mem_util = get_gpu_utilization()
        gpu_price = GPU_PRICE.get(CURRENT_GPU, 1.0)

        # write log file
        with open(log_file, "w") as flog:
            flog.write(f"Simulated {NUM_USERS} users inference\n")
            for r in sorted(results, key=lambda x: x[0]):
                flog.write(f"User {r[0]}: Total={r[1]:.2f}s, Avg latency={r[2]:.4f}s, FPS={r[3]:.2f}\n")
            flog.write(f"Average total time: {avg_total_time:.2f}s\n")
            flog.write(f"Max total time difference: {max_diff:.2f}s\n")
            flog.write(f"Total throughput (FPS): {total_fps:.2f}\n")
            flog.write(f"GPU Utilization: {gpu_util:.2f}% | Memory Utilization: {mem_util:.2f}%\n")

        # append CSV
        with open(CSV_FILE, "a", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([NUM_USERS, avg_total_time, avg_latency, avg_fps,
                             total_fps, gpu_util, mem_util, gpu_price])

        summary_rows.append((NUM_USERS, avg_latency, avg_fps, total_fps, gpu_util))

        if not SILENT:
            for r in sorted(results, key=lambda x: x[0]):
                print(f"User {r[0]}: Total={r[1]:.2f}s, Avg latency={r[2]:.4f}s, FPS={r[3]:.2f}")
            print(f"Average total time: {avg_total_time:.2f}s")
            print(f"Max total time difference: {max_diff:.2f}s")
            print(f"Total throughput (FPS): {total_fps:.2f}")
            print(f"GPU Utilization: {gpu_util:.2f}% | Memory Utilization: {mem_util:.2f}%")

    # --------------------------
    # Check all CSVs in log dir
    # --------------------------
    csv_files = glob.glob(os.path.join(LOG_DIR, "*.csv"))
    all_data = {}

    for csv_file in csv_files:
        user_label = os.path.basename(csv_file).replace(".csv", "")
        users, avg_latency, avg_fps_per_user, total_fps, gpu_utils = [], [], [], [], []

        with open(csv_file, "r") as fcsv:
            reader = csv.DictReader(fcsv)
            for row in reader:
                users.append(int(row["Users"]))
                avg_latency.append(float(row["Avg_Latency(s)"]))
                avg_fps_per_user.append(float(row["Avg_FPS_per_user"]))
                total_fps.append(float(row["Total_Throughput_FPS"]))
                gpu_utils.append(float(row["GPU_Util(%)"]))
        all_data[user_label] = {
            "users": users,
            "avg_latency": avg_latency,
            "avg_fps_per_user": avg_fps_per_user,
            "total_fps": total_fps,
            "gpu_utils": gpu_utils
        }

    # --------------------------
    # Plot: Grouped bar chart with value labels (2 decimal places)
    # --------------------------
    metrics = ["total_fps", "avg_latency", "gpu_utils"]
    titles = ["Total Throughput (FPS)", "Average Latency (s)", "GPU Utilization (%)"]
    ylabel = ["FPS", "s", "%"]

    plt.figure(figsize=(16, 12))
    x = np.arange(len(summary_rows))
    n_users = len(all_data)
    width = 0.8 / max(1, n_users)  # 自動調整柱寬

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        for idx, (label, data) in enumerate(all_data.items()):
            bars = plt.bar(x + idx*width, data[metric], width=width, label=label)
            # 加上柱頂數值 (2位小數)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', 
                         ha='center', va='bottom', fontsize=8)
        plt.xticks(x + width*(n_users-1)/2, [r[0] for r in summary_rows])
        plt.xlabel("Number of Users")
        plt.ylabel(ylabel[i])
        plt.title(titles[i])
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, f"all_users_comparison_{CURRENT_GPU}.png"))
    if not SILENT:
        plt.show()
