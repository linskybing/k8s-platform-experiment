import time
from ultralytics import YOLO
from multiprocessing import Process, Queue
import os
import subprocess
import matplotlib.pyplot as plt
import csv

# --------------------------
# Config
# --------------------------
MODEL_PATH = "yolo11n.pt"
DATA_PATH = "coco128.yaml"
EPOCHS = 1
LOG_DIR = "training_logs"
os.makedirs(LOG_DIR, exist_ok=True)

GPU_PRICE = {
    "RTX4090": 1600,
    "RTX4080": 1200,
    "A100": 15000,
    "H100": 30000
}
CURRENT_GPU = "RTX4090"
SILENT = True
CSV_FILE = os.path.join(LOG_DIR, "training_summary.csv")


# --------------------------
# GPU utility
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
# Worker function
# --------------------------
def train_user(user_id, queue):
    """Train YOLO for a single user and put result into queue."""
    model = YOLO(MODEL_PATH)

    start = time.time()
    model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        batch=1,
        device=0,   # 確保單 GPU
        imgsz=640,
        verbose=False
    )
    end = time.time()
    elapsed = end - start

    try:
        steps_per_epoch = model.trainer.data_loader_len()
    except Exception:
        steps_per_epoch = 1
    fps_estimate = (EPOCHS * steps_per_epoch) / elapsed if elapsed > 0 else 0.0

    queue.put((user_id, elapsed, fps_estimate))


# --------------------------
# Main flow
# --------------------------
if __name__ == "__main__":
    # Prepare CSV
    with open(CSV_FILE, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "Users", "Avg_Time(s)", "Avg_FPS_est", "Max_Time_Diff(s)",
            "GPU_Util(%)", "Mem_Util(%)", "GPU_Price", "CP(FPS/$)"
        ])

    summary_rows = []

    for NUM_USERS in range(1, 8):
        log_file = os.path.join(LOG_DIR, f"training_{NUM_USERS}_users.log")
        if not SILENT:
            print(f"\nSimulating {NUM_USERS} users training. Log -> {log_file}")

        # Multiprocessing
        queue = Queue()
        processes = []

        for uid in range(1, NUM_USERS + 1):
            p = Process(target=train_user, args=(uid, queue))
            p.start()
            processes.append(p)

        results = []
        for _ in range(NUM_USERS):
            results.append(queue.get())

        for p in processes:
            p.join()

        # Aggregate metrics
        total_times = [r[1] for r in results]
        avg_time = sum(total_times) / NUM_USERS
        max_diff = max(total_times) - min(total_times)
        avg_fps = sum(r[2] for r in results) / NUM_USERS

        gpu_util, mem_util = get_gpu_utilization()
        gpu_price = GPU_PRICE.get(CURRENT_GPU, 1.0)
        cp_value = avg_fps / gpu_price if gpu_price else 0.0

        # Write log
        with open(log_file, "w") as flog:
            flog.write(f"Simulated {NUM_USERS} users training\n")
            for uid, elapsed, fps_est in sorted(results, key=lambda x: x[0]):
                flog.write(f"User {uid}: Time={elapsed:.2f}s, Est FPS={fps_est:.2f}\n")
            flog.write(f"Average Time: {avg_time:.2f}s\n")
            flog.write(f"Max Time Difference: {max_diff:.2f}s\n")
            flog.write(f"Avg FPS (estimate): {avg_fps:.2f}\n")
            flog.write(f"GPU Utilization: {gpu_util:.2f}% | Memory Utilization: {mem_util:.2f}%\n")
            flog.write(f"Cost-Performance (FPS/$): {cp_value:.6f}\n")

        # CSV summary
        with open(CSV_FILE, "a", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([NUM_USERS, avg_time, avg_fps, max_diff,
                             gpu_util, mem_util, gpu_price, cp_value])

        summary_rows.append((NUM_USERS, avg_time, avg_fps, gpu_util, cp_value))

    # --------------------------
    # Plot results
    # --------------------------
    users = [r[0] for r in summary_rows]
    avg_times = [r[1] for r in summary_rows]
    avg_fps_list = [r[2] for r in summary_rows]
    gpu_utils = [r[3] for r in summary_rows]
    cp_vals = [r[4] for r in summary_rows]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(users, avg_fps_list, marker="o")
    plt.title("Estimated Training Throughput (FPS) vs Users")
    plt.xlabel("Number of Users")
    plt.ylabel("Estimated FPS")

    plt.subplot(2, 2, 2)
    plt.plot(users, avg_times, marker="o", color="orange")
    plt.title("Average Training Time vs Users")
    plt.xlabel("Number of Users")
    plt.ylabel("Time (s)")

    plt.subplot(2, 2, 3)
    plt.plot(users, gpu_utils, marker="o", color="green")
    plt.title("GPU Utilization vs Users")
    plt.xlabel("Number of Users")
    plt.ylabel("GPU Util (%)")

    plt.subplot(2, 2, 4)
    plt.plot(users, cp_vals, marker="o", color="red")
    plt.title("Training Cost-Performance (FPS/$) vs Users")
    plt.xlabel("Number of Users")
    plt.ylabel("CP (FPS / $)")

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, f"training_summary_{CURRENT_GPU}.png"))
    if not SILENT:
        plt.show()
