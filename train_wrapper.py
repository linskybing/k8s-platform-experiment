import os
import glob
import time
import csv
import threading
import subprocess
import re
import pynvml

LOG_DIR = "training_logs"
os.makedirs(LOG_DIR, exist_ok=True)

GPU_PRICE = {
    "RTX4070 SUPER": 19900,
    "RTX5070 TI": 26990,
    "RTX2080 TI": 26700,
}
CURRENT_GPU = "RTX5070 TI"
CSV_FILE = os.path.join(LOG_DIR, f"{CURRENT_GPU}.csv")

IMAGE_DIR = "coco/images/val2017"
IMGS = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))

USE_MPS_LIMIT = True
SILENT = False
gpu_vals, mem_vals, stop_monitor = [], [], False

header = [
    "Users", "MPS_Limit(%)", "Total_Time(s)", "Avg_Latency(s)", "Avg_YOLO_FPS", "Aggregate_FPS",
    "GPU_Max(%)", "GPU_Avg(%)", "GPU_Min(%)",
    "Mem_Max(%)", "Mem_Avg(%)", "Mem_Min(%)",
    "GPU_Price"
]

# ---------------- CSV Map ---------------- #
def read_csv_to_map(file_path):
    data_map = {}
    if os.path.exists(file_path):
        with open(file_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    key = int(row["Users"])
                    data_map[key] = row
                except:
                    continue
    return data_map

def write_map_to_csv(file_path, data_map):
    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for k in sorted(data_map.keys()):
            writer.writerow(data_map[k])

# ---------------- GPU Monitor ---------------- #
def get_gpu_utilization_nvml(handle):
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_util = util.gpu
    mem_util = (mem.used / mem.total) * 100 if mem.total > 0 else 0.0
    return gpu_util, mem_util

def monitor_gpu(handle, interval=0.5):
    global gpu_vals, mem_vals, stop_monitor
    while not stop_monitor:
        gpu, mem = get_gpu_utilization_nvml(handle)
        gpu_vals.append(gpu)
        mem_vals.append(mem)
        time.sleep(interval)

# ---------------- Map Update ---------------- #
def update_map(data_map, num_users, handle, silent=SILENT):
    global gpu_vals, mem_vals, stop_monitor

    USER_LIMITS = {i+1: 100 / num_users for i in range(num_users)} if USE_MPS_LIMIT else {}

    stop_monitor = False
    gpu_vals, mem_vals = [], []
    monitor_thread = threading.Thread(target=monitor_gpu, args=(handle, 0.5))
    monitor_thread.start()

    procs = []
    for user_id, limit in USER_LIMITS.items():
        env = os.environ.copy()
        if USE_MPS_LIMIT:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(limit)
        else:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(100)
        p = subprocess.Popen(
            ["python", "user_train.py", str(user_id)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        procs.append((user_id, limit, p))

    results = []
    for user_id, limit, p in procs:
        stdout, stderr = p.communicate()
        match = re.search(
            r"\[User (\d+)\] Total Time: ([\d.]+)s \| Avg Latency: ([\d.]+)s \| Avg FPS: ([\d.]+)",
            stdout
        )
        if match:
            total_time = float(match.group(2))
            avg_latency = float(match.group(3))
            avg_fps = float(match.group(4))
            results.append((total_time, avg_latency, avg_fps))

    stop_monitor = True
    monitor_thread.join()

    gpu_max = max(gpu_vals) if gpu_vals else 0.0
    gpu_avg = sum(gpu_vals)/len(gpu_vals) if gpu_vals else 0.0
    gpu_min = min(gpu_vals) if gpu_vals else 0.0
    mem_max = max(mem_vals) if mem_vals else 0.0
    mem_avg = sum(mem_vals)/len(mem_vals) if mem_vals else 0.0
    mem_min = min(mem_vals) if mem_vals else 0.0
    gpu_price = GPU_PRICE.get(CURRENT_GPU, 1.0)

    total_time_avg = sum(r[0] for r in results) / num_users
    avg_latency_val = sum(r[1] for r in results) / num_users
    avg_fps_val = sum(r[2] for r in results) / num_users
    aggregate_fps = sum(r[2] for r in results)
    mps_limit_avg = sum(USER_LIMITS.values()) / num_users if USE_MPS_LIMIT else 0

    new_row = {
        "Users": num_users,
        "MPS_Limit(%)": round(mps_limit_avg,2),
        "Total_Time(s)": round(total_time_avg,2),
        "Avg_Latency(s)": round(avg_latency_val,4),
        "Avg_YOLO_FPS": round(avg_fps_val,2),
        "Aggregate_FPS": round(aggregate_fps,2),
        "GPU_Max(%)": round(gpu_max,2),
        "GPU_Avg(%)": round(gpu_avg,2),
        "GPU_Min(%)": round(gpu_min,2),
        "Mem_Max(%)": round(mem_max,2),
        "Mem_Avg(%)": round(mem_avg,2),
        "Mem_Min(%)": round(mem_min,2),
        "GPU_Price": gpu_price
    }

    data_map[num_users] = new_row

    if not silent:
        print(f"Users: {num_users} | Total_Time: {total_time_avg:.2f}s | "
              f"Avg_FPS: {avg_fps_val:.2f} | Aggregate_FPS: {aggregate_fps:.2f} | "
              f"GPU Avg: {gpu_avg:.2f}% | Mem Avg: {mem_avg:.2f}%")

# ---------------- Main ---------------- #
if __name__ == "__main__":
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    data_map = read_csv_to_map(CSV_FILE)

    for NUM_USERS in range(1, 13):
        update_map(data_map, NUM_USERS, handle, silent=SILENT)

    write_map_to_csv(CSV_FILE, data_map)

    pynvml.nvmlShutdown()
