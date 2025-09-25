import os
import time
import csv
import threading
import subprocess
import re
import pynvml
from dotenv import load_dotenv

NUM_RUNS = 10
DATASET_NAME = "coco128"
LOG_DIR = f"inference_logs/{DATASET_NAME}"
os.makedirs(LOG_DIR, exist_ok=True)

load_dotenv()

GPU_PRICE = {
    "RTX4070 SUPER": int(os.getenv("GPU_PRICE_RTX4070_SUPER", 0)),
    "RTX5070 TI": int(os.getenv("GPU_PRICE_RTX5070_TI", 0)),
    "RTX2080 TI": int(os.getenv("GPU_PRICE_RTX2080_TI", 0)),
}
CURRENT_GPU = os.getenv("CURRENT_GPU", "RTX5070 TI")
CSV_FILE = os.path.join(LOG_DIR, f"{CURRENT_GPU}.csv")

SILENT = False
USE_MPS_LIMIT = False

gpu_vals, mem_vals, power_vals = [], [], []
stop_monitor = False

header = ["Users", "Total_Time(s)", "Avg_Latency(s)", "Avg_YOLO_FPS", "Aggregate_FPS",
          "GPU_Max(%)", "GPU_Avg(%)", "GPU_Min(%)",
          "Mem_Max(%)", "Mem_Avg(%)", "Mem_Min(%)",
          "Power_Max(W)", "Power_Avg(W)", "Power_Min(W)",
          "MPS_Limit(%)", "GPU_Price"]

# ---------------- CSV Map ---------------- #
def read_csv_to_map(file_path):
    data_map = {}
    if os.path.exists(file_path):
        with open(file_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = int(row["Users"])
                data_map[key] = row
    return data_map

def write_map_to_csv(file_path, data_map):
    rows = [header]
    for k in sorted(data_map.keys()):
        row = [data_map[k][col] for col in header]
        rows.append(row)
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

# ---------------- GPU Monitor ---------------- #
def get_gpu_utilization_nvml(handle):
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    gpu_util = util.gpu
    mem_util = (mem.used / mem.total) * 100 if mem.total > 0 else 0.0
    return gpu_util, mem_util, power

def monitor_gpu(handle, interval=0.5):
    global gpu_vals, mem_vals, power_vals, stop_monitor
    while not stop_monitor:
        gpu, mem, power = get_gpu_utilization_nvml(handle)
        gpu_vals.append(gpu)
        mem_vals.append(mem)
        power_vals.append(power)
        time.sleep(interval)

# ---------------- Map Update ---------------- #
def update_map(data_map, num_users, silent=SILENT):
    global gpu_vals, mem_vals, power_vals, stop_monitor

    USER_LIMITS = {i+1: 100 / num_users for i in range(num_users)}

    stop_monitor = False
    gpu_vals, mem_vals, power_vals = [], [], []
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
            ["python", "user_inference.py", str(user_id), str(NUM_RUNS)],
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
    power_max = max(power_vals) if power_vals else 0.0
    power_avg = sum(power_vals)/len(power_vals) if power_vals else 0.0
    power_min = min(power_vals) if power_vals else 0.0
    gpu_price = GPU_PRICE.get(CURRENT_GPU, 1.0)

    avg_total_time = sum(r[0] for r in results) / num_users
    avg_latency = sum(r[1] for r in results) / num_users
    avg_fps = sum(r[2] for r in results) / num_users
    aggregate_fps = sum(r[2] for r in results)
    mps_limit = sum(USER_LIMITS.values()) / num_users if USE_MPS_LIMIT else 0

    new_row = {
        "Users": num_users,
        "Total_Time(s)": round(avg_total_time,2),
        "Avg_Latency(s)": round(avg_latency,4),
        "Avg_YOLO_FPS": round(avg_fps,2),
        "Aggregate_FPS": round(aggregate_fps,2),
        "GPU_Max(%)": round(gpu_max,2),
        "GPU_Avg(%)": round(gpu_avg,2),
        "GPU_Min(%)": round(gpu_min,2),
        "Mem_Max(%)": round(mem_max,2),
        "Mem_Avg(%)": round(mem_avg,2),
        "Mem_Min(%)": round(mem_min,2),
        "Power_Max(W)": round(power_max,2),
        "Power_Avg(W)": round(power_avg,2),
        "Power_Min(W)": round(power_min,2),
        "MPS_Limit(%)": round(mps_limit,2),
        "GPU_Price": gpu_price
    }

    data_map[num_users] = new_row

    if not silent:
        print(f"Users: {num_users} | Avg Total Time: {avg_total_time:.4f}s | "
              f"Avg Latency: {avg_latency:.6f}s | Avg FPS: {avg_fps:.2f} | "
              f"Aggregate FPS: {aggregate_fps:.2f} | GPU Avg: {gpu_avg:.2f}% | Mem Avg: {mem_avg:.2f}% | "
              f"Power_Avg(W): {power_avg:.2f}W")

# ---------------- Main ---------------- #
if __name__ == "__main__":
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    data_map = read_csv_to_map(CSV_FILE)

    for num_users in range(1, 13):
        update_map(data_map, num_users, silent=SILENT)

    write_map_to_csv(CSV_FILE, data_map)

    pynvml.nvmlShutdown()
