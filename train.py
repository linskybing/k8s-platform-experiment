import time
from ultralytics import YOLO
from multiprocessing import Process, Manager
import os
import subprocess
import csv
import threading
import shutil
import glob
import cv2
import pynvml

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

IMAGE_DIR = "coco/images/val2017"
IMGS = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))

gpu_vals, mem_vals = [], []
stop_monitor = False

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

def inference_fps(model, image_paths, repeat=1):
    total_images = len(image_paths) * repeat
    start = time.time()
    for _ in range(repeat):
        for img_path in image_paths:
            img = cv2.imread(img_path)
            _ = model(img)
    end = time.time()
    elapsed = end - start
    fps = total_images / elapsed if elapsed > 0 else 0.0
    return fps

def train_user(user_id, return_dict):
    tmp_model_path = f"/tmp/yolo11n_user{user_id}.pt"
    shutil.copy(MODEL_PATH, tmp_model_path)
    project_dir = f"/tmp/train_user{user_id}"
    model = YOLO(tmp_model_path)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=5,
        device=0,
        imgsz=640,
        verbose=False,
        save=False,
        exist_ok=True,
        project=project_dir
    )
    fps = inference_fps(model, IMGS)
    return_dict[user_id] = fps

def run_user_process(user_id, return_dict):
    train_user(user_id, return_dict)

if __name__ == "__main__":
    manager = Manager()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    with open(CSV_FILE, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "Users", "Total_Time(s)", "Avg_YOLO_FPS",
            "GPU_Max(%)", "GPU_Avg(%)", "GPU_Min(%)",
            "Mem_Max(%)", "Mem_Avg(%)", "Mem_Min(%)",
            "GPU_Price"
        ])

    for NUM_USERS in range(1, 12):
        print(f"\nSimulating {NUM_USERS} users training...")

        stop_monitor = False
        gpu_vals, mem_vals = [], []
        monitor_thread = threading.Thread(target=monitor_gpu, args=(handle, 0.5))
        monitor_thread.start()

        start = time.time()
        processes = []
        return_dict = manager.dict()

        for user_id in range(1, NUM_USERS + 1):
            p = Process(target=run_user_process, args=(user_id, return_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        end = time.time()
        stop_monitor = True
        monitor_thread.join()

        elapsed = end - start

        avg_fps = sum(return_dict.values()) / len(return_dict) if return_dict else 0.0

        gpu_max = max(gpu_vals) if gpu_vals else 0.0
        gpu_avg = sum(gpu_vals) / len(gpu_vals) if gpu_vals else 0.0
        gpu_min = min(gpu_vals) if gpu_vals else 0.0
        mem_max = max(mem_vals) if mem_vals else 0.0
        mem_avg = sum(mem_vals) / len(mem_vals) if mem_vals else 0.0
        mem_min = min(mem_vals) if mem_vals else 0.0

        gpu_price = GPU_PRICE.get(CURRENT_GPU, 1.0)

        with open(CSV_FILE, "a", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                NUM_USERS, round(elapsed, 2), round(avg_fps, 2),
                round(gpu_max, 2), round(gpu_avg, 2), round(gpu_min, 2),
                round(mem_max, 2), round(mem_avg, 2), round(mem_min, 2),
                gpu_price
            ])

    print("\nTraining + YOLO FPS measurement finished. CSV saved:", CSV_FILE)
    pynvml.nvmlShutdown()