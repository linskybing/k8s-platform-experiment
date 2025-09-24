import time
from ultralytics import YOLO
import cv2
import multiprocessing
import os

# --------------------------
# Config
# --------------------------
IMAGE_PATH = "coco/val2017"  # 單張或資料夾
MODEL_PATH = "yolo11n.pt"
NUM_RUNS = 40
DATASET_NAME = "coco128"
LOG_DIR = f"inference_logs/{DATASET_NAME}"
os.makedirs(LOG_DIR, exist_ok=True)

def user_inference(user_id):
    model = YOLO(MODEL_PATH)

    if os.path.isdir(IMAGE_PATH):
        images = [os.path.join(IMAGE_PATH, f) for f in os.listdir(IMAGE_PATH) if f.endswith(('.jpg', '.png'))]
        if not images:
            raise ValueError(f"No images found in {IMAGE_PATH}")
        img = cv2.imread(images[0])
    else:
        img = cv2.imread(IMAGE_PATH)

    # Warmup
    for _ in range(5):
        _ = model(img)

    # Timed inference
    start = time.time()
    for _ in range(NUM_RUNS):
        _ = model(img)
    end = time.time()

    elapsed = end - start
    avg_latency = elapsed / NUM_RUNS
    fps = NUM_RUNS / elapsed

    return user_id, elapsed, avg_latency, fps

if __name__ == "__main__":
    for NUM_USERS in range(1, 13):
        log_file = os.path.join(LOG_DIR, f"inference_{NUM_USERS}_users.log")
        print(f"\nSimulating {NUM_USERS} users. Log -> {log_file}")

        global_start = time.time()
        with multiprocessing.Pool(NUM_USERS) as pool:
            results = pool.map(user_inference, range(1, NUM_USERS + 1))
        global_end = time.time()
        global_elapsed = global_end - global_start
        
        total_images = NUM_USERS * NUM_RUNS
        total_fps = total_images / global_elapsed

        total_elapsed_list = [r[1] for r in results]
        avg_latency_list = [r[2] for r in results]
        fps_list = [r[3] for r in results]

        avg_time = sum(total_elapsed_list) / NUM_USERS
        max_diff = max(total_elapsed_list) - min(total_elapsed_list)

        # 寫入 log
        with open(log_file, "w") as f:
            f.write(f"Simulated {NUM_USERS} users inference\n")
            for r in sorted(results, key=lambda x: x[0]):
                f.write(f"User {r[0]}: Total={r[1]:.2f}s, Avg latency={r[2]:.4f}s, FPS={r[3]:.2f}\n")
            f.write(f"Average total time: {avg_time:.2f}s\n")
            f.write(f"Max total time difference: {max_diff:.2f}s\n")
            f.write(f"Total throughput (FPS): {total_fps:.2f}\n")

        # 控制台打印
        for r in sorted(results, key=lambda x: x[0]):
            print(f"User {r[0]}: Total={r[1]:.2f}s, Avg latency={r[2]:.4f}s, FPS={r[3]:.2f}")
        print(f"Average total time: {avg_time:.2f}s")
        print(f"Max total time difference: {max_diff:.2f}s")
        print(f"Total throughput (FPS): {total_fps:.2f}")
