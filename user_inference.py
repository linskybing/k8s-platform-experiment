import sys
import time
from ultralytics import YOLO
import cv2
import os

IMAGE_PATH = "coco/images/val2017"
MODEL_PATH = "yolo11n.pt"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python user_inference.py <user_id> <num_runs>")
        sys.exit(1)

    user_id = int(sys.argv[1])
    NUM_RUNS = int(sys.argv[2])

    model = YOLO(MODEL_PATH)

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

    for _ in range(5):
        _ = model(img, verbose=False, show=False, save=False)

    start = time.time()
    for _ in range(NUM_RUNS):
        _ = model(img, verbose=False, show=False, save=False)
    end = time.time()

    elapsed = end - start
    avg_latency = elapsed / NUM_RUNS
    avg_fps = NUM_RUNS / elapsed if elapsed > 0 else 0.0

    print(f"[User {user_id}] Total Time: {elapsed:.4f}s | Avg Latency: {avg_latency:.6f}s | Avg FPS: {avg_fps:.2f}")
