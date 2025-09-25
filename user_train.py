import sys
import os
import shutil
import time
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "yolo11n.pt"
DATA_DIR = "coco128"
DATA_YAML = os.path.join(DATA_DIR, "coco128.yaml")
EPOCHS = 1
BATCH_SIZE = 5
IMG_SIZE = 640

def train_user(user_id):
    tmp_model_path = f"/tmp/yolo11n_user{user_id}.pt"
    shutil.copy(MODEL_PATH, tmp_model_path)
    project_dir = f"/tmp/train_user{user_id}"

    model = YOLO(tmp_model_path)

    num_images = 128 * EPOCHS 

    start_time = time.time()
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=0,
        imgsz=IMG_SIZE,
        verbose=False,
        save=False,
        exist_ok=True,
        project=project_dir
    )
    elapsed = time.time() - start_time

    avg_latency = elapsed
    fps = num_images / elapsed if elapsed > 0 else 0.0

    print(f"[User {user_id}] Total Time: {elapsed:.3f}s | Avg Latency: {avg_latency:.6f}s | Avg FPS: {fps:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python user_train.py <user_id>")
        sys.exit(1)

    user_id = int(sys.argv[1])
    train_user(user_id)
