import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import csv
from dotenv import load_dotenv

LOG_DIR = "inference_logs"
OUTPUT_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
load_dotenv()

# --------------------------
# GPU 價格 (USD)
# --------------------------
gpu_price = {
    "RTX4070 SUPER": int(os.getenv("GPU_PRICE_RTX4070_SUPER", 0)),
    "RTX5070 TI": int(os.getenv("GPU_PRICE_RTX5070_TI", 0)),
    "RTX2080 TI": int(os.getenv("GPU_PRICE_RTX2080_TI", 0)),
    "RTX3090": 46900
}

# --------------------------
# 讀取 CSV
# --------------------------
all_data = {}
csv_files = glob.glob(os.path.join(LOG_DIR, "*.csv"))

for csv_file in csv_files:
    label = os.path.basename(csv_file).replace(".csv", "")
    users, agg_fps, avg_power = [], [], []

    with open(csv_file, "r") as fcsv:
        reader = csv.DictReader(fcsv)
        for row in reader:
            u = int(row["Users"])
            if u > 13:
                continue
            users.append(u)
            agg_fps.append(float(row.get("Aggregate_FPS", 0)))
            avg_power.append(float(row.get("Power_Avg(W)", 0)))

    if users:
        all_data[label] = {
            "users": users,
            "agg_fps": agg_fps,
            "avg_power": avg_power
        }

# --------------------------
# 計算 FPS/Price、Price/FPS、W/FPS (最後一個 row)
# --------------------------
gpu_order = ["RTX3090", "RTX5070 TI", "RTX4070 SUPER", "RTX2080 TI"]
gpu_colors = {
    "RTX3090": "#1f77b4",
    "RTX5070 TI": "#ff7f0e",
    "RTX4070 SUPER": "#2ca02c",
    "RTX2080 TI": "#d62728"
}

fps_per_price = []
price_per_fps = []
w_per_fps = []

for gpu in gpu_order:
    if gpu not in all_data:
        fps_per_price.append(0)
        price_per_fps.append(0)
        w_per_fps.append(0)
        continue
    last_fps = all_data[gpu]["agg_fps"][-1]
    last_w = all_data[gpu]["avg_power"][-1]
    fps_per_price.append(last_fps / gpu_price[gpu] if gpu_price[gpu] > 0 else 0)   # FPS / Price
    price_per_fps.append(gpu_price[gpu] / last_fps if last_fps > 0 else 0)         # Price / FPS
    w_per_fps.append(last_w / last_fps if last_fps > 0 else 0)                     # W / FPS

x = np.arange(len(gpu_order))
width = 0.5  # 單個長條寬度

# --------------------------
# FPS / Price 長條圖
# --------------------------
plt.figure(figsize=(10,6))
bars = plt.bar(x, fps_per_price, width=width, color=[gpu_colors[g] for g in gpu_order])
plt.xticks(x, gpu_order)
plt.xlabel("GPU")
plt.ylabel("FPS / Price")
plt.title("GPU Performance per Price (Max Load)")

for bar, val in zip(bars, fps_per_price):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01, f"{val:.4f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gpu_fps_per_price.png"), dpi=300)
plt.close()

# --------------------------
# Price / FPS 長條圖
# --------------------------
plt.figure(figsize=(10,6))
bars = plt.bar(x, price_per_fps, width=width, color=[gpu_colors[g] for g in gpu_order])
plt.xticks(x, gpu_order)
plt.xlabel("GPU")
plt.ylabel("Price / FPS")
plt.title("GPU Cost per Performance (Max Load)")

for bar, val in zip(bars, price_per_fps):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01, f"{val:.4f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gpu_price_per_fps.png"), dpi=300)
plt.close()

# --------------------------
# W / FPS 長條圖
# --------------------------
plt.figure(figsize=(10,6))
bars = plt.bar(x, w_per_fps, width=width, color=[gpu_colors[g] for g in gpu_order])
plt.xticks(x, gpu_order)
plt.xlabel("GPU")
plt.ylabel("W / FPS")
plt.title("GPU Power per Performance (Max Load)")

for bar, val in zip(bars, w_per_fps):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01, f"{val:.4f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gpu_w_per_fps.png"), dpi=300)
plt.close()
