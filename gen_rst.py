import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import csv

LOG_DIR = "inference_logs"
OUTPUT_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# GPU 總記憶體 (MB)
# --------------------------
gpu_total_mem = {
    "RTX3090": 24576,
    "RTX5070 TI": 16384,
    "RTX4070 SUPER": 12288,
    "RTX2080 TI": 11264
}

# --------------------------
# 讀取 CSV
# --------------------------
all_data = {}
csv_files = glob.glob(os.path.join(LOG_DIR, "*.csv"))

for csv_file in csv_files:
    label = os.path.basename(csv_file).replace(".csv", "")
    users, avg_latency, avg_fps_per_user, gpu_util, avg_power, max_mem_gb, agg_fps = [], [], [], [], [], [], []

    with open(csv_file, "r") as fcsv:
        reader = csv.DictReader(fcsv)
        for row in reader:
            u = int(row["Users"])
            if u > 13:
                continue
            users.append(u)
            avg_latency.append(float(row["Avg_Latency(s)"]))
            avg_fps_per_user.append(float(row["Avg_YOLO_FPS"]))
            gpu_util.append(float(row["GPU_Max(%)"]))
            avg_power.append(float(row.get("Power_Avg(W)", 0)))
            agg_fps.append(float(row.get("Aggregate_FPS", 0)))

            # Mem_Max(%) → GB
            mem_percent = float(row.get("Mem_Max(%)", 0))
            total_mem = gpu_total_mem.get(label, 0)
            max_mem_gb.append(mem_percent * total_mem / 1024 / 100)

    if users:
        all_data[label] = {
            "users": users,
            "avg_latency": avg_latency,
            "avg_fps_per_user": avg_fps_per_user,
            "gpu_util": gpu_util,
            "avg_power": avg_power,
            "max_mem_gb": max_mem_gb,
            "agg_fps": agg_fps
        }

# --------------------------
# 畫圖設定
# --------------------------
metrics = ["avg_fps_per_user", "agg_fps", "avg_latency", "gpu_util", "avg_power", "max_mem_gb"]
titles = ["Average FPS per User", "Aggregate FPS", "Average Latency (s)", "GPU Utilization (%)",
          "Average Power (W)", "Max Memory (GB)"]
ylabels = ["FPS", "FPS", "s", "%", "W", "GB"]

gpu_order = ["RTX3090", "RTX5070 TI", "RTX4070 SUPER", "RTX2080 TI"]
gpu_colors = {
    "RTX3090": "#1f77b4",
    "RTX5070 TI": "#ff7f0e",
    "RTX4070 SUPER": "#2ca02c",
    "RTX2080 TI": "#d62728"
}

users = all_data[list(all_data.keys())[0]]["users"]
x = np.arange(len(users))
width = 0.18
gap = 0.02

for i, metric in enumerate(metrics):
    # ----------------- 長條圖 -----------------
    plt.figure(figsize=(14, 8))
    bars_plotted = False
    for idx, label in enumerate(gpu_order):
        if label not in all_data:
            continue
        data = all_data[label][metric]
        offset = idx * (width + gap)
        plt.bar(x + offset, data, width=width, label=label, color=gpu_colors.get(label))
        bars_plotted = True

    if bars_plotted:
        total_group_width = len([l for l in gpu_order if l in all_data]) * width + \
                            (len([l for l in gpu_order if l in all_data]) - 1) * gap
        plt.xticks(x + total_group_width / 2 - width / 2, users)
        plt.xlabel("Number of Users")
        plt.ylabel(ylabels[i])
        plt.title(titles[i])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_bar.png"), dpi=300)
    plt.close()

    # ----------------- 趨勢圖 (每 GPU 一條線) -----------------
    plt.figure(figsize=(14, 8))
    for label in gpu_order:
        if label not in all_data:
            continue
        data = all_data[label][metric]
        plt.plot(users, data, marker='o', linewidth=2, label=label, color=gpu_colors.get(label))

    plt.xlabel("Number of Users")
    plt.ylabel(ylabels[i])
    plt.title(f"{titles[i]} Trend by GPU")
    plt.xticks(users)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_trend_by_gpu.png"), dpi=300)
    plt.close()
