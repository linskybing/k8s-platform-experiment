import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import csv
import pandas as pd
from dotenv import load_dotenv

LOG_DIR = "inference_logs/"
OUTPUT_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
load_dotenv()

# --------------------------
# 讀取 CSV
# --------------------------
all_data = {}
csv_files = glob.glob(os.path.join(LOG_DIR, "*.csv"))

for csv_file in csv_files:
    label = os.path.basename(csv_file).replace(".csv", "")
    users, avg_latency, avg_fps_per_user, gpu_util = [], [], [], []

    with open(csv_file, "r") as fcsv:
        reader = csv.DictReader(fcsv)
        for row in reader:
            user_count = int(row["Users"])
            if user_count > 12:
                continue
            users.append(user_count)
            avg_latency.append(float(row["Avg_Latency(s)"]))
            avg_fps_per_user.append(float(row["Avg_YOLO_FPS"]))
            gpu_util.append(float(row["GPU_Max(%)"]))

    all_data[label] = {
        "users": users,
        "avg_latency": avg_latency,
        "avg_fps_per_user": avg_fps_per_user,
        "gpu_util": gpu_util
    }

# --------------------------
# 固定 GPU 排序與顏色
# --------------------------
gpu_order = ["RTX3090", "RTX5070 TI", "RTX4070 SUPER", "RTX2080 TI"]
gpu_colors = {
    "RTX3090": "#1f77b4",
    "RTX5070 TI": "#ff7f0e",
    "RTX4070 SUPER": "#2ca02c",
    "RTX2080 TI": "#d62728"
}
all_data = {k: all_data[k] for k in gpu_order if k in all_data}

# --------------------------
# 畫圖設定
# --------------------------
metrics = ["avg_fps_per_user", "avg_latency", "gpu_util"]
titles = ["Average FPS per User", "Average Latency (s)", "GPU Utilization (%)"]
ylabels = ["FPS", "s", "%"]

n_labels = len(all_data)
users = all_data[list(all_data.keys())[0]]["users"]
x = np.arange(len(users))
total_width = 0.8  # 一組柱子總寬度
width = total_width / n_labels  # 每個柱子的寬度自動調整

# --------------------------
# 畫長條圖（Grouped Bar）並加數字標籤
# --------------------------
for i, metric in enumerate(metrics):
    plt.figure(figsize=(18, 10))
    for idx, (label, data) in enumerate(all_data.items()):
        bars = plt.bar(x + idx*width, data[metric], width=width,
                       label=label, color=gpu_colors.get(label, None))
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.01 * max(data[metric]),
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10
            )

    plt.xticks(x + width*(n_labels-1)/2, users)
    plt.xlabel("Number of Users")
    plt.ylabel(ylabels[i])
    plt.title(titles[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_summary.png"), dpi=300)
    plt.close()

# --------------------------
# 畫折線圖（趨勢比較）並加數字標籤
# --------------------------
for i, metric in enumerate(metrics):
    plt.figure(figsize=(18, 10))
    for label in gpu_order:
        if label in all_data:
            data = all_data[label]
            plt.plot(data["users"], data[metric], marker="o",
                     label=label, color=gpu_colors.get(label, None), linewidth=2)
            # 折線圖加數字標籤
            for u, val in zip(data["users"], data[metric]):
                plt.text(
                    u, val + 0.02 * max(data[metric]),
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=10
                )

    plt.xlabel("Number of Users")
    plt.ylabel(ylabels[i])
    plt.title(titles[i] + " (Trend)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_trend.png"), dpi=300)
    plt.close()
