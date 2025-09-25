import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import csv

LOG_DIR = "inference_logs/coco128"
OUTPUT_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# 讀取所有 CSV
# --------------------------
all_data = {}
csv_files = glob.glob(os.path.join(LOG_DIR, "*.csv"))

for csv_file in csv_files:
    label = os.path.basename(csv_file).replace(".csv", "")
    users, avg_latency, avg_fps_per_user, gpu_util = [], [], [], []

    with open(csv_file, "r") as fcsv:
        reader = csv.DictReader(fcsv)
        for row in reader:
            users.append(int(row["Users"]))
            avg_latency.append(float(row["Avg_Latency(s)"]))
            avg_fps_per_user.append(float(row["Avg_YOLO_FPS"]))
            gpu_util.append(float(row["GPU_Max(%)"]))  # 對應你 CSV 的 GPU 最大使用率

    all_data[label] = {
        "users": users,
        "avg_latency": avg_latency,
        "avg_fps_per_user": avg_fps_per_user,
        "gpu_util": gpu_util
    }

# --------------------------
# 畫單圖並存檔（平均 FPS / 平均延遲 / GPU 使用率）
# --------------------------
metrics = ["avg_fps_per_user", "avg_latency", "gpu_util"]
titles = ["Average FPS per User", "Average Latency (s)", "GPU Utilization (%)"]
ylabels = ["FPS", "s", "%"]

x = np.arange(len(all_data[list(all_data.keys())[0]]["users"]))
n_labels = len(all_data)
width = 0.8 / max(1, n_labels)

for i, metric in enumerate(metrics):
    plt.figure(figsize=(10, 6))
    for idx, (label, data) in enumerate(all_data.items()):
        bars = plt.bar(x + idx*width, data[metric], width=width, label=label)
        # 加上數值標註
        for bar in bars:
            height = bar.get_height()
            if metric == "avg_latency":
                plt.text(bar.get_x() + bar.get_width()/2, height,
                         f'{height:.4f}', ha='center', va='bottom', fontsize=8)
            else:
                plt.text(bar.get_x() + bar.get_width()/2, height,
                         f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.xticks(x + width*(n_labels-1)/2, all_data[list(all_data.keys())[0]]["users"])
    plt.xlabel("Number of Users")
    plt.ylabel(ylabels[i])
    plt.title(titles[i])
    plt.legend()
    plt.tight_layout()
    # 高解析度存檔
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_summary.png"), dpi=300)
    plt.close()
