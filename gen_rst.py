import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import csv

LOG_DIR = "inference_logs/coco128"
OUTPUT_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
            users.append(int(row["Users"]))
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
# 畫圖（Grouped Bar）
# --------------------------
metrics = ["avg_fps_per_user", "avg_latency", "gpu_util"]
titles = ["Average FPS per User", "Average Latency (s)", "GPU Utilization (%)"]
ylabels = ["FPS", "s", "%"]

n_labels = len(all_data)
users = all_data[list(all_data.keys())[0]]["users"]
x = np.arange(len(users))
width = 0.8 / n_labels  # 每個 GPU bar 的寬度

for i, metric in enumerate(metrics):
    plt.figure(figsize=(14, 8))
    for idx, (label, data) in enumerate(all_data.items()):
        # 偏移 x 軸位置，使 bar 分開
        bars = plt.bar(x + idx*width, data[metric], width=width, label=label)
        for bar in bars:
            height = bar.get_height()
            # 將文字放在長條上方，避免重疊
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01*height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # x 軸置中
    plt.xticks(x + width*(n_labels-1)/2, users)
    plt.xlabel("Number of Users")
    plt.ylabel(ylabels[i])
    plt.title(titles[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_summary.png"), dpi=300)
    plt.close()
