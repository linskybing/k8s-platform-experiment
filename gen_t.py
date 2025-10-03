import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import csv

LOG_DIR = "training_logs"
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

# GPU 價格 (TWD)
gpu_price_twd = {
    "RTX3090": 50000,
    "RTX5070 TI": 28000,
    "RTX4070 SUPER": 22000,
    "RTX2080 TI": 35000
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
            try:
                u = int(row["Users"])
                if u > 13:
                    continue
            except:
                continue

            afps = float(row.get("Aggregate_FPS", 0))
            latency = float(row.get("Avg_Latency(s)", 0))
            fps_user = float(row.get("Avg_YOLO_FPS", 0))
            util = float(row.get("GPU_Max(%)", 0))
            power = float(row.get("Power_Avg(W)", 0))
            mem_percent = float(row.get("Mem_Max(%)", 0))
            total_mem = gpu_total_mem.get(label, 0)
            mem_gb = mem_percent * total_mem / 1024 / 100

            users.append(u)
            avg_latency.append(latency)
            avg_fps_per_user.append(fps_user)
            gpu_util.append(util)
            avg_power.append(power)
            max_mem_gb.append(mem_gb)
            agg_fps.append(afps)

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
# 計算 Price / Mem (TWD/GB) 單值
# --------------------------
gpu_order = ["RTX3090", "RTX5070 TI", "RTX4070 SUPER", "RTX2080 TI"]
gpu_colors = {
    "RTX3090": "#1f77b4",
    "RTX5070 TI": "#ff7f0e",
    "RTX4070 SUPER": "#2ca02c",
    "RTX2080 TI": "#d62728"
}

price_per_mem = []
for gpu in gpu_order:
    if gpu in all_data and len(all_data[gpu]["max_mem_gb"]) > 0:
        mem_gb = all_data[gpu]["max_mem_gb"][-1]
        price_per_mem.append(gpu_price_twd[gpu] / mem_gb if mem_gb > 0 else 0)
    else:
        price_per_mem.append(0)

for idx, gpu in enumerate(gpu_order):
    if gpu in all_data:
        all_data[gpu]["price_per_mem"] = [price_per_mem[idx]]

# --------------------------
# 畫 Users 長條圖 (FPS, Latency, Power, GPU Util, Memory)
# --------------------------
metrics_users = ["avg_fps_per_user", "agg_fps", "avg_latency", "gpu_util", "avg_power", "max_mem_gb"]
titles_users = ["Average FPS per User", "Aggregate FPS", "Average Latency (s)", "GPU Utilization (%)", "Average Power (W)", "Memory (GB)"]
ylabels_users = ["FPS", "FPS", "s", "%", "W", "GB"]
width = 0.20
gap = 0.01
figsize_users = (14, 8)

for i, metric in enumerate(metrics_users):
    plt.figure(figsize=figsize_users)
    max_users = max([len(all_data[g]["users"]) for g in gpu_order if g in all_data])
    x = np.arange(max_users)
    
    for idx, gpu in enumerate(gpu_order):
        if gpu not in all_data:
            continue
        data = all_data[gpu][metric]
        users_len = len(data)
        offset = idx * (width + gap)
        plt.bar(x[:users_len] + offset, data, width=width, label=gpu, color=gpu_colors.get(gpu))
        for xi, val in zip(x[:users_len], data):
            plt.text(xi + offset, val * 1.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)

    plt.xlabel("Number of Users")
    plt.ylabel(ylabels_users[i])
    plt.title(titles_users[i])
    plt.xticks(x + (len(gpu_order)-1)*(width+gap)/2, range(1, max_users+1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_bar.png"), dpi=300)
    plt.close()

# --------------------------
# 畫 Price / Mem 長條圖 (GPU 為 X)
# --------------------------
plt.figure(figsize=(12, 6))
for idx, gpu in enumerate(gpu_order):
    if gpu not in all_data:
        continue
    value = all_data[gpu]["price_per_mem"][0]
    plt.bar(idx, value, width=0.5, color=gpu_colors.get(gpu))
    plt.text(idx, value*1.01, f"{value:.2f}", ha='center', va='bottom', fontsize=10)

plt.xticks(range(len(gpu_order)), gpu_order)
plt.ylabel("Price / Memory (TWD/GB)")
plt.title("GPU Price per Memory")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "price_per_mem_bar.png"), dpi=300)
plt.close()

# --------------------------
# 畫 Users 趨勢圖 (FPS, Latency, Power, GPU Util, Memory)
# --------------------------
metrics_users = ["avg_fps_per_user", "agg_fps", "avg_latency", "gpu_util", "avg_power", "max_mem_gb"]
titles_users = ["Average FPS per User", "Aggregate FPS", "Average Latency (s)", "GPU Utilization (%)", "Average Power (W)", "Memory (GB)"]
ylabels_users = ["FPS", "FPS", "s", "%", "W", "GB"]
figsize_users = (14, 8)

for i, metric in enumerate(metrics_users):
    plt.figure(figsize=figsize_users)
    
    for idx, gpu in enumerate(gpu_order):
        if gpu not in all_data:
            continue
        x = all_data[gpu]["users"]
        y = all_data[gpu][metric]
        plt.plot(x, y, marker='o', label=gpu, color=gpu_colors.get(gpu))
        # 標註每個點的數值
        for xi, yi in zip(x, y):
            plt.text(xi, yi * 1.01, f"{yi:.2f}", ha='center', va='bottom', fontsize=10)
    
    plt.xlabel("Number of Users")
    plt.ylabel(ylabels_users[i])
    plt.title(titles_users[i])
    plt.xticks(range(1, max([len(all_data[g]["users"]) for g in gpu_order if g in all_data])+1))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_trend.png"), dpi=300)
    plt.close()

# --------------------------
# 畫 Price / Mem 趨勢圖 (GPU 為 X)
# --------------------------
plt.figure(figsize=(12, 6))
x = np.arange(len(gpu_order))
y = [all_data[g]["price_per_mem"][0] if g in all_data else 0 for g in gpu_order]
plt.plot(x, y, marker='o', linestyle='-', color='tab:blue')
for xi, yi in zip(x, y):
    plt.text(xi, yi*1.01, f"{yi:.2f}", ha='center', va='bottom', fontsize=10)

plt.xticks(x, gpu_order)
plt.ylabel("Price / Memory (TWD/GB)")
plt.title("GPU Price per Memory")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "price_per_mem_trend.png"), dpi=300)
plt.close()