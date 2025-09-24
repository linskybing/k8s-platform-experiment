import os
from ultralytics.utils.downloads import download

# --------------------------
# Config
# --------------------------
DATA_DIR = "datasets"
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------
# COCO128
# --------------------------
coco128_path = os.path.join(DATA_DIR, "coco128")
if not os.path.exists(coco128_path):
    print("COCO128 not found. Downloading...")
    download("https://ultralytics.com/assets/coco128.zip", unzip=True)
else:
    print(f"COCO128 already exists at {coco128_path}")

# --------------------------
# COCO2017 val
# --------------------------
coco2017_val_path = os.path.join(DATA_DIR, "coco2017val")
if not os.path.exists(coco2017_val_path):
    print("COCO2017 val not found. Downloading...")
    download("https://ultralytics.com/assets/coco2017val.zip", unzip=True)
else:
    print(f"COCO2017 val already exists at {coco2017_val_path}")
