#!/bin/bash

USER_ID=$1
if [ -z "$USER_ID" ]; then
    echo "Usage: $0 <user_id>"
    exit 1
fi

MODEL_PATH="yolo11n.pt"
DATA_YAML="coco128/coco128.yaml"
EPOCHS=1
IMG_SIZE=640
NUM_IMAGES=$((128 * EPOCHS))
PROJECT_DIR="/tmp/train_user${USER_ID}"
TMP_MODEL="/tmp/yolo11n_user${USER_ID}.pt"
LOG_FILE="${PROJECT_DIR}/train.log"

mkdir -p "$PROJECT_DIR"
cp "$MODEL_PATH" "$TMP_MODEL"

echo "[User $USER_ID] Starting training..."

# 執行 Python 並捕捉 OOM 或其他 RuntimeError
python3 - <<EOF >"$LOG_FILE" 2>&1
from ultralytics import YOLO
import sys

try:
    model = YOLO("$TMP_MODEL")
    model.train(
        data="$DATA_YAML",
        epochs=$EPOCHS,
        device=0,
        imgsz=$IMG_SIZE,
        verbose=False,
        save=False,
        exist_ok=True,
        project="$PROJECT_DIR"
    )
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("[User $USER_ID] CUDA OOM detected!", file=sys.stderr)
        sys.exit(100)
    else:
        print(f"[User $USER_ID] RuntimeError:", e, file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"[User $USER_ID] Unexpected error:", e, file=sys.stderr)
    sys.exit(1)
EOF

PY_EXIT_CODE=$?
if [ $PY_EXIT_CODE -eq 100 ]; then
    echo "[User $USER_ID] Exiting due to CUDA OOM."
    exit 1
elif [ $PY_EXIT_CODE -ne 0 ]; then
    echo "[User $USER_ID] Python error detected, exiting."
    exit 1
fi

# 解析小時數
HOURS=$(grep "epochs completed" "$LOG_FILE" | head -n1 | sed -E 's/.*in ([0-9.]+) hours.*/\1/')
ELAPSED=$(awk "BEGIN{print $HOURS*3600}")


# 計算 latency & fps
AVG_LATENCY=$(awk -v e="$ELAPSED" -v n="$NUM_IMAGES" 'BEGIN{print (n>0)? e/n : 0}')
FPS=$(awk -v e="$ELAPSED" -v n="$NUM_IMAGES" 'BEGIN{print (e>0)? n/e : 0}')

# 取得 GFLOPs
GFLOPS=$(grep -oP '([\d.]+)\s*GFLOPs?' "$LOG_FILE" | head -n1)
GFLOPS=$(echo "${GFLOPS:-0.0}" | grep -oP '[\d.]+')

# 最終結果寫入 log 並 echo
RESULT_LINE="[User $USER_ID] Total Time: $ELAPSED s | Avg Latency: $AVG_LATENCY s | Avg FPS: $FPS | GFLOPs: $GFLOPS"
echo "$RESULT_LINE" | tee -a "$LOG_FILE"

echo "[User $USER_ID] Training completed. Log saved at $LOG_FILE"
