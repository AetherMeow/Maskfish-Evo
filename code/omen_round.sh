#!/bin/bash
# 在笔记本（WSL）上执行的一轮：下载数据 → 标定 → 准备 → 训练 → 挑最佳 → 上传新网
# 由编排器通过 `bash -s <round> <epochs> <lr>` 调用（脚本走 stdin，参数按位置传）
set -euo pipefail

R="${1:?需要轮次}"
EPOCHS="${2:-6}"
LR="${3:-2e-4}"
XLATE="http://10.20.0.1:18765"
DIR="/home/aether/jieqi-lab/runs/cloudrl/r${R}"
TRAINER="/home/aether/jieqi-lab/maskfish/trainer"
PREV="/home/aether/jieqi-lab/runs/cloudrl/prev.nnue"

export PATH="$HOME/jieqi-env/bin:$PATH"
mkdir -p "$DIR"
cd "$DIR"

# ---------- 1) 下载 + 解压 ----------
if [ ! -s rl.txt ]; then
  curl -s -o rl.txt.zst "$XLATE/r${R}.zst"
  zstd -d -q -f rl.txt.zst -o rl.txt
  rm -f rl.txt.zst
fi
echo "  局面数 $(wc -l < rl.txt)"

# ---------- 2) 标定 ----------
cd "$TRAINER"
if [ ! -s "$DIR/fit.log" ]; then
  python3 fit_scaling.py "$DIR/rl.txt" --by-material 8 \
    --table-out "$DIR/scaling.json" 2>&1 | tee "$DIR/fit.log" | grep -E "fitted scale" || true
fi
FIT=$(grep -oE "fitted scale: [0-9.]+" "$DIR/fit.log" | grep -oE "[0-9.]+" | head -1)
if [ -z "$FIT" ]; then echo "  !! 刻度拟合失败"; exit 1; fi
SCALE=$(awk -v f="$FIT" 'BEGIN{printf "%.6f", 361.0/f}')
echo "  刻度 $FIT -> score-scale $SCALE"

# ---------- 3) 准备数据 ----------
if [ ! -s "$DIR/prep/train-00000.features.npy" ]; then
  python3 prepare_data.py "$DIR/rl.txt" --out "$DIR/prep" \
    --score-scale "$SCALE" --workers 12 2>&1 | tail -1
fi

# ---------- 4) 训练（从上一轮的网续训） ----------
mkdir -p "$DIR/net"
if [ ! -s "$DIR/net/epoch_1.nnue" ]; then
  if [ -s "$PREV" ]; then INIT=(--init-net "$PREV"); else INIT=(); fi
  python3 train.py "$DIR/prep" --val /home/aether/jieqi-lab/data/gen1/prep10/val \
    --out "$DIR/net" --epochs "$EPOCHS" --epoch-size 10000000 --batch-size 16384 \
    --lr "$LR" --gamma 0.995 --lambda 1.0 --scaling 361 \
    --device cuda --seed 42 "${INIT[@]}" \
    2>&1 | tee "$DIR/train.log" | grep -E "^epoch [0-9]+: "
fi

# ---------- 5) 挑 val 最低的 epoch ----------
python3 - "$DIR" <<'PY'
import os, re, sys
d = sys.argv[1]
best = None
for line in open(os.path.join(d, "train.log"), encoding="utf-8", errors="ignore"):
    m = re.match(r"epoch (\d+): .*val loss ([0-9.]+)", line)
    if m:
        e, v = int(m.group(1)), float(m.group(2))
        if best is None or v < best[1]:
            best = (e, v)
assert best, "no val loss found"
p = os.path.join(d, "net", "epoch_%d.nnue" % best[0])
open(os.path.join(d, "best_net.txt"), "w").write(p + "\n")
print("  最佳 epoch=%d val=%.6f" % best)
PY

# ---------- 6) 上传新网 + 更新 prev ----------
BEST=$(cat "$DIR/best_net.txt")
echo "  上传 $(basename "$BEST") ($(stat -c%s "$BEST") 字节)"
curl -s -X PUT --data-binary @"$BEST" "$XLATE/upload/r${R}.nnue"
cp -f "$BEST" "$PREV"
echo "  ROUND_DONE"
