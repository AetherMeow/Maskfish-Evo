#!/bin/bash
# 云服务器一键部署 + 基准测试
# 用法：bash deploy_server.sh            （纯部署 + 基准）
#       SKIP_BENCH=1 bash deploy_server.sh  （只部署，不跑基准）
set -euo pipefail
LOG=/root/deploy.log
exec > >(tee -a "$LOG") 2>&1

echo "=================================================================="
echo " Jieqi 生成节点部署    $(date '+%F %T')"
echo "=================================================================="

# ---------- 1) 系统信息 ----------
echo
echo "=== [1/7] 系统信息 ==="
. /etc/os-release 2>/dev/null || true
echo "  OS      : ${PRETTY_NAME:-unknown}"
echo "  Kernel  : $(uname -r)"
echo "  vCPU    : $(nproc)"
echo "  内存    : $(free -g | awk '/^Mem:/{print $2}') GiB"
echo "  磁盘    : $(df -h / | awk 'NR==2{print $4" 可用"}')"
echo "  CPU 型号: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"

# ---------- 2) 依赖 ----------
echo
echo "=== [2/7] 安装依赖 ==="
export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq build-essential python3 zstd rsync time coreutils 2>&1 | tail -2
elif command -v dnf >/dev/null; then
  dnf install -y -q gcc gcc-c++ make python3 zstd rsync time 2>&1 | tail -2
fi
echo "  g++: $(g++ --version | head -1)"
echo "  python3: $(python3 --version)"

# ---------- 3) 解包 ----------
echo
echo "=== [3/7] 解包 ==="
cd /root
rm -rf jieqi-gen
tar xzf /root/jieqi-gen.tgz
[ -f /root/jieqi-gen/maskfish/src/Makefile ] || { echo "!! 包内容不对"; exit 1; }
[ -s /root/jieqi-gen/net_rl2.nnue ] || { echo "!! 网络文件缺失"; exit 1; }
echo "  引擎源码 $(ls /root/jieqi-gen/maskfish/src/*.cpp | wc -l) 个 .cpp"
echo "  网络 $(ls -lh /root/jieqi-gen/net_rl2.nnue | awk '{print $5}')"

# ---------- 4) 编译 ----------
echo
echo "=== [4/7] 编译引擎（ARCH=x86-64-avx2，故意不用 -march=native）==="
cd /root/jieqi-gen/maskfish/src
make clean >/dev/null 2>&1 || true
make -j "$(nproc)" build ARCH=x86-64-avx2 COMP=gcc 2>&1 | tail -4
[ -x ./maskfish ] || { echo "!! 编译失败，看上面报错"; exit 1; }
echo "  产物: $(ls -lh ./maskfish | awk '{print $5}')"

# ---------- 5) 冒烟 ----------
echo
echo "=== [5/7] 冒烟测试（加载网络）==="
printf 'uci\nquit\n' | ./maskfish 2>&1 | head -2
echo "  --- 用我们的网搜一步 ---"
FEN='1pxxkxxxx/9/a3n4/x1x1x1x1x/9/8P/X1X1X1X2/9/7B1/XXXXKXXXX w R2A2C1P4N1B1r2a1c1p5n1b1 0 1'
printf 'setoption name EvalFile value /root/jieqi-gen/net_rl2.nnue\nposition fen %s\ngo nodes 5000\nquit\n' "$FEN" \
  | ./maskfish 2>&1 | grep -aE "info depth|bestmove" | tail -2

# ---------- 6) 基准 ----------
if [ "${SKIP_BENCH:-0}" = "1" ]; then
  echo; echo "=== [6/7] 跳过基准测试（SKIP_BENCH=1）==="
else
  echo
  echo "=== [6/7] 基准测试（真实负载，约 3~5 分钟）==="
  cd /root/jieqi-gen/maskfish/trainer
  NPROC=$(nproc)
  W=$(( NPROC > 8 ? NPROC - 8 : NPROC ))     # 留 8 个核给裁判和系统
  echo "  workers = $W"
  rm -f /root/bench.txt /root/bench.txt.part*
  T0=$(date +%s)
  python3 gen_data.py --engine ../src/maskfish --net /root/jieqi-gen/net_rl2.nnue \
    --games 12000 --nodes 5000 --adjudicate 1000 --workers "$W" --seed 424242 \
    --out /root/bench.txt 2>&1 | tail -2
  T1=$(date +%s)
  POS=$(wc -l < /root/bench.txt)
  EL=$(( T1 - T0 ))
  python3 - "$POS" "$EL" "$W" <<'PY'
import sys
pos, el, w = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rate = pos / max(el,1) * 3600
base = 3_200_000
print()
print(f"  实测：{pos:,} 局面 / {el} 秒  =  {rate:,.0f} 局面/小时（{w} workers）")
print(f"  笔记本基准：{base:,} 局面/小时（24 线程，16 workers）")
print(f"  加速比：{rate/base:.2f}x")
print()
if rate > 4_500_000:
    print("  ✓ 加速到位 —— 可以跑 10 轮（每轮 1000 万局面约 {:.1f} 小时）".format(1e7/rate))
elif rate > 3_500_000:
    print("  ~ 一般 —— 每轮约 {:.1f} 小时，可以跑但收益有限".format(1e7/rate))
else:
    print("  ✗ 太慢 —— 建议释放实例，换可用区或换实例规格")
PY
fi

# ---------- 7) 结论 ----------
echo
echo "=== [7/7] 部署完成 $(date '+%F %T') ==="
echo "  引擎      : /root/jieqi-gen/maskfish/src/maskfish"
echo "  生成脚本  : /root/jieqi-gen/maskfish/trainer/gen_data.py"
echo "  网络      : /root/jieqi-gen/net_rl2.nnue"
echo "  日志      : $LOG"
echo "=================================================================="
