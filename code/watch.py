#!/usr/bin/env python3
"""看门狗：受管理的后台任务，退出即唤醒 agent。

监控 orchestrator.py 的健康，并在下列情况退出（退出就会唤醒 agent）：
  1. 全部 10 轮完成
  2. 编排器进程消失（崩溃）
  3. 生成停滞：云上局面数 25 分钟没有增长（且仍处于生成阶段）
  4. 训练/传输阶段超过 75 分钟没有进入下一轮
  5. 云服务器连续 3 次不可达（抢占式被回收 / 实例故障）

正常运行时每 5 分钟记一行进度到 logs/watch.log，供 agent 随时查看。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROUNDS = int(__import__("os").environ.get("ROUNDS", "10"))
HOME = Path("/home/aether/work/jieqi-run")
STATE = HOME / "state.json"
WATCHLOG = HOME / "logs" / "watch.log"
POLL = 300                 # 5 分钟一次
STALL_GEN = 25 * 60        # 生成阶段允许 25 分钟无增长
STALL_TRAIN = 75 * 60      # 训练/传输阶段允许 75 分钟
SSH_FAILS_MAX = 3

SRV = "root@47.111.114.23"
SRV_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
           "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15",
           "-o", "LogLevel=ERROR", SRV]


def log(msg: str):
    line = f"[{time.strftime('%F %T')}] {msg}"
    with open(WATCHLOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def orch_alive() -> bool:
    p = subprocess.run(["pgrep", "-f", "orchestrator[.]py"], capture_output=True, text=True)
    return p.stdout.strip() != ""


def cur_round() -> int:
    if STATE.is_file():
        try:
            return int(json.loads(STATE.read_text()).get("round", 0))
        except Exception:
            return 0
    return 0


def cloud_progress(r: int) -> tuple[int, str] | None:
    """返回 (进度签名, 阶段)；不可达返回 None。
    分块模式下：签名 = 已完成块数 * 1e9 + 当前块已生成局面数，单调增即可判活。"""
    cmd = (
        f"d=/root/runs/r{r}; "
        "done=$(ls $d/c*.txt.zst 2>/dev/null | wc -l); "
        "parts=$(cat $d/c*.txt.part[0-9]* 2>/dev/null | wc -l); "
        "echo \"$((done*1000000000+parts))\""
    )
    try:
        p = subprocess.run(SRV_SSH + [cmd], capture_output=True, text=True, timeout=90)
        if p.returncode != 0:
            return None
        return int(p.stdout.strip() or 0), "gen"
    except Exception:
        return None


def main() -> int:
    log("=" * 70)
    log(f"看门狗启动：目标 {ROUNDS} 轮")
    log("=" * 70)

    last_round_done = cur_round()
    last_pos = -1
    last_change = time.time()
    phase = "gen"
    ssh_fails = 0

    while True:
        time.sleep(POLL)
        done = cur_round()

        # 1) 全部完成
        if done >= ROUNDS and not orch_alive():
            log(f"✅ 全部 {ROUNDS} 轮完成，编排器已退出")
            log("   → 该门控 10 个网了")
            return 0

        # 2) 编排器崩了
        if not orch_alive():
            log(f"❌ 编排器进程消失了（已完成 {done}/{ROUNDS} 轮）")
            log("   → 需要检查 logs/orchestrator.log 并重启")
            return 2

        # 轮次推进 → 重置停滞计时
        if done != last_round_done:
            log(f"  第 {done}/{ROUNDS} 轮完成")
            last_round_done = done
            last_pos, last_change, phase = -1, time.time(), "gen"
            continue

        r = done + 1

        # 3) 云上进度
        pr = cloud_progress(r)
        if pr is None:
            ssh_fails += 1
            log(f"  !! 云端不可达（第 {ssh_fails} 次）")
            if ssh_fails >= SSH_FAILS_MAX:
                log("❌ 云服务器连续不可达 —— 抢占式很可能被回收了")
                log(f"   → 已完成 {done}/{ROUNDS} 轮；需要重建实例并续跑")
                return 3
            continue
        ssh_fails = 0
        pos, ph = pr

        if ph == "post":
            if phase != "post":
                log(f"  第 {r} 轮生成结束，进入压缩/传输/训练阶段")
                phase, last_change = "post", time.time()
            elif time.time() - last_change > STALL_TRAIN:
                log(f"❌ 第 {r} 轮在传输/训练阶段卡了 {(time.time()-last_change)/60:.0f} 分钟")
                return 4
            else:
                log(f"  第 {r} 轮：传输/训练中（已 {(time.time()-last_change)/60:.0f} 分钟）")
            continue

        # 生成阶段
        if pos != last_pos:
            last_pos, last_change = pos, time.time()
        idle = time.time() - last_change
        chunks_done = pos // 1000000000
        cur = pos % 1000000000
        log(f"  第 {r} 轮生成中：已完成 {chunks_done} 块，当前块 {cur:,} 局面，静止 {idle/60:.1f} 分钟")
        if idle > STALL_GEN:
            log(f"❌ 第 {r} 轮生成停滞 {idle/60:.0f} 分钟（局面数不再增长）")
            return 5

    return 0


if __name__ == "__main__":
    sys.exit(main())
