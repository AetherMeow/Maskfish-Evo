#!/usr/bin/env python3
"""10 轮自对弈 RL 循环的编排器（跑在家里这台机器上）。

数据通路：
    云上 ──scp──▶ 家里 ──HTTP(10.20.0.1:18765)──▶ 笔记本（训练）
            ◀──scp── 家里 ◀──────HTTP PUT──────── 笔记本

每轮：
  1. 云上生成 1000 万局面（40 workers，用"当前最好的网"自对弈）
  2. zstd 压缩 → scp 拉回 → 放进 HTTP 传输目录
  3. 笔记本下载 → 标定刻度 → 准备数据 → GPU 训练（续训，λ=1.0，6 遍）
  4. 笔记本挑 val 最低的 epoch → PUT 回家里
  5. 家里 scp 推回云上，成为下一轮的"当前最好的网"
  6. 记状态（可断点续跑）

取舍说明：
  * 每轮**无条件采纳**新网，不做门控——门控一轮 1~2 小时，10 轮会超过生成时间。
    所有网都留在 nets/ 里，最后一轮结束后统一门控挑最好的。
  * 每轮只用**本轮新数据**训练 6 遍（对应 jqé 的"最近对局窗口"）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROUNDS = int(os.environ.get("ROUNDS", "10"))
GAMES = int(os.environ.get("GAMES", "280000"))       # ≈1000 万局面
NODES = int(os.environ.get("NODES", "5000"))
WORKERS_SRV = int(os.environ.get("WORKERS_SRV", "40"))
CHUNKS = int(os.environ.get("CHUNKS", "7"))
GAMES_PER_CHUNK = 0  # 由 GAMES/CHUNKS 算出
EPOCHS = int(os.environ.get("EPOCHS", "6"))
LR = os.environ.get("LR", "2e-4")

SRV = "root@47.111.114.23"
SRV_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
           "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20",
           "-o", "LogLevel=ERROR", SRV]

HOME = Path("/home/aether/work/jieqi-run")
TRANSFER = HOME / "transfer"
UPLOAD = TRANSFER / "in"
STATE_FILE = HOME / "state.json"
LOGDIR = HOME / "logs"
NODES_DIR = HOME / "nets"
XLATE = "http://10.20.0.1:18765"

OMEN_SCRIPT = HOME / "omen_round.sh"


def log(msg: str):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOGDIR / "orchestrator.log", "a") as f:
        f.write(line + "\n")


def run(cmd: list[str], what: str, timeout: int = 7200,
        input_text: str | None = None) -> str:
    try:
        p = subprocess.run(cmd, input=input_text, capture_output=True,
                           text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode != 0:
            log(f"  !! {what} rc={p.returncode}")
            for line in out.strip().splitlines()[-10:]:
                log(f"     {line}")
            raise RuntimeError(what)
        return out
    except subprocess.TimeoutExpired:
        log(f"  !! {what} 超时 {timeout}s")
        raise RuntimeError(f"{what} timeout")


def srv(cmd: str, what: str, timeout: int = 7200) -> str:
    return run(SRV_SSH + [cmd], what, timeout)


def omen(script: str, args: list[str], what: str, timeout: int = 21600) -> str:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
           "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=60",
           "omen", "wsl", "-d", "Ubuntu", "--", "bash", "-s"] + args
    return run(cmd, what, timeout, input_text=script)


def retry(fn, what: str, tries: int = 3):
    for i in range(1, tries + 1):
        try:
            return fn()
        except RuntimeError:
            if i < tries:
                log(f"  {what}: 第 {i} 次失败，15 秒后重试")
                time.sleep(15)
    raise SystemExit(f"{what} 连续失败，编排器停止")


def load_state() -> dict:
    if STATE_FILE.is_file():
        return json.loads(STATE_FILE.read_text())
    return {"round": 0, "best_net": "/root/jieqi-gen/net_rl2.nnue", "history": []}


def save_state(st: dict):
    STATE_FILE.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    for d in (LOGDIR, TRANSFER, UPLOAD, NODES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    global GAMES_PER_CHUNK
    GAMES_PER_CHUNK = max(1, GAMES // CHUNKS)
    st = load_state()
    script = OMEN_SCRIPT.read_text()
    log("=" * 70)
    log(f"编排器启动：从第 {st['round']+1} 轮开始，共 {ROUNDS} 轮")
    log(f"  每轮 {GAMES} 局 / {NODES} 节点 / 云上 {WORKERS_SRV} workers / 训练 {EPOCHS} 遍 lr={LR}")
    log(f"  当前最好的网：{st['best_net']}")
    log("=" * 70)

    for r in range(st["round"] + 1, ROUNDS + 1):
        t0 = time.time()
        srv_dir = f"/root/runs/r{r}"
        log(f"--- 第 {r}/{ROUNDS} 轮（网：{Path(st['best_net']).name}）---")

        # 1+2) 分块生成 + 逐块增量拉回
        local_txt = TRANSFER / f"r{r}.txt"
        merged = int(st.get("chunks", {}).get(str(r), 0))   # 已合并进 local_txt 的块数
        # 分块模式的标记物是 .zst 文件（不是 local_txt）
        have_zst = sorted(TRANSFER.glob(f"r{r}_c*.zst"))
        have_merged_zst = (TRANSFER / f"r{r}.zst").is_file()
        if merged > 0 and not have_zst and not have_merged_zst:
            log(f"  !! state 说块 {merged} 已合并但本地找不到任何块文件，从头来")
            merged = 0
            st.setdefault("chunks", {})[str(r)] = 0
            save_state(st)
        if merged == 0 and local_txt.is_file():
            local_txt.unlink()
        total_pos = int(st.get("pos", {}).get(str(r), 0))
        log(f"  第 {r} 轮：已合并 {merged}/{CHUNKS} 块，本地 {total_pos:,} 局面")

        for c in range(merged + 1, CHUNKS + 1):
            gen = (
                "set -e\n"
                f"mkdir -p {srv_dir}\n"
                "cd /root/jieqi-gen/maskfish/trainer\n"
                f"if [ ! -s {srv_dir}/c{c}.txt ]; then\n"
                f"  python3 gen_data.py --engine ../src/maskfish --net {st['best_net']} "
                f"--games {GAMES_PER_CHUNK} --nodes {NODES} --adjudicate 1000 "
                f"--workers {WORKERS_SRV} --seed {7000 + r * 100 + c} --out {srv_dir}/c{c}.txt 2>&1 | tail -1\n"
                "fi\n"
                f"zstd -19 -T0 -q -f {srv_dir}/c{c}.txt -o {srv_dir}/c{c}.txt.zst\n"
                f"rm -f {srv_dir}/c{c}.txt\n"
                f"stat -c%s {srv_dir}/c{c}.txt.zst\n"
            )
            retry(lambda: srv(gen, f"r{r} 块{c} 生成", timeout=3 * 3600), f"r{r} 块{c} 生成")
            zst = TRANSFER / f"r{r}_c{c}.zst"
            chunk_local = TRANSFER / f"r{r}_c{c}.txt"
            retry(lambda: run(["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                               "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                               f"{SRV}:{srv_dir}/c{c}.txt.zst", str(zst)],
                              f"r{r} 块{c} 拉回", timeout=1800), f"r{r} 块{c} 拉回")
            # 不重新压缩：保留 .zst（zstd 多帧可拼接），流式数一下行数
            out_cnt = run(["bash", "-c", f"zstd -dc {zst} | wc -l"],
                          f"r{r} 块{c} 计数", timeout=900)
            n_chunk = int(out_cnt.strip() or 0)
            total_pos += n_chunk
            merged = c
            st.setdefault("chunks", {})[str(r)] = merged
            st.setdefault("pos", {})[str(r)] = total_pos
            save_state(st)
            log(f"  第 {r} 轮 块 {c}/{CHUNKS} 已合并（{n_chunk:,} 局面，累计 {total_pos:,}）")
            try:
                srv(f"rm -f {srv_dir}/c{c}.txt.zst", f"r{r} 块{c} 清理", timeout=120)
            except Exception:
                pass
        npos = str(total_pos)
        log(f"  第 {r} 轮数据齐了：{total_pos:,} 局面")

        # 放进 HTTP 传输目录：直接拼已压缩的分块（瞬时，且 zstd 多帧合法）
        dst = TRANSFER / f"r{r}.zst"
        chunk_zsts = sorted(TRANSFER.glob(f"r{r}_c*.zst"))
        if chunk_zsts:
            with open(dst, "wb") as fo:
                for cz in chunk_zsts:
                    fo.write(cz.read_bytes())
            for cz in chunk_zsts:
                cz.unlink(missing_ok=True)
        elif not dst.is_file():
            # 兜底：没有分块（例如第 1 轮的遗留数据），退回压缩
            retry(lambda: run(["zstd", "-3", "-T0", "-q", "-f", str(local_txt),
                               "-o", str(dst)], f"r{r} 压缩", timeout=1800), f"r{r} 压缩")
        log(f"  已备好 {dst.name}（{dst.stat().st_size/1e6:.0f} MB）供笔记本下载")

        # 3~4) 笔记本：下载 → 训练 → 上传
        out = retry(lambda: omen(script, [str(r), str(EPOCHS), LR], f"r{r} 笔记本"),
                    f"r{r} 笔记本")
        for line in out.strip().splitlines():
            if line.strip():
                log(f"    {line.strip()}")
        if "ROUND_DONE" not in out:
            raise SystemExit(f"第 {r} 轮笔记本步骤未完成")

        # 5) 新网推回云上
        newnet = UPLOAD / f"r{r}.nnue"
        if not newnet.is_file():
            raise SystemExit(f"第 {r} 轮没收到新网")
        retry(lambda: run(["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                           "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                           str(newnet), f"{SRV}:/root/jieqi-gen/net_r{r}.nnue"],
                          f"r{r} 推网", timeout=900), f"r{r} 推网")
        (NODES_DIR / f"r{r}.nnue").write_bytes(newnet.read_bytes())

        # 6) 记状态
        dt = time.time() - t0
        st["round"] = r
        st["best_net"] = f"/root/jieqi-gen/net_r{r}.nnue"
        st["history"].append({"round": r, "net": st["best_net"],
                              "minutes": round(dt / 60, 1),
                              "positions": int(npos)})
        save_state(st)
        log(f"--- 第 {r} 轮完成，{dt/60:.1f} 分钟，新网 {Path(st['best_net']).name} ---")

        # 轮次结束：清理本地大文件（笔记本已存副本）
        dst.unlink(missing_ok=True)
        (TRANSFER / f"r{r}.txt").unlink(missing_ok=True)

    log("=== 全部轮次完成 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
