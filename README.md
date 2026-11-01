# Jieqi NNUE: beating the teacher with self-play RL

An experiment in **reveal chess (揭棋 / Jieqi)** on top of
[Maskfish](https://github.com/nguyenhungitc/maskfish) (a Stockfish derivative with
Jieqi rules) and [AB-JChess](https://github.com/maimaiao/AB-JChess) (a strong
Jieqi engine with a publicly released NNUE).

The result: starting from a teacher-distilled network that was **161 Elo weaker**
than AB-JChess at equal time, nine rounds of self-play reinforcement learning —
run unattended on a rented 48-vCPU spot instance plus a laptop GPU for **¥38–46
total** — produced a network that is **statistically level at equal nodes and
ahead at equal time**.

| Network | vs AB-JChess @ 500 ms (equal time) |
|---|---|
| original Maskfish net | −161 |
| e4 — distilled from AB, 4.67 M positions | −100 |
| a5 — distilled, 8.68 M positions + calibrated | −63 |
| **r9 — 9 rounds of self-play RL** | **+27 [+3, +50]** |

| Network | vs AB-JChess @ 20 000 nodes (equal nodes) |
|---|---|
| rl2net — starting point of the RL loop | −100 |
| **r9 — after 9 rounds** | **−16 [−39, +7]** |

Net improvement over the whole project: **−161 → +27, i.e. +188 Elo**.
The RL loop alone accounted for roughly +90 of it, for about ¥40 of cloud spend.

---

## What this is, and what it is not

**It is** a complete, reproducible pipeline for self-play RL on an NNUE engine:
a Rust/Python data generator running on cheap cloud CPU, a GPU trainer on a
consumer laptop, and an orchestrator that survives spot-instance reclamation,
account suspension, reboots and network drops.

**It is not** a claim of superiority over AB-JChess. 800 games at 500 ms give
+27 Elo with a lower confidence bound of +3: that is "just crossed the line",
not dominance. At longer time controls AB-JChess may well be stronger — we
measured a shorter time control only. Treat the headline number as *"the
distillation gap has been closed"*, not as *"our engine wins"*.

---

## The pipeline

```
cloud (48 vCPU, ¥9.3/h)              home box (orchestrator)         laptop (GPU)
──────────────────────               ──────────────────────          ───────────
gen_data.py, 40 workers   ──scp──▶   receives 7 chunks per round
  self-play at 5000 nodes             zstd-compressed, resumable
  ~6.7 M positions/hour                        │
                                               │  HTTP over overlay VPN
                                               ▼
                                      (10.20.0.1:18765)     ──────▶  prepare_data.py
                                                                    train.py (CUDA)
                                                                    6 epochs, λ = 1.0
                                                                           │
                                      PUT /upload/<net>      ◀────────────┘
                                               │
                                      scp ─────┴────▶ becomes the generator
                                                      for the next round
```

One round: 9.5 M self-play positions (≈ 100 minutes on the cloud) plus
≈ 15 minutes of GPU training. We ran 9 rounds; 86.6 M positions were generated
in total and all of it was kept.

`code/` contains the three pieces that matter:

| File | Runs on | Role |
|---|---|---|
| `orchestrator.py` | home box | drives the loop; splits generation into 7 chunks, pulls each one as soon as it is ready, resumes from `state.json` |
| `watch.py` | home box | watchdog, runs as a *managed background job* so that its exit notifies the operator; detects stalls, orchestrator death, and cloud unreachability (spot reclamation) |
| `omen_round.sh` | laptop | one round of training: download → fit score scale → prepare features → train → pick epoch → upload |
| `deploy_server.sh` | cloud | one-shot provisioning + **throughput benchmark before committing to a run** |
| `transfer_server.py` | home box | tiny HTTP server for laptop→home transfers (an SSH pipe through `cmd.exe` corrupts binary data; HTTP does not) |

### Throughput measured on the rented instance

```
48 vCPU (Intel Xeon 6982P-C, Granite Rapids P-core)
6,698,620 positions/hour with 40 workers
= 2.09× a 24-thread laptop (3.2 M/h)
```

That benchmark is worth repeating before any long run: the same nominal instance
type can be 30 % slower in a different availability zone, and the whole cost
model rests on that number.

---

## Results

All numbers are **pentanomial Elo** from
[`jqe`](https://github.com/nguyenhungitc/jqe)'s match harness, single-threaded
engines, 64 MB hash, fixed seed. Two protocols were used and **they disagree by
135 Elo**, which is why both are reported:

* **equal nodes** (20 000) isolates evaluation quality,
* **equal time** (500 ms) is what a human actually plays against.

### Self-play RL gains (equal nodes)

| Comparison | Games | Score | Elo |
|---|---|---|---|
| r4 vs rl2net (start) | 1695 | 0.555 | +38 [+23, +54] |
| r9 vs rl2net (start) | 1080 | 0.592 | +65 [+44, +86] |
| r9 vs r4 | 1600 | 0.547 | +33 [+17, +50] |

+65 Elo over 9 rounds, and the rate did not decay: rounds 2–4 gave +7.0/round,
rounds 5–9 gave +6.6/round.

### Architecture: does a 4× wider net pay off with data?

`att` = 6 king buckets × 4 attack buckets = 24 buckets, 30 240 inputs, 62 M
parameters — the same bucket structure as AB-JChess (which uses 34 800 inputs
and an unknown training corpus). `default` = 6 × 1 buckets, 15 M parameters.

Both trained from scratch, **identical data, identical budget, identical seed,
identical learning-rate schedule**:

| Training data | `att` vs `default` |
|---|---|
| 10.2 M positions | **−10.7 ± 9.7** (4000 games) |
| **86.6 M positions** (6 passes, 520 M samples) | **+13 [−8, +34]** (1000 games) |

The sign flips, a +24 Elo swing, which supports "capacity needs data" — but +13
with a lower bound of −8 is **not significant**. Extrapolating the observed rate
(≈ +24 Elo per 8.5× data) puts the crossover's payoff at +40 only around
700 M positions, i.e. 8× more data than we had.

Also worth knowing: `att` trains at 109 k positions/s vs `default`'s 111 k — a
4× parameter increase costs **2 % throughput** on this setup, because the data
pipeline is the bottleneck, not the GPU. The real costs are size (46.8 MB vs
10.8 MB per network) and VRAM (7.2 GB vs 5.8 GB peak on a 12 GB card, which
rules out a further 4×).

**Conclusion: do not switch to the wider net at this data volume.** Continue RL
rounds instead; the measured +6.6 Elo/round is far cheaper than 8× the data.

---

## Traps we fell into (so you do not have to)

**Validation loss is not a strength proxy.** In the RL loop it *rises* while the
engine gets stronger:

```
round   best epoch   val loss      cumulative Elo (equal nodes)
  r1        2        0.006807              +17
  r2        2        0.007319              —
  r3        1        0.007703              —
  r4        1        0.008079              +38
  r9        2        0.008512              +65
```

Four independent occasions confirmed this. The reason is structural: the
validation set comes from the *previous* distribution, while each RL round
deliberately trains the network away from it. In an RL loop, val loss measures
"distance from the old distribution" — which is the objective, not the cost.
Use games. Pick the epoch by val loss only as a cheap tie-breaker, and expect
"epoch 1 or 2" to win almost every time.

**Equal nodes and equal time are different questions.** The same pair of engines
measured −189 and −63 under the two protocols. Reporting only one is how a
project convinces itself it is 3× further ahead (or behind) than it is.

**A spot instance reclaimed in the middle of a round loses the whole round.**
Generate in chunks and pull each chunk as soon as it exists; the loss window
goes from 100 minutes to 13. Our first attempt lost 9.3 M positions this way
(they were recovered only because the instance was *suspended for arrears*
rather than released).

**An Alibaba Cloud voucher cannot pay arrears** — `优惠券无法用于抵扣欠费`.
Pay-as-you-go deducts from your cash balance hourly and applies vouchers only at
billing time, so a zero balance suspends the account a few hours into the run no
matter how large the voucher is. Keep a small cash buffer.

**Killing the orchestrator does not kill its remote SSH children.** After
`kill`, the cloud-side `gen_data.py` keeps generating into the same file as the
freshly started one. Stop the remote side first.

**An SSH pipe through `cmd.exe` corrupts binary data.** Push and pull files over
HTTP or base64, never through a Windows shell pipeline.

---

## Reproducing

```bash
# 1. cloud: provision and benchmark (refuses to continue if throughput is poor)
scp code/deploy_server.sh root@<cloud>:/root/ && ssh root@<cloud> 'bash /root/deploy_server.sh'

# 2. edit the constants at the top of code/orchestrator.py, then
ROUNDS=9 GAMES=280000 CHUNKS=7 EPOCHS=6 LR=2e-4 python3 code/orchestrator.py &

# 3. watch it (its exit wakes you: completion, crash, stall, or cloud gone)
ROUNDS=9 python3 code/watch.py
```

Requirements: Maskfish (engine + `trainer/`), `jqe` (referee and match harness),
a CUDA GPU, and any Linux box that can reach both ends.

Generation needs **only the Python standard library** plus a C++ compiler for
the engine — no numpy, no torch. The trainer needs PyTorch and a GPU.

---

## Repository layout

```
code/            orchestrator, watchdog, per-round trainer script, provisioning,
                 HTTP transfer server
nets/            maskfish-jieqi-r9.nnue  — the best network (+27 Elo vs AB)
results/         raw summary.json from every match reported above
docs/            methodology, and the engineering notes behind the numbers
```

## Licence and upstream

GPL-3.0 — see `LICENSE`.

This is **not a fork of Maskfish**. It contains only what we produced: the
pipeline, the network, and the measurements. The engine is upstream, unmodified
except for one three-line edit needed to make the AVX2 build compile. Both that
edit and the exact upstream snapshot are documented in
[`docs/upstream.md`](docs/upstream.md).

## Credits

* [Maskfish](https://github.com/nguyenhungitc/maskfish) — engine, NNUE
  architecture and trainer this work is built on.
* [AB-JChess](https://github.com/maimaiao/AB-JChess) — the teacher network, and
  the benchmark every number above is measured against.
* [jqe](https://github.com/nguyenhungitc/jqe) — referee and match harness used
  for every game reported here.
