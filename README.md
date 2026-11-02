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

**It is not** a claim of dominance over AB-JChess. The advantage is real but
small: +20 to +27 Elo across 200–2000 ms, and only the 500 ms point has a
confidence interval that excludes zero. It is also flat, not growing — which is
itself informative, because the *distillation* baseline lost more ground as the
time control lengthened. Treat the headline as *"the distillation gap has been
closed and slightly reversed"*, not as *"our engine wins"*.

### Time-control dependence

Our earlier distillation attempts were sensitive to the time control: AB-JChess
pulled further ahead from 200 ms to 500 ms. After the RL loop, the gap is
roughly constant:

| Time control | r9 vs AB-JChess | Games |
|---|---|---|
| 200 ms | +21 [−5, +47] | 600 |
| **500 ms** | **+27 [+3, +50]** | 800 |
| 2000 ms | +20 [−13, +53] | 400 |

A 10× spread in thinking time produces no trend. The practical reading: on this
hardware the network is worth roughly the same amount of Elo whatever the clock,
which is what one expects if the gain comes from evaluation quality rather than
from a search-speed artefact.

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

### Does iterating matter, or just the sample count?

The RL loop's nine rounds are nine different data distributions: each round is
generated by the network the previous round produced. A natural objection —
raised by a reader of the first version of this document — is that the apparent
benefit of iterating might be an artefact of everything *else* that differs
between the loop and a single training run: the warm start, the learning-rate
schedule, the fact that each round keeps its best-validation checkpoint while a
one-shot run keeps its last, and the sample count.

That objection was right. Three controlled follow-ups dismantled the conclusion:

| Experiment | Design | vs `cloudr9` (the loop) |
|---|---|---|
| `a6def` | from scratch, static union, 6 passes (520 M samples), keeps epoch 6 | **−21 [−42, −1]** |
| E3 | `a6def` epoch 3 **vs** epoch 6 — same network, different checkpoint | epoch 3 is **+14 [−11, +40]** better |
| E1 (`scr9`) | from scratch, **last round only** (58.6 M samples) | **−55 [−76, −35]** |
| E2' (`unionwarm`) | **warm from r8**, union data, **same 58.6 M samples** as one loop round | **−4 [−24, +15]** |

Read them in order:

* **E2' is the decisive one.** It holds the warm start, the learning rate and the
  sample count fixed and varies *only* the data composition — the loop's fresh
  on-policy round versus a random subsample of the nine-round union. Result: **no
  difference (−4, CI straddling zero).** Mixing in eight rounds of stale,
  weaker-generator data costs nothing.
* **E3 removes most of `a6def`'s deficit.** The union run kept its final epoch;
  epoch 3 was 14 Elo better. Roughly two thirds of the original −21 was an
  overfit checkpoint, not a training-structure effect.
* **E1 shows what does matter: total data.** A from-scratch network on the last
  round's data alone (58.6 M samples) is 55 Elo weaker. Note the confound — it
  also saw fewer samples than the loop's ≈144 M — so this bounds "data volume
  matters" rather than isolating the warm start.

**Corrected conclusion: iterating over a moving distribution does not beat one
training run over the union of what that iteration produced.** The first version
of this section claimed it did; that claim rested on a comparison with four
simultaneous differences, and it is withdrawn.

What remains true, and is still the reason to run the loop: the loop produces
**progressively better-labelled data**, because each round's labels come from a
stronger searcher than the previous round's. That is genuine self-improvement and
it is what took the network from −100 to −16 at equal nodes. But the *training*
need not be interleaved with generation: one run at the end over the accumulated
corpus does the same job, which is simpler and leaves the GPU idle instead of
cycling it.

One caveat on the mechanism, measured afterwards: on 2000 real training positions,
a network 65 Elo stronger produces labels that differ from the weaker network's by
only 1.3 % of win probability on average, and the two agree within 5 % on 98.6 % of
positions. So "better labels" is a small per-position effect that accumulates over
86 M positions — which is also why mixing in eight rounds of stale data cost
nothing (E2'). Relabelling at 50 000 nodes instead of 5 000 moved both networks
closer to an external judge (Spearman +0.014 / +0.011 against AB-JChess's deep
search), i.e. **label depth matters more than which network produces the label** —
but that comparison could not be made unbiased: with AB-JChess as judge, the
AB-distilled `rl2net` looked closer (60 % paired), and with `r9`'s own deep search
as judge, `r9` looked closer (73 % paired). Each oracle is correlated with one of
the two candidates, so neither ranks them honestly. Only games settle it.

The experiment that would settle "better labels versus more labels" — generating
the same 86.6 M positions with the starting network only, then training once — was
deliberately not run. It is worth being explicit about why: the answer is
predictable a priori (labels from a 65-Elo-weaker searcher cannot be *better*, or
the network would be pointless), and the money is better spent on more RL rounds,
which buy a measured +6.6 Elo each. The question is deferred, not answered.

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

## The network

`nets/maskfish-jieqi-r9.nnue` — the final network of the loop.

```
size    10,882,319 bytes
md5     37d0d78242a507d9bdc5c73f282f7ee6
sha256  24901914342726f56b0915892fc8da0b2406c48076fbf8f1c183286581071cd1
arch    (7560, 2048, 15, 32, 1)     # HalfKAv2_hm_jieqi, default bucket shape
```

Verified to load and search with Maskfish at the documented upstream snapshot:

```
info string NNUE evaluation using maskfish-jieqi-r9.nnue (15MiB, (7560, 2048, 15, 32, 1))
info depth 1 ... score cp 11 nodes 1283 nps 183285 pv i3i4
```

Use it with `setoption name EvalFile value <path>`; no rebuild is needed, the
default build already accepts this architecture. The `att` networks from the
architecture experiment are not included — they are 46.8 MB each and lost the
argument they were built to win.

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
