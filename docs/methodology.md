# Methodology and engineering notes

Everything here is measured, not estimated. Where a number is an extrapolation it
is labelled as such.

---

## 1. The two match protocols, and why both are mandatory

`jqe`'s `engine/tools/match.py` can impose either a node budget or a time budget
per move. The choice changes the answer by more than 100 Elo for the same pair
of engines:

| Pair | equal nodes 20 000 | equal time 500 ms |
|---|---|---|
| Maskfish (original) vs AB-JChess | −189 | −161 |
| a5 vs AB-JChess | −189 [−222, −159] | −63 |

The engine that has the faster evaluator wins under equal time and loses under
equal nodes; both facts are true simultaneously. Concretely, our network runs at
≈ 240 k nodes/s against AB-JChess's ≈ 78 k/s on the same hardware, so at a fixed
500 ms we search roughly three times as deep.

Rules we settled on:

* **equal nodes** answers *"whose evaluation function is better?"*
* **equal time** answers *"who wins?"* — report this one in the headline.
* Never mix numbers produced under different protocols in one table.
* Use pentanomial Elo (`summary.json` → `pent`), not the naive W/L ratio: games
  are paired by opening, and the pentanomial model accounts for that.

Seed, hash (64 MB) and thread count (1) are held fixed across every match in
this repository.

---

## 2. What the RL loop actually optimises

The label for a position is the search score of the *current* network at 5 000
nodes; the student is the same network, distilled from its own search. That is
search distillation inside a self-play loop:

```
target = λ · sigmoid(score / score_scaling) + (1 − λ) · (result + 1)/2
```

with `λ = 1.0` here. The fixed point is a network that already agrees with its
own search — at which point the loop stops improving. We are nowhere near it:
+6.6 Elo/round and not decaying.

`λ < 1` (blending in game results) was tested and is *worse*: rla (λ=1.0) vs rlb
(λ=0.9) over 2000 games = −7 [−22, +7], i.e. no difference, and the outcome
label carries 17–25× more variance than the search label because of hidden
pieces. Do not bother.

**Score-scale calibration matters and must be redone every round.** Each round's
positions come from a different network with a different score distribution; we
fit `sigmoid(score / s)` by logistic regression on the round's own data
(`fit_scaling.py --by-material 8`) and use that `s` for training. The fitted
value drifted 415.8 → 403.0 over nine rounds. Reusing a stale scale silently
shrinks or stretches every label.

**How much to train per round.** Six passes over the round's data, warm-started
from the previous round's network at `lr = 2e-4`, `gamma = 0.995`. In practice
the best-validation epoch is almost always 1 or 2, i.e. the network is already
close to the new distribution and further passes only overfit it.

---

## 3. The dataset is the product

Nine rounds produced 86 580 962 positions. Every one of them was kept:

| round | positions | note |
|---|---|---|
| r1 | 9 307 018 | generated before an account suspension; recovered from the instance disk |
| r2 | 9 508 519 | |
| r3 | 9 479 111 | |
| r4 | 9 583 610 | |
| r5 | 9 661 081 | |
| r6 | 9 697 981 | |
| r7 | 9 742 325 | |
| r8 | 9 820 841 | |
| r9 | 9 780 476 | |

At the end a network trained on the *static union* of all nine rounds (6 passes,
520 M samples) was compared to the network produced by the *sequential* loop.
Both consumed the same number of samples; the difference is that the loop's data
distribution moves with the network. This is the only experiment in the project
that isolates "iterate" from "just train more".

---

## 4. Cost model

| Item | Rate | Used | Cost |
|---|---|---|---|
| ecs.c9i.12xlarge (48 vCPU, 96 GiB), pay-as-you-go | ¥9.2955/h | ≈ 15 h | ≈ ¥139 |
| spot price for the same instance | ¥3.388/h | first attempt | ≈ ¥5 (lost) |
| public egress | ¥0.8/GB | ≈ 15 GB | ≈ ¥12 |
| **total** | | | **¥38–46** |

Throughput on that instance: 6 698 620 positions/hour at 40 workers, measured
with a 12 000-game run before the pipeline was trusted with a long job.

Cost per round: ≈ ¥16 for 9.5 M positions plus 15 minutes of laptop GPU time.
Cost per Elo: **≈ ¥0.6**.

The 48-vCPU shape was chosen because 32 vCPU (16 physical cores) is no faster
than the laptop and therefore pointless; 48 vCPU is the smallest size that
produces a real speed-up at all. Spot is 2.7× cheaper and, with chunked
generation, defensible — but a reclamation costs an operator round-trip, so
pay-as-you-go is the right default for an unattended overnight run.

---

## 5. Failure modes encountered, in order of how much they cost

1. **Account suspension with a zero cash balance** (¥5 + 100 minutes). Vouchers
   cannot pay arrears. Recovered because suspension preserves the disk; the
   9.3 M positions were pulled afterwards.
2. **Spot reclamation between generation and transfer** (¥5 + 90 minutes). Fixed
   by chunking: 7 chunks of 40 k games, each compressed, pulled and merged
   locally before the next one starts. `state.json` tracks how many chunks are
   merged so a restart neither duplicates nor skips data.
3. **Killing the orchestrator left remote `gen_data.py` processes running**, which
   then raced a freshly started run on the same output file (load 65 on a 48-core
   box). Always stop the remote side first.
4. **The watchdog itself must be a managed job.** A `nohup`-ed process that only
   writes a log file notifies nobody; on a metered instance that is money burning
   silently. `watch.py` exits with a distinct code per failure mode so the exit
   event itself carries the diagnosis.
5. **Remote training survives its SSH session.** When the laptop rebooted
   mid-round, the cloud-side data had already been transferred and the *laptop*
   kept training; on reconnect the round was complete and only the upload was
   missing. Resume logic has to tolerate "the step already finished without me".

---

## 6. Reproducibility checklist

* Same seed for every match reported (`20261026`).
* Same opening book and paired openings (pentanomial).
* Single-threaded engines, 64 MB hash, no tablebase.
* Both protocols reported for the headline claim.
* Confidence intervals from the harness, quoted as measured — including the ones
  that straddle zero.
