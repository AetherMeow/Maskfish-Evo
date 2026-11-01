# Upstream relationship and the one engine change

This repository is **not** a fork. It contains what we actually produced: a
training/orchestration pipeline, a trained network, and the measurements. The
engine is upstream and unmodified except for a single three-line edit described
below.

## What we built on

| Project | Role | Licence |
|---|---|---|
| [Maskfish](https://github.com/nguyenhungitc/maskfish) | engine, NNUE architecture (`HalfKAv2_hm_jieqi`), feature extraction, data generator, GPU trainer | GPL-3.0 |
| [AB-JChess](https://github.com/maimaiao/AB-JChess) | the teacher network we distilled from, and the benchmark every number here is measured against | GPL-3.0 |
| [jqe](https://github.com/nguyenhungitc/jqe) | referee and match harness used for every game reported | — |

Maskfish snapshot used: **2026-10-27**. The `ChanceModel`, `ChanceTables`,
`ChanceBranchDepth`, the `--arch` matrix (`default` / `att` / `ctx` / `threats`
and their 384 combinations), `fit_scaling.py` and `chance_tables.py` are all
**upstream Maskfish features**, not ours. We used them; we did not write them.

## The single change we made to the engine

`src/nnue/nnz_helper.h`, inside `NNZCursor::record2()`, the `else` branch of an
`if constexpr (sizeof(SIMD::vec_t) == 32)` test:

```cpp
else
{
    // 4 chunks per vector
-   static_assert(sizeof(SIMD::vec_t) == 16);
    record8(nnz_mask(neurons1) | nnz_mask(neurons2) << 4);
}
```

**Why.** `sizeof(SIMD::vec_t)` does not depend on the class template parameter
`Dimensions`, so the assertion is evaluated even though the branch is discarded
by `if constexpr`. On any build where `SIMD::vec_t` is 32 bytes (AVX2) the
`else` branch is dead but still checked, and compilation fails. Deleting the
redundant assertion enables the AVX2 path.

**Reproduce it.** Delete that one line, then build:

```bash
cd src
make -j"$(nproc)" build ARCH=x86-64-avx2        # AVX2, works on the cloud instances
```

We deliberately avoid `ARCH=native`: the generator runs on rented hardware whose
CPU changes between instances, and a native build is not portable.

## On licence compliance

We distribute a network file and Python/shell scripts — not engine binaries.
Under GPL-3.0 no source-distribution obligation is triggered by that alone, but
we license this repository under GPL-3.0 anyway, both because the network is a
derived artefact of a GPL-3.0 training pipeline and because that is the intent of
the projects this builds on.
