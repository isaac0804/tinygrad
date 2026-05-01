# Tinygrad Runtime Configs for NLLB Inference

This document covers the environment variables relevant to tuning and debugging
NLLB inference on the AMD Radeon 780M (OpenCL, `gfx1103`).

---

## Kernel Optimization

These are the primary levers for improving inference throughput.

### `BEAM` (default: `0`)

Beam search over kernel hyperparameters (tiling, unrolling, local sizes).
For each unique kernel AST, tries `N` candidate configs, benchmarks them on the
GPU, and saves the fastest to `cache.db`. Subsequent runs reuse cached results
with no re-search overhead.

```powershell
$env:BEAM="2"   # fast, good results
$env:BEAM="5"   # thorough, slower to search
```

**Notes:**
- Results are saved incrementally — safe to interrupt (Ctrl+C) and resume
- fp32 and fp16 kernels have different ASTs → separate cache entries, no conflicts
- First run is slow (minutes to hours); all subsequent runs use the cache
- `BEAM_PADTO=1` adds padding-to-multiples-of-32 to the search space (can help on AMD)

### `FUSE_OPTIM` (default: `0`)

Enables kernel fusion — merges elementwise ops (layer norm, residual adds,
activations) into adjacent matmuls, reducing memory roundtrips.

```powershell
$env:FUSE_OPTIM="1"
```

**Notes:**
- Produces different kernel ASTs than without it → needs its own `BEAM` pass
- Can reduce kernel launch count significantly for transformer decode steps
- Worth combining with `BEAM` for best results

### `NOOPT` (default: `0`)

Disables all kernel optimizations. Useful as a baseline comparison.

```powershell
$env:NOOPT="1"   # slowest, for debugging only
```

### `MV` (default: `1`)

Enables the matrix-vector heuristic — optimized path for single-token decode
steps where one dimension is 1. Already on by default.

```powershell
$env:MV="0"      # disable to test impact
$env:MV_BLOCKSIZE="4"
$env:MV_THREADS_PER_ROW="8"
$env:MV_ROWS_PER_THREAD="4"
```

---

## Compilation Cache

### `CCACHE` (default: `1`)

Master switch for the compiler disk cache. When enabled, compiled OpenCL
binaries are stored in `cache.db` and reused on subsequent runs — no
recompilation cost.

```powershell
$env:CCACHE="0"   # force recompile everything (debugging only)
```

### `CACHEDB` (default: `~/.cache/tinygrad/cache.db`)

Path to the SQLite cache database. Currently stored on C: drive.

```powershell
$env:CACHEDB="D:\tinygrad_cache\cache.db"   # redirect to D: if C: fills up
```

### `CACHELEVEL` (default: `2`)

Controls which layers of caching are active.
- `0` = no caching at all
- `1` = BEAM search results + compiled binaries cached
- `2` = full caching (default)

### `IGNORE_BEAM_CACHE` (default: `0`)

Force re-run beam search even if a cached result exists. Use after hardware
changes or to re-benchmark.

```powershell
$env:IGNORE_BEAM_CACHE="1"
```

### `ASSERT_COMPILE` (default: `0`)

Raise an error if any kernel triggers a fresh compilation. Use in production
to verify everything is cached.

```powershell
$env:ASSERT_COMPILE="1"
```

---

## Debugging & Profiling

### `DEBUG` (default: `0`)

Verbosity level for the runtime. Key levels:

| Level | Output |
|-------|--------|
| `1` | Device opens, JIT captures, kernel/copy names |
| `2` | Per-kernel timing, GFLOPS, memory bandwidth, BEAM rounds |
| `3` | Applied optimization options per kernel |
| `4` | Full kernel source code |
| `5` | Pre-lowered AST |

```powershell
$env:DEBUG="2"   # recommended for watching beam search progress
```

Example output at `DEBUG=2`:
```
*** CL  2059 r_128_2_16_4_7_2_4_4_4   arg 5  tm 181.52us  18 GFLOPS  239 GB/s
```
- Kernel name encodes its tiling shape
- `tm Xus` = execution time
- `GFLOPS` = compute throughput
- `GB/s` = memory bandwidth (transformer decode is memory-bound; higher = better)

### `BEAM_DEBUG` (default: `0`)

Controls beam search verbosity specifically.
- `0` = silent
- `1` = print AST + final winner
- `>1` = per-candidate timing table

```powershell
$env:BEAM_DEBUG="1"
```

### `PROFILE` (default: `0`)

Record CPU+GPU profiling events to `~/.cache/tinygrad/profile.pkl`. View with
the `VIZ` tool.

---

## Beam Search Tuning

These fine-tune the search space and stopping criteria.

| Variable | Default | Description |
|----------|---------|-------------|
| `BEAM_PADTO` | `0` | Add pad-to-multiples-of-32 actions to search space. Try `1` on AMD. |
| `BEAM_UPCAST_MAX` | `256` | Max product of upcast/unroll axes considered. Raise to search wider. |
| `BEAM_LOCAL_MAX` | `1024` | Max product of local/workgroup axes considered. |
| `BEAM_UOPS_MAX` | `3000` | Skip kernels with more UOps than this (avoids enormous shaders). |
| `BEAM_MIN_PROGRESS` | `0.01` | Stop searching a kernel if improvement drops below this (µs). |
| `PARALLEL` | CPU count | Worker processes for parallel candidate compilation. |

---

## Recommended Configurations

### Standard fp32 inference (after beam search complete)
```powershell
$env:TMP="D:\tmp"
$env:TEMP="D:\tmp"
python -m examples.nllb --model_path "D:\models\nllb-600M-bc-bm-250817.zip" `
    --src_lang eng_Latn --tgt_lang fra_Latn `
    --text "..."
```

### fp16 mixed-precision inference
```powershell
$env:TMP="D:\tmp"
$env:TEMP="D:\tmp"
python -m examples.nllb_fp16 --model_path "D:\models\nllb-600M-bc-bm-250817.zip" `
    --src_lang eng_Latn --tgt_lang fra_Latn `
    --text "..."
```

### fp32 beam search (run once, results cached)
```powershell
$env:TMP="D:\tmp"
$env:TEMP="D:\tmp"
$env:BEAM="2"
$env:DEBUG="2"
python -m examples.nllb --model_path "D:\models\nllb-600M-bc-bm-250817.zip" `
    --src_lang eng_Latn --tgt_lang fra_Latn `
    --text "..."
```

### fp16 beam search (run separately after fp32 beam search)
```powershell
$env:TMP="D:\tmp"
$env:TEMP="D:\tmp"
$env:BEAM="2"
$env:DEBUG="2"
python -m examples.nllb_fp16 --model_path "D:\models\nllb-600M-bc-bm-250817.zip" `
    --src_lang eng_Latn --tgt_lang fra_Latn `
    --text "..."
```

### Experimental: kernel fusion + beam search
```powershell
$env:TMP="D:\tmp"
$env:TEMP="D:\tmp"
$env:FUSE_OPTIM="1"
$env:BEAM="2"
python -m examples.nllb --model_path "D:\models\nllb-600M-bc-bm-250817.zip" `
    --src_lang eng_Latn --tgt_lang fra_Latn `
    --text "..."
```
> Note: `FUSE_OPTIM=1` produces different kernel ASTs — a separate beam search
> pass is needed and results are cached independently from the non-fused kernels.

---

## Performance Baseline (AMD Radeon 780M, gfx1103, OpenCL)

### Single-sentence (30-token budget, 23 generated tokens)

| Mode | JIT decode (23 tokens) | Throughput | vs fp32 baseline |
|------|----------------------|------------|-----------------|
| fp32, no beam | ~50s | ~0.48 tok/s | 1.00x |
| fp16 mixed, no beam | ~18.1s | ~1.27 tok/s | 2.64x |
| fp32, after BEAM=2 | TBD | TBD | TBD |
| fp16 mixed, after BEAM=2 | ~17.3s | ~1.33 tok/s | 2.77x |

> fp16 beam search (`BEAM=2`) must be run once to populate `cache.db` before
> using `beam_fp16` config — adds ~35 new beam entries on top of fp32 entries.

### Extended benchmark (13 samples, 7 language pairs, 3 length tiers, max 80 tokens)

Model: `nllb-600M-bc-bm-250817.zip` · Hardware: AMD Radeon 780M, gfx1103, OpenCL

| Language Pair | Src tok | fp16 tok/s | beam_fp16 tok/s | fuse_fp16 tok/s | fuse_beam_fp16 tok/s |
|---------------|---------|-----------|----------------|----------------|---------------------|
| English→French (short) | 9 | 0.62 | 0.37 | 0.50 | 0.43 |
| English→German (short) | 10 | 0.52 | 0.62 | 0.43 | 0.41 |
| English→Chinese (short) | 9 | 1.37 | 0.89 | 0.96 | 0.90 |
| English→French (medium) | 15 | 1.33 | 0.78 | 0.96 | 0.90 |
| English→Spanish (medium) | 20 | 1.44 | 0.90 | 1.01 | 0.94 |
| English→Arabic (medium) | 26 | 1.62 | 1.64 | 1.13 | 0.98 |
| English→Japanese (medium) | 21 | 1.34 | 1.54 | 0.92 | 0.99 |
| English→French (long) | 58 | 2.76 | 2.66 | 1.90 | 1.94 |
| English→German (long) | 57 | 1.92 | 2.08 | 1.79 | 1.90 |
| English→Chinese (long) | 67 | 2.46 | 2.04 | 1.69 | 1.60 |
| French→English (medium) | 30 | 1.31 | 1.33 | 0.84 | 0.82 |
| Spanish→English (medium) | 26 | 2.44 | 1.94 | 1.75 | 1.69 |
| Spanish→French (medium) | 24 | 1.40 | 1.12 | 0.87 | 0.92 |
| **Average** | | **1.58** | **1.38** | **1.13** | **1.11** |

**Key observations:**
- `fp16` is the fastest config overall at **1.58 tok/s** average — heuristics are already near-optimal for fp16 on this GPU
- `FUSE_OPTIM` configs are consistently **slower** (~28% below `fp16`) — kernel fusion introduces overhead that outweighs the reduced launch count on this iGPU; the fused kernels produce different ASTs and their heuristic configs are suboptimal without a dedicated FUSE_OPTIM beam search pass
- `beam_fp16` vs `fp16`: only −13% on average; occasionally faster on specific pairs (German short, Arabic, Japanese medium)
- Both `fuse_*` configs would likely improve significantly after running `BEAM=2` with `FUSE_OPTIM=1` to cache optimal kernel configs
- Throughput scales strongly with sequence length for all configs (short: ~0.4–1.4 tok/s, long: ~1.6–2.8 tok/s) — JIT amortisation dominates at short lengths
