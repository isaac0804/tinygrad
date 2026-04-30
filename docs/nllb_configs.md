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

| Mode | JIT decode (24 tokens) | Throughput |
|------|----------------------|------------|
| fp32, no beam | ~50s | ~0.48 tok/s |
| fp16 mixed, no beam | ~35s | ~0.66 tok/s |
| fp32, after BEAM=2 | TBD | TBD |
| fp16 mixed, after BEAM=2 | TBD | TBD |
