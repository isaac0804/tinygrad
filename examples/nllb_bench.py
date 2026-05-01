#!/usr/bin/env python3
"""
NLLB inference benchmark — compares multiple configurations:

  1. Baseline   — no beam cache (NOOPT=1), fp32
  2. Default    — heuristic opts (no BEAM), fp32
  3. Beam       — with beam search cache, fp32
  4. FP16       — mixed precision, heuristic opts
  5. Beam+FP16  — beam cache + mixed precision
  6. Fuse       — FUSE_OPTIM=1, fp32
  7. Fuse+Beam  — FUSE_OPTIM=1 + beam cache, fp32

Each config runs N_TOKENS decode steps and reports:
  - encode time
  - first decode step time
  - JIT decode time
  - tokens/sec
  - translation (for correctness check)

Usage:
  python examples/nllb_bench.py --model_path "D:\\models\\nllb-600M-bc-bm-250817.zip"

  # Run only specific configs:
  python examples/nllb_bench.py --model_path "..." --configs default beam fp16

  # More decode tokens for stable timing:
  python examples/nllb_bench.py --model_path "..." --tokens 50
"""

import argparse, time, os, gc
from pathlib import Path
from typing import List, Optional, Union
from contextlib import contextmanager

import numpy as np

TEXT    = "The United Nations Chief says there is no military solution in Syria"
SRC     = "eng_Latn"
TGT     = "fra_Latn"
EXPECTED = "Le Chef des Nations Unies dit qu'il n'y a pas de solution militaire en Syrie"

# ---------------------------------------------------------------------------

@contextmanager
def tinygrad_env(**kwargs):
  """Temporarily set tinygrad ContextVars via os.environ before import."""
  # We cannot change ContextVars after tinygrad is imported, so each config
  # is run in a subprocess. This context manager sets env vars for the
  # current process (used when running a single config directly).
  old = {}
  for k, v in kwargs.items():
    old[k] = os.environ.get(k)
    if v is None:
      os.environ.pop(k, None)
    else:
      os.environ[k] = str(v)
  try:
    yield
  finally:
    for k, v in old.items():
      if v is None:
        os.environ.pop(k, None)
      else:
        os.environ[k] = v


def run_inference(model_path: str, max_tokens: int, fp16: bool = False,
                  noopt: bool = False, fuse: bool = False) -> dict:
  """
  Run a single inference pass and return timing dict.
  Must be called after setting env vars (NOOPT, FUSE_OPTIM, BEAM, etc.)
  because tinygrad reads them at import time via ContextVar.
  """
  # Late imports so env vars are already set
  # Ensure repo root is on sys.path (needed when run as a subprocess)
  import sys as _sys
  _repo_root = str(Path(__file__).resolve().parent.parent)
  if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

  from tinygrad import Tensor, TinyJit, Variable, dtypes
  from tinygrad.nn.state import get_state_dict
  from extra.models.nllb import NLLBConfig, NLLBModel, remap_weights
  from examples.nllb import NLLBTokenizer, load_weights_from_zip
  from examples.nllb_fp16 import NLLBModelFP16, cast_to_fp16

  Tensor.training = False

  # Build model
  config = NLLBConfig()
  if fp16:
    model = NLLBModelFP16(config, max_cache_len=max_tokens + 10)
  else:
    model = NLLBModel(config, max_cache_len=max_tokens + 10)

  # Load weights
  load_weights_from_zip(model, model_path)

  if fp16:
    cast_to_fp16(model)

  # Tokenize
  tokenizer = NLLBTokenizer(model_path)
  input_ids, attention_mask = tokenizer.encode(TEXT, SRC)
  forced_bos = tokenizer.get_lang_id(TGT)
  eos_id = tokenizer.eos_token_id

  src_ids  = Tensor(np.array([input_ids],     dtype=np.int32))
  src_mask = Tensor(np.array([attention_mask], dtype=np.int32))

  # Encode
  t0 = time.perf_counter()
  enc_out = model.encode(src_ids, src_mask).realize()
  encode_ms = (time.perf_counter() - t0) * 1000

  # First decode step (warm-up, cross-attn cache)
  generated = [eos_id, forced_bos]
  dec_in0 = Tensor(np.array([generated], dtype=np.int32))
  t0 = time.perf_counter()
  logits0 = model.decode(dec_in0, enc_out, src_mask, start_pos=0)
  next_token = int(logits0[0, -1].argmax(axis=-1).numpy())
  first_decode_ms = (time.perf_counter() - t0) * 1000
  generated.append(next_token)

  # JIT decode loop
  max_start  = max_tokens + len(generated)
  decode_jit = TinyJit(lambda tok, sp: model.decode(tok, enc_out, src_mask, start_pos=sp))

  t0 = time.perf_counter()
  for _ in range(1, max_tokens):
    sp     = Variable("start_pos", 1, max_start).bind(len(generated) - 1)
    dec_in = Tensor(np.array([[generated[-1]]], dtype=np.int32))
    logits = decode_jit(dec_in, sp)
    next_token = int(logits[0, -1].argmax(axis=-1).numpy())
    generated.append(next_token)
    if next_token == eos_id:
      break
  jit_ms = (time.perf_counter() - t0) * 1000

  output_ids  = generated[2:]
  translation = tokenizer.decode(output_ids)
  jit_tokens  = len(output_ids) - 1  # exclude first-step token
  tok_per_s   = jit_tokens / (jit_ms / 1000) if jit_ms > 0 else 0

  return {
    "encode_ms":       encode_ms,
    "first_decode_ms": first_decode_ms,
    "jit_ms":          jit_ms,
    "jit_tokens":      jit_tokens,
    "tok_per_s":       tok_per_s,
    "translation":     translation,
    "correct":         translation.strip() == EXPECTED.strip(),
  }


# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------

CONFIGS = {
  "baseline": {
    "label":       "Baseline (NOOPT, fp32)",
    "env":         {"NOOPT": "1"},
    "fp16":        False,
    "fuse":        False,
    "description": "All optimizations disabled. Slowest possible — pure reference.",
  },
  "default": {
    "label":       "Default (heuristic opts, fp32)",
    "env":         {},
    "fp16":        False,
    "fuse":        False,
    "description": "Tinygrad's built-in heuristic kernel configs. No beam search.",
  },
  "beam": {
    "label":       "Beam (BEAM cache, fp32)",
    "env":         {},
    "fp16":        False,
    "fuse":        False,
    "description": "Uses beam search results from cache.db. Run BEAM=2 first to populate.",
  },
  "fp16": {
    "label":       "FP16 (mixed precision, heuristic opts)",
    "env":         {},
    "fp16":        True,
    "fuse":        False,
    "description": "Linear weights in fp16, layer norms + lm_head in fp32.",
  },
  "beam_fp16": {
    "label":       "Beam+FP16 (beam cache + mixed precision)",
    "env":         {},
    "fp16":        True,
    "fuse":        False,
    "description": "fp16 mixed precision with beam-searched kernel configs.",
  },
  "fuse": {
    "label":       "Fuse (FUSE_OPTIM, fp32)",
    "env":         {"FUSE_OPTIM": "1"},
    "fp16":        False,
    "fuse":        True,
    "description": "Kernel fusion enabled. Run BEAM=2 FUSE_OPTIM=1 first for best results.",
  },
  "fuse_beam": {
    "label":       "Fuse+Beam (FUSE_OPTIM + beam cache, fp32)",
    "env":         {"FUSE_OPTIM": "1"},
    "fp16":        False,
    "fuse":        True,
    "description": "Kernel fusion + beam search cache.",
  },
}

# ---------------------------------------------------------------------------
# Subprocess runner — each config needs a fresh Python process so that
# tinygrad ContextVars are initialised with the correct env vars.
# ---------------------------------------------------------------------------

def run_config_subprocess(config_name: str, model_path: str,
                          max_tokens: int) -> dict:
  """Run a single config in a subprocess and return its result dict."""
  import subprocess, json, sys

  cfg = CONFIGS[config_name]
  env = os.environ.copy()
  env.update(cfg["env"])
  # Tell the child which config to run
  env["_BENCH_CONFIG"]     = config_name
  env["_BENCH_MODEL_PATH"] = model_path
  env["_BENCH_MAX_TOKENS"] = str(max_tokens)

  # Ensure the repo root is on PYTHONPATH for the child process
  repo_root = str(Path(__file__).resolve().parent.parent)
  pythonpath = env.get("PYTHONPATH", "")
  env["PYTHONPATH"] = repo_root + (os.pathsep + pythonpath if pythonpath else "")

  cmd = [sys.executable, __file__, "--_run_config"]
  result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=600)
  if result.returncode != 0:
    return {"error": result.stderr[-2000:]}
  try:
    # Last line of stdout is the JSON result
    lines = [l for l in result.stdout.strip().splitlines() if l.startswith("{")]
    return json.loads(lines[-1])
  except Exception as e:
    return {"error": f"Could not parse output: {e}\nstdout: {result.stdout[-1000:]}"}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_result(name: str, cfg: dict, result: dict) -> str:
  if "error" in result:
    return f"  {'ERROR':<12} {result['error'][:80]}"
  correct = "OK" if result["correct"] else "WRONG"
  return (
    f"  encode:       {result['encode_ms']:7.0f} ms\n"
    f"  first decode: {result['first_decode_ms']:7.0f} ms\n"
    f"  JIT decode:   {result['jit_ms']:7.0f} ms  ({result['jit_tokens']} tokens)\n"
    f"  throughput:   {result['tok_per_s']:7.2f} tok/s\n"
    f"  translation:  {result['translation']}\n"
    f"  correct:      {correct}"
  )


def print_summary(results: dict):
  print("\n" + "="*70)
  print("SUMMARY")
  print("="*70)
  print(f"{'Config':<20} {'tok/s':>8}  {'JIT ms':>8}  {'encode ms':>10}  {'OK?':>5}")
  print("-"*70)

  # Find best tok/s for relative comparison
  valid = {k: v for k, v in results.items() if "tok_per_s" in v}
  if not valid:
    print("  No valid results.")
    return
  best_tps = max(v["tok_per_s"] for v in valid.values())
  baseline_tps = valid.get("default", {}).get("tok_per_s", None)

  for name, result in results.items():
    if "error" in result:
      print(f"  {name:<20} {'ERROR':>8}")
      continue
    tps     = result["tok_per_s"]
    jit_ms  = result["jit_ms"]
    enc_ms  = result["encode_ms"]
    correct = "YES" if result["correct"] else "NO"
    speedup = f" ({tps/baseline_tps:.2f}x)" if baseline_tps and baseline_tps > 0 else ""
    print(f"  {name:<20} {tps:>7.2f}{speedup:<8}  {jit_ms:>8.0f}  {enc_ms:>10.0f}  {correct:>5}")

  print(f"\n  Best: {max(valid, key=lambda k: valid[k]['tok_per_s'])}"
        f" at {best_tps:.2f} tok/s")
  if baseline_tps:
    print(f"  Baseline (default): {baseline_tps:.2f} tok/s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(description="NLLB benchmark across configurations")
  parser.add_argument("--model_path", type=str,
                      default=r"D:\models\nllb-600M-bc-bm-250817.zip")
  parser.add_argument("--tokens", type=int, default=30,
                      help="Number of decode tokens (more = more stable timing)")
  parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                      choices=list(CONFIGS.keys()),
                      help="Which configs to run (default: all)")
  parser.add_argument("--_run_config", action="store_true",
                      help=argparse.SUPPRESS)  # internal: run single config
  args = parser.parse_args()

  # Internal mode: child process running a single config
  if args._run_config:
    import json
    config_name = os.environ["_BENCH_CONFIG"]
    model_path  = os.environ["_BENCH_MODEL_PATH"]
    max_tokens  = int(os.environ["_BENCH_MAX_TOKENS"])
    cfg = CONFIGS[config_name]
    result = run_inference(model_path, max_tokens, fp16=cfg["fp16"], fuse=cfg["fuse"])
    print(json.dumps(result))
    return

  # Parent mode: run each config in a subprocess
  print(f"NLLB Benchmark")
  print(f"  model:   {args.model_path}")
  print(f"  text:    {TEXT}")
  print(f"  tokens:  {args.tokens}")
  print(f"  configs: {', '.join(args.configs)}")
  print(f"  expected: {EXPECTED}")
  print()

  results = {}
  for name in args.configs:
    cfg = CONFIGS[name]
    print(f"[{name}] {cfg['label']}")
    print(f"  {cfg['description']}")
    t_start = time.perf_counter()
    result  = run_config_subprocess(name, args.model_path, args.tokens)
    elapsed = time.perf_counter() - t_start
    results[name] = result
    print(fmt_result(name, cfg, result))
    print(f"  wall time: {elapsed:.1f}s")
    print()

  print_summary(results)


if __name__ == "__main__":
  main()
