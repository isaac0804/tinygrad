#!/usr/bin/env python3
"""
NLLB extended benchmark — multiple samples, multiple language pairs, longer sequences.

Tests fp16 and beam_fp16 configs across a diverse set of translation tasks:
  - Short sentences (~10 tokens)
  - Medium sentences (~25 tokens)
  - Long sentences (~50+ tokens)
  - Multiple target languages (French, Spanish, German, Chinese, Arabic, Japanese)
  - Multiple source languages (English, French, Spanish)

Usage:
  python examples/nllb_bench_extended.py --model_path "D:\\models\\nllb-600M-bc-bm-250817.zip"
  python examples/nllb_bench_extended.py --model_path "..." --configs fp16
  python examples/nllb_bench_extended.py --model_path "..." --max_tokens 100
"""

import argparse, time, os, json, sys
from pathlib import Path

import numpy as np

MODEL_PATH_DEFAULT = r"D:\models\nllb-600M-bc-bm-250817.zip"

# ---------------------------------------------------------------------------
# Test corpus: (src_lang, tgt_lang, source_text)
# ---------------------------------------------------------------------------

SAMPLES = [
  # --- Short (~8-12 source tokens) ---
  ("eng_Latn", "fra_Latn",
   "The cat sat on the mat."),
  ("eng_Latn", "deu_Latn",
   "Good morning! How are you today?"),
  ("eng_Latn", "zho_Hans",
   "She loves reading books every evening."),

  # --- Medium (~20-30 source tokens) ---
  ("eng_Latn", "fra_Latn",
   "The United Nations Chief says there is no military solution in Syria."),
  ("eng_Latn", "spa_Latn",
   "Scientists have discovered a new species of deep-sea fish near the Pacific Ocean floor."),
  ("eng_Latn", "arb_Arab",
   "Climate change is one of the most pressing challenges facing humanity in the twenty-first century."),
  ("eng_Latn", "jpn_Jpan",
   "The stock market experienced significant volatility following the central bank's interest rate announcement."),

  # --- Long (~40-60 source tokens) ---
  ("eng_Latn", "fra_Latn",
   "The World Health Organization has warned that antimicrobial resistance is one of the greatest threats "
   "to global health, food security, and development, urging governments to take immediate action to "
   "regulate the use of antibiotics in both human medicine and agriculture."),
  ("eng_Latn", "deu_Latn",
   "Researchers at the Massachusetts Institute of Technology have developed a new type of battery "
   "that can store renewable energy for months at a time without significant loss, potentially "
   "revolutionizing how solar and wind power can be used to supply electricity during periods of low generation."),
  ("eng_Latn", "zho_Hans",
   "The ancient Silk Road was a network of trade routes connecting China to the Mediterranean world, "
   "facilitating not only the exchange of goods such as silk, spices, and precious metals, "
   "but also the spread of religions, technologies, and artistic influences across Eurasia for over a millennium."),

  # --- Non-English source ---
  ("fra_Latn", "eng_Latn",
   "La France est connue pour sa cuisine raffinée, ses vins exceptionnels et son patrimoine culturel incomparable."),
  ("spa_Latn", "eng_Latn",
   "El cambio climático es una amenaza global que requiere una respuesta coordinada de todos los países del mundo."),
  ("spa_Latn", "fra_Latn",
   "La inteligencia artificial está transformando rápidamente la manera en que las empresas operan y toman decisiones."),
]


# ---------------------------------------------------------------------------
# Config definitions (subset of nllb_bench.py)
# ---------------------------------------------------------------------------

CONFIGS = {
  "fp16": {
    "label": "FP16 (heuristic opts)",
    "env":   {},
    "fp16":  True,
  },
  "beam_fp16": {
    "label": "Beam+FP16 (beam cache + mixed precision)",
    "env":   {},
    "fp16":  True,
    "use_beam_cache": True,
  },
  "fuse_fp16": {
    "label": "FP16 + FUSE_OPTIM (kernel fusion, heuristic opts)",
    "env":   {"FUSE_OPTIM": "1"},
    "fp16":  True,
  },
  "fuse_beam_fp16": {
    "label": "FP16 + FUSE_OPTIM + Beam cache",
    "env":   {"FUSE_OPTIM": "1"},
    "fp16":  True,
    "use_beam_cache": True,
  },
}


# ---------------------------------------------------------------------------
# Single-config inference over all samples
# ---------------------------------------------------------------------------

def run_all_samples(model_path: str, max_tokens: int, fp16: bool, use_beam_cache: bool = False) -> list:
  import sys as _sys
  repo_root = str(Path(__file__).resolve().parent.parent)
  if repo_root not in _sys.path:
    _sys.path.insert(0, repo_root)

  from tinygrad import Tensor, TinyJit, Variable
  from extra.models.nllb import NLLBConfig, NLLBModel
  from examples.nllb import NLLBTokenizer, load_weights_from_zip
  from examples.nllb_fp16 import NLLBModelFP16, cast_to_fp16

  Tensor.training = False
  config = NLLBConfig()

  if fp16:
    model = NLLBModelFP16(config, max_cache_len=max_tokens + 20)
  else:
    model = NLLBModel(config, max_cache_len=max_tokens + 20)

  load_weights_from_zip(model, model_path)
  if fp16:
    cast_to_fp16(model)

  tokenizer = NLLBTokenizer(model_path)
  eos_id = tokenizer.eos_token_id

  results = []

  for src_lang, tgt_lang, text in SAMPLES:
    # Reset KV cache between samples (cross-attn cache has encoder-specific keys/values)
    model.reset_cache()

    input_ids, attention_mask = tokenizer.encode(text, src_lang)
    forced_bos = tokenizer.get_lang_id(tgt_lang)

    src_ids  = Tensor(np.array([input_ids],     dtype=np.int32))
    src_mask = Tensor(np.array([attention_mask], dtype=np.int32))

    t0 = time.perf_counter()
    enc_out = model.encode(src_ids, src_mask).realize()
    encode_ms = (time.perf_counter() - t0) * 1000

    generated = [eos_id, forced_bos]
    dec_in0 = Tensor(np.array([generated], dtype=np.int32))
    t0 = time.perf_counter()
    logits0 = model.decode(dec_in0, enc_out, src_mask, start_pos=0)
    next_token = int(logits0[0, -1].argmax(axis=-1).numpy())
    first_decode_ms = (time.perf_counter() - t0) * 1000
    generated.append(next_token)

    max_start = max_tokens + len(generated) + 5
    # New JIT per sample — enc_out and src_mask are captured in the closure
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
    jit_tokens  = len(output_ids) - 1
    tok_per_s   = jit_tokens / (jit_ms / 1000) if jit_ms > 0 else 0

    results.append({
      "src_lang":       src_lang,
      "tgt_lang":       tgt_lang,
      "source":         text,
      "translation":    translation,
      "src_tokens":     len(input_ids),
      "tgt_tokens":     jit_tokens,
      "encode_ms":      encode_ms,
      "first_decode_ms": first_decode_ms,
      "jit_ms":         jit_ms,
      "tok_per_s":      tok_per_s,
    })

    print(f"  [{src_lang}->{tgt_lang}] {len(input_ids)} src tok -> {jit_tokens} tgt tok  "
          f"{tok_per_s:.2f} tok/s", flush=True)

  return results


# ---------------------------------------------------------------------------
# Subprocess runner (same pattern as nllb_bench.py — fresh process per config)
# ---------------------------------------------------------------------------

def run_config_subprocess(config_name: str, model_path: str, max_tokens: int) -> list:
  cfg = CONFIGS[config_name]
  env = os.environ.copy()
  env.update(cfg.get("env", {}))
  env["_BENCH_CONFIG"]     = config_name
  env["_BENCH_MODEL_PATH"] = model_path
  env["_BENCH_MAX_TOKENS"] = str(max_tokens)
  repo_root = str(Path(__file__).resolve().parent.parent)
  pp = env.get("PYTHONPATH", "")
  env["PYTHONPATH"] = repo_root + (os.pathsep + pp if pp else "")

  import subprocess
  cmd = [sys.executable, __file__, "--_run_config"]
  result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
  if result.returncode != 0:
    print(f"  ERROR: {result.stderr[-1000:]}", file=sys.stderr)
    return []
  lines = [l for l in result.stdout.strip().splitlines() if l.startswith("[")]
  try:
    return json.loads(lines[-1])
  except Exception as e:
    print(f"  Parse error: {e}\nstdout: {result.stdout[-500:]}", file=sys.stderr)
    return []


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

LANG_NAMES = {
  "eng_Latn": "English",
  "fra_Latn": "French",
  "deu_Latn": "German",
  "spa_Latn": "Spanish",
  "zho_Hans": "Chinese",
  "arb_Arab": "Arabic",
  "jpn_Jpan": "Japanese",
}


def print_results(config_name: str, cfg: dict, results: list):
  print(f"\n{'='*80}")
  print(f"  {cfg['label']}")
  print(f"{'='*80}")
  print(f"  {'Pair':<15} {'Src tok':>7} {'Tgt tok':>7} {'tok/s':>7}  Translation (truncated to 80 chars)")
  print(f"  {'-'*75}")
  tps_all = []
  for r in results:
    pair = f"{LANG_NAMES.get(r['src_lang'], r['src_lang'])[:7]}->{LANG_NAMES.get(r['tgt_lang'], r['tgt_lang'])[:7]}"
    tps_all.append(r["tok_per_s"])
    tgt_preview = r["translation"][:78].encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace")
    print(f"  {pair:<15} {r['src_tokens']:>7} {r['tgt_tokens']:>7} {r['tok_per_s']:>7.2f}  {tgt_preview}")
  if tps_all:
    avg = sum(tps_all) / len(tps_all)
    print(f"\n  avg tok/s across all samples: {avg:.2f}")
    print(f"  min: {min(tps_all):.2f}  max: {max(tps_all):.2f}")


def print_comparison(all_results: dict):
  print(f"\n{'='*80}")
  print("COMPARISON SUMMARY")
  print(f"{'='*80}")

  configs = list(all_results.keys())
  if not configs:
    return

  # Per-sample comparison
  samples_ref = all_results[configs[0]]
  print(f"\n  {'Pair':<15} {'Src tok':>7}", end="")
  for c in configs:
    print(f"  {c:>12} tok/s", end="")
  print()
  print(f"  {'-'*70}")

  for i, ref in enumerate(samples_ref):
    pair = f"{LANG_NAMES.get(ref['src_lang'], ref['src_lang'])[:7]}->{LANG_NAMES.get(ref['tgt_lang'], ref['tgt_lang'])[:7]}"
    print(f"  {pair:<15} {ref['src_tokens']:>7}", end="")
    for c in configs:
      rs = all_results[c]
      tps = rs[i]["tok_per_s"] if i < len(rs) else float("nan")
      print(f"  {tps:>17.2f}", end="")
    print()

  # Overall averages
  print(f"\n  {'Average':<23}", end="")
  for c in configs:
    rs = all_results[c]
    avg = sum(r["tok_per_s"] for r in rs) / len(rs) if rs else 0
    print(f"  {avg:>17.2f}", end="")
  print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", default=MODEL_PATH_DEFAULT)
  parser.add_argument("--max_tokens", type=int, default=80,
                      help="Max decode tokens per sample (longer = more stable timing for long seqs)")
  parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                      choices=list(CONFIGS.keys()))
  parser.add_argument("--_run_config", action="store_true", help=argparse.SUPPRESS)
  args = parser.parse_args()

  # Child mode: run single config, emit JSON
  if args._run_config:
    config_name = os.environ["_BENCH_CONFIG"]
    model_path  = os.environ["_BENCH_MODEL_PATH"]
    max_tokens  = int(os.environ["_BENCH_MAX_TOKENS"])
    cfg = CONFIGS[config_name]
    results = run_all_samples(
      model_path, max_tokens,
      fp16=cfg["fp16"],
      use_beam_cache=cfg.get("use_beam_cache", False),
    )
    print(json.dumps(results))
    return

  # Parent mode
  print(f"NLLB Extended Benchmark")
  print(f"  model:      {args.model_path}")
  print(f"  samples:    {len(SAMPLES)}")
  print(f"  max_tokens: {args.max_tokens}")
  print(f"  configs:    {', '.join(args.configs)}")
  print()

  all_results = {}
  for name in args.configs:
    cfg = CONFIGS[name]
    print(f"\n[{name}] {cfg['label']}")
    t_start = time.perf_counter()
    results = run_config_subprocess(name, args.model_path, args.max_tokens)
    elapsed = time.perf_counter() - t_start
    all_results[name] = results
    print_results(name, cfg, results)
    print(f"\n  wall time: {elapsed:.1f}s")

  if len(all_results) > 1:
    print_comparison(all_results)


if __name__ == "__main__":
  main()
