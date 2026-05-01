#!/usr/bin/env python3
"""
NLLB-200 (No Language Left Behind) inference with tinygrad.

Supports translation between 200+ languages using the distilled 600M model.

Usage:
  # Load from a local zip file
  python examples/nllb.py --model_path /path/to/model.zip \
      --src_lang eng_Latn --tgt_lang fra_Latn \
      --text "UN Chief says there is no military solution in Syria"

  # Load from HuggingFace (downloads on first run)
  python examples/nllb.py --src_lang eng_Latn --tgt_lang fra_Latn \
      --text "UN Chief says there is no military solution in Syria"

  # Translate with a different HF model
  python examples/nllb.py --model_id facebook/nllb-200-distilled-1.3B \
      --src_lang eng_Latn --tgt_lang zho_Hans \
      --text "The quick brown fox jumps over the lazy dog"

Requirements:
  pip install sentencepiece transformers   # for the tokenizer only
"""

import argparse, os, zipfile, io, tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from tinygrad import Tensor, TinyJit, Variable, dtypes, Device
from tinygrad.nn.state import load_state_dict, torch_load, get_state_dict
from tinygrad.helpers import fetch, Timing, getenv

from extra.models.nllb import NLLBConfig, NLLBModel, remap_weights

# ---------------------------------------------------------------------------
# Tokenizer wrapper
# We reuse HuggingFace's NllbTokenizer (sentencepiece-based) just for
# tokenization — no model code from HF is used at inference time.
# ---------------------------------------------------------------------------

class NLLBTokenizer:
  def __init__(self, source: str):
    """
    source: either a HuggingFace model ID or a path to a local directory / zip file
            containing sentencepiece.bpe.model and tokenizer_config.json.
    """
    try:
      from transformers import NllbTokenizer as HFTokenizer
    except ImportError:
      raise ImportError("pip install transformers sentencepiece  # needed for the tokenizer")

    if source.endswith(".zip") or zipfile.is_zipfile(source):
      # Load tokenizer files from inside the zip
      self.hf = self._load_from_zip(source, HFTokenizer)
    else:
      self.hf = HFTokenizer.from_pretrained(source)

    self.pad_token_id: int = self.hf.pad_token_id       # 1
    self.eos_token_id: int = self.hf.eos_token_id       # 2
    self.bos_token_id: int = self.hf.bos_token_id       # 0

  @staticmethod
  def _load_from_zip(zip_path: str, HFTokenizer):
    """Extract tokenizer files from zip to a temp dir and load from there."""
    tokenizer_files = [
      "sentencepiece.bpe.model",
      "tokenizer.json",
      "tokenizer_config.json",
      "special_tokens_map.json",
    ]
    tmp_dir = Path(tempfile.mkdtemp(prefix="nllb_tok_"))
    with zipfile.ZipFile(zip_path, "r") as z:
      for name in z.namelist():
        basename = Path(name).name
        if basename in tokenizer_files:
          data = z.read(name)
          (tmp_dir / basename).write_bytes(data)
    return HFTokenizer.from_pretrained(str(tmp_dir))

  def encode(self, text: str, src_lang: str) -> tuple:
    """Tokenize text for the encoder. Returns (input_ids list, attention_mask list)."""
    self.hf.src_lang = src_lang
    enc = self.hf(text, return_tensors=None)
    return enc["input_ids"], enc["attention_mask"]

  def get_lang_id(self, lang_code: str) -> int:
    """Return the vocabulary index of a language token (e.g. 'fra_Latn')."""
    return self.hf.convert_tokens_to_ids(lang_code)

  def decode(self, token_ids: List[int]) -> str:
    return self.hf.decode(token_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------

def load_weights_from_zip(model: NLLBModel, zip_path: str):
  """Load pytorch_model.bin from inside a zip file."""
  print(f"Loading weights from zip: {zip_path}")
  with zipfile.ZipFile(zip_path, "r") as z:
    # Find the pytorch_model.bin entry (may be in a subdirectory)
    bin_entries = [n for n in z.namelist() if n.endswith("pytorch_model.bin")]
    if not bin_entries:
      raise FileNotFoundError(f"No pytorch_model.bin found inside {zip_path}")
    entry = bin_entries[0]
    print(f"  reading: {entry} ({z.getinfo(entry).file_size / 1e6:.1f} MB)")
    # Extract to D:\models\extracted (or a sibling dir of the zip) to avoid /tmp space issues
    zip_dir = Path(zip_path).parent
    extract_root = zip_dir / "extracted"
    extract_root.mkdir(exist_ok=True)
    extracted_path = extract_root / entry
    if not extracted_path.exists():
      z.extract(entry, str(extract_root))

  _load_from_bin(model, str(extracted_path))


def load_weights_from_hf(model: NLLBModel, model_id: str = "facebook/nllb-200-distilled-600M"):
  """Download and load weights from HuggingFace."""
  cache_dir = Path(os.path.expanduser("~/.cache/tinygrad/nllb"))
  cache_dir.mkdir(parents=True, exist_ok=True)
  pt_url = f"https://huggingface.co/{model_id}/resolve/main/pytorch_model.bin"
  pt_cache = cache_dir / (model_id.replace("/", "_") + ".bin")
  if not pt_cache.exists():
    print(f"Downloading weights for {model_id} ...")
    fetch(pt_url, pt_cache)
  _load_from_bin(model, str(pt_cache))


def _load_from_bin(model: NLLBModel, path: str):
  print(f"  loading weights from {path} ...")
  with Timing("  load time: "):
    raw = torch_load(path)
  remapped = remap_weights(raw)
  load_state_dict(model, remapped, strict=False)
  print(f"  loaded {len(remapped)} weight tensors")


# ---------------------------------------------------------------------------
# Weight download helper (kept for backward compat)
# ---------------------------------------------------------------------------

def load_model_weights(model: NLLBModel, model_id: str = "facebook/nllb-200-distilled-600M"):
  """
  Download and load weights from HuggingFace.
  Uses the pytorch_model.bin (or model.safetensors) via tinygrad's torch_load.
  """
  from pathlib import Path

  # Try to find a cached safetensors file first, then fall back to pytorch bin
  cache_dir = Path(os.path.expanduser("~/.cache/tinygrad/nllb"))
  cache_dir.mkdir(parents=True, exist_ok=True)

  # Prefer safetensors
  safetensors_url = f"https://huggingface.co/{model_id}/resolve/main/model.safetensors"
  pt_url = f"https://huggingface.co/{model_id}/resolve/main/pytorch_model.bin"

  cache_file = cache_dir / (model_id.replace("/", "_") + ".safetensors")
  pt_cache_file = cache_dir / (model_id.replace("/", "_") + ".bin")

  if not cache_file.exists() and not pt_cache_file.exists():
    print(f"Downloading weights for {model_id} ...")
    try:
      data = fetch(safetensors_url, cache_file)
      cache_file = data
    except Exception:
      data = fetch(pt_url, pt_cache_file)
      pt_cache_file = data

  weight_file = cache_file if cache_file.exists() else pt_cache_file

  print(f"Loading weights from {weight_file} ...")
  with Timing("  load time: "):
    if str(weight_file).endswith(".safetensors"):
      from tinygrad.nn.state import safe_load
      raw = safe_load(str(weight_file))
    else:
      raw = torch_load(str(weight_file))

  remapped = remap_weights(raw)
  load_state_dict(model, remapped, strict=False)
  print(f"  loaded {len(remapped)} weight tensors")


# ---------------------------------------------------------------------------
# Greedy / beam search decode
# ---------------------------------------------------------------------------

def beam_search_decode(
  model: "NLLBModel",
  input_ids: List[int],
  attention_mask: List[int],
  forced_bos_token_id: int,
  eos_token_id: int = 2,
  max_new_tokens: int = 200,
  beam_width: int = 4,
  length_penalty: float = 1.0,
  verbose: bool = False,
) -> List[int]:
  """
  Sequence-level beam search decoding for NLLB.

  Runs `beam_width` hypothesis beams in parallel by expanding the batch dimension
  to `beam_width`.  The model's KV-cache is initialised with bsz=beam_width on the
  first decode call so all subsequent single-token steps are fully batched.

  Args:
    model:                 NLLBModel (or NLLBModelFP16) instance.
    input_ids:             Encoder token ids (list of ints, length src_len).
    attention_mask:        Encoder attention mask (list of ints, length src_len).
    forced_bos_token_id:   Target-language BOS token id.
    eos_token_id:          EOS token id (default 2).
    max_new_tokens:        Maximum decoder steps.
    beam_width:            Number of parallel beams (default 4).
    length_penalty:        Exponent applied to sequence length when scoring
                           completed beams — values > 1 favour longer outputs.
    verbose:               Print per-step diagnostics.

  Returns:
    List of output token ids (excluding the initial [eos, lang_id] prompt).
  """
  import math
  Tensor.training = False

  # ── Encode (bsz=1) ──────────────────────────────────────────────────────────
  src_ids  = Tensor(np.array([input_ids],      dtype=np.int32))   # (1, src_len)
  src_mask = Tensor(np.array([attention_mask], dtype=np.int32))   # (1, src_len)

  with Timing("Encode: "):
    enc_out = model.encode(src_ids, src_mask).realize()            # (1, src_len, d)

  # Expand encoder outputs to beam_width copies: (B, src_len, d) and (B, src_len)
  enc_out_b  = enc_out.repeat([beam_width, 1, 1]).realize()       # (B, src_len, d)
  src_mask_b = src_mask.repeat([beam_width, 1]).realize()         # (B, src_len)

  # ── First decode step ───────────────────────────────────────────────────────
  # Feed [eos, forced_bos] for all B beams simultaneously so the cross-attn
  # cache (shape B×H×src_len×head_dim) is built correctly.
  init_tokens = [eos_token_id, forced_bos_token_id]
  dec_in0 = Tensor(
    np.tile(np.array([init_tokens], dtype=np.int32), (beam_width, 1))
  )  # (B, 2)

  with Timing("First decode step: "):
    logits0 = model.decode(dec_in0, enc_out_b, src_mask_b, start_pos=0)
    # logits0: (B, 2, vocab) — take last position's logits for beam 0 (all same)
    log_probs0 = logits0[0, -1].log_softmax(axis=-1).numpy()     # (vocab,)

  # Initialise beams from the top-k tokens of beam 0
  vocab_size = log_probs0.shape[-1]
  top_k_ids  = np.argsort(log_probs0)[::-1][:beam_width]

  # Each beam: (score, token_sequence, finished)
  # token_sequence includes the init_tokens prefix (stripped at the end)
  beams: list = []
  for i, tok in enumerate(top_k_ids):
    beams.append({
      "score":    float(log_probs0[tok]),
      "tokens":   list(init_tokens) + [int(tok)],
      "finished": int(tok) == eos_token_id,
    })

  if verbose:
    print(f"  beam_width={beam_width}  initial top tokens: {top_k_ids.tolist()}")

  # ── JIT-compiled single-token decode ────────────────────────────────────────
  max_start  = max_new_tokens + len(init_tokens) + 5
  decode_jit = TinyJit(
    lambda tok, sp: model.decode(tok, enc_out_b, src_mask_b, start_pos=sp)
  )

  completed: list = []

  with Timing("Beam decode (JIT): "):
    for step in range(1, max_new_tokens):
      # Select active beams
      active = [b for b in beams if not b["finished"]]
      if not active:
        break

      # Build decoder input: last token of each active beam, shape (B, 1)
      # Pad inactive beams with pad token so tensor is always (beam_width, 1)
      last_tokens = np.array(
        [[b["tokens"][-1]] for b in beams], dtype=np.int32
      )  # (B, 1)
      dec_in = Tensor(last_tokens)

      sp     = Variable("start_pos", 1, max_start).bind(len(beams[0]["tokens"]) - 1)
      logits = decode_jit(dec_in, sp)                             # (B, 1, vocab)
      lp     = logits[:, -1, :].log_softmax(axis=-1).numpy()     # (B, vocab)

      # Expand each active beam by top-k
      candidates = []
      for b_idx, beam in enumerate(beams):
        if beam["finished"]:
          # Keep finished beams as-is (score doesn't change)
          candidates.append((beam["score"], b_idx, eos_token_id, True))
          continue
        top_k = np.argsort(lp[b_idx])[::-1][:beam_width]
        for tok in top_k:
          new_score = beam["score"] + float(lp[b_idx][tok])
          candidates.append((new_score, b_idx, int(tok), int(tok) == eos_token_id))

      # Length-normalise and pick top beam_width
      def norm_score(score: float, length: int) -> float:
        return score / (length ** length_penalty) if length > 0 else score

      cur_len = len(beams[0]["tokens"])
      candidates.sort(key=lambda c: norm_score(c[0], cur_len + (0 if c[3] else 1)), reverse=True)

      new_beams = []
      for score, b_idx, tok, finished in candidates[:beam_width]:
        new_beam = {
          "score":    score,
          "tokens":   beams[b_idx]["tokens"] + [tok],
          "finished": finished,
        }
        new_beams.append(new_beam)

      beams = new_beams

      if verbose:
        best = max(beams, key=lambda b: norm_score(b["score"], len(b["tokens"])))
        print(f"  step {step+1}: best score={best['score']:.3f}  "
              f"tok={best['tokens'][-1]}", end="\r")

      # Move finished beams to completed pool
      still_running = []
      for b in beams:
        if b["finished"]:
          completed.append(b)
        else:
          still_running.append(b)

      # If we have enough completed beams, stop early
      if len(completed) >= beam_width:
        break

      # Pad beams list back to beam_width with copies of the best active beam
      # so the batch tensor shape stays constant
      if still_running:
        best_active = max(still_running, key=lambda b: b["score"])
        while len(still_running) < beam_width:
          still_running.append(dict(best_active))
        beams = still_running
      else:
        break

  if verbose:
    print()

  # ── Select best completed (or fallback to best running) beam ────────────────
  pool = completed if completed else beams

  def final_score(b: dict) -> float:
    out_len = max(len(b["tokens"]) - len(init_tokens), 1)
    return b["score"] / (out_len ** length_penalty)

  best_beam = max(pool, key=final_score)

  # Strip init_tokens prefix and trailing EOS
  output = best_beam["tokens"][len(init_tokens):]
  if output and output[-1] == eos_token_id:
    output = output[:-1]

  return output


def greedy_decode(
  model: NLLBModel,
  input_ids: List[int],
  attention_mask: List[int],
  forced_bos_token_id: int,
  eos_token_id: int = 2,
  max_new_tokens: int = 200,
  verbose: bool = False,
) -> List[int]:
  """
  Autoregressive greedy decoding with KV-cache and TinyJit.

  Steps:
  1. Encode the source sequence once.
  2. Run the first decode step with [eos, forced_bos] as the initial decoder input
     (M2M100 decoder starts with eos as the first token).
  3. JIT-compile the single-token decode step; reuse compiled kernels for all
     subsequent steps via Variable start_pos.
  """
  Tensor.training = False

  # --- Encode ---
  src_ids = Tensor(np.array([input_ids], dtype=np.int32))
  src_mask = Tensor(np.array([attention_mask], dtype=np.int32))

  with Timing("Encode: "):
    enc_out = model.encode(src_ids, src_mask).realize()

  if verbose:
    print(f"  encoder output shape: {enc_out.shape}")

  # --- First decode step (not JIT'd — establishes cross-attn KV cache) ---
  # Feed both initial tokens [eos, forced_bos] at once.
  generated = [eos_token_id, forced_bos_token_id]
  dec_in0 = Tensor(np.array([generated], dtype=np.int32))

  if verbose:
    print(f"  initial decoder tokens: {generated}")

  with Timing("First decode step (warm-up + cross-attn cache): "):
    logits0 = model.decode(dec_in0, enc_out, src_mask, start_pos=0)
    next_token = int(logits0[0, -1].argmax(axis=-1).numpy())
  generated.append(next_token)

  if verbose:
    print(f"  step 1: token {next_token}")

  if next_token == eos_token_id or max_new_tokens <= 1:
    return generated[2:]

  # --- JIT-compiled single-token decode steps ---
  # start_pos ranges from 2 (after the two initial tokens) up to max_new_tokens+2
  max_start = max_new_tokens + len(generated) - 1  # conservative upper bound
  decode_jit = TinyJit(
    lambda tok, sp: model.decode(tok, enc_out, src_mask, start_pos=sp)
  )

  with Timing("Decode (JIT): "):
    for step in range(1, max_new_tokens):
      sp = Variable("start_pos", 1, max_start).bind(len(generated) - 1)
      dec_in = Tensor(np.array([[generated[-1]]], dtype=np.int32))
      logits = decode_jit(dec_in, sp)
      next_token = int(logits[0, -1].argmax(axis=-1).numpy())
      generated.append(next_token)

      if verbose:
        print(f"  step {step+1}: token {next_token}", end="\r")

      if next_token == eos_token_id:
        break

  if verbose:
    print()

  # Strip the initial prompt tokens (eos + lang_id) from the output
  return generated[2:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(description="NLLB-200 translation with tinygrad")
  parser.add_argument("--text", type=str, default="UN Chief says there is no military solution in Syria",
                      help="Source text to translate")
  parser.add_argument("--src_lang", type=str, default="eng_Latn",
                      help="Source language BCP-47 code (e.g. eng_Latn, fra_Latn, zho_Hans)")
  parser.add_argument("--tgt_lang", type=str, default="fra_Latn",
                      help="Target language BCP-47 code")
  parser.add_argument("--max_tokens", type=int, default=200,
                      help="Maximum number of tokens to generate")
  parser.add_argument("--beam_width", type=int, default=1,
                      help="Beam width for sequence-level beam search (1 = greedy)")
  parser.add_argument("--length_penalty", type=float, default=1.0,
                      help="Length penalty exponent for beam search (>1 favours longer outputs)")
  parser.add_argument("--model_path", type=str, default=None,
                      help="Path to a local .zip file containing the model weights and tokenizer")
  parser.add_argument("--model_id", type=str, default="facebook/nllb-200-distilled-600M",
                      help="HuggingFace model ID (used when --model_path is not set)")
  parser.add_argument("--verbose", action="store_true", help="Print step-by-step info")
  args = parser.parse_args()

  print(f"Device:    {Device.DEFAULT}")
  print(f"Source:    [{args.src_lang}] {args.text}")
  print(f"Target:    {args.tgt_lang}")
  print()

  # Build tokenizer — from local zip or HuggingFace
  if args.model_path:
    print(f"Model:     {args.model_path} (local zip)")
    tokenizer = NLLBTokenizer(args.model_path)
  else:
    print(f"Model:     {args.model_id}")
    tokenizer = NLLBTokenizer(args.model_id)

  # Build model
  config = NLLBConfig()  # defaults match nllb-200-distilled-600M
  model = NLLBModel(config, max_cache_len=args.max_tokens + 10)

  # Load weights
  if args.model_path:
    load_weights_from_zip(model, args.model_path)
  else:
    load_weights_from_hf(model, args.model_id)

  # Tokenize
  input_ids, attention_mask = tokenizer.encode(args.text, args.src_lang)
  forced_bos = tokenizer.get_lang_id(args.tgt_lang)

  print(f"Input tokens ({len(input_ids)}): {input_ids}")
  print(f"Forced BOS token id for '{args.tgt_lang}': {forced_bos}")
  print()

  # Decode
  with Timing("Total inference: "):
    if args.beam_width > 1:
      output_ids = beam_search_decode(
        model, input_ids, attention_mask,
        forced_bos_token_id=forced_bos,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_tokens,
        beam_width=args.beam_width,
        length_penalty=args.length_penalty,
        verbose=args.verbose,
      )
    else:
      output_ids = greedy_decode(
        model, input_ids, attention_mask,
        forced_bos_token_id=forced_bos,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_tokens,
        verbose=args.verbose,
      )

  translation = tokenizer.decode(output_ids)
  print(f"\nTranslation: {translation}")


if __name__ == "__main__":
  main()
