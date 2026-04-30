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
