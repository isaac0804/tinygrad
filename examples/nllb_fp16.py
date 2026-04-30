#!/usr/bin/env python3
"""
NLLB-200 fp16 experiment.

Strategy for numerical stability:
  - All linear weights (q/k/v/out proj, fc1/fc2, embed_tokens) -> fp16
  - Layer norm weights/biases stay fp32  (layer norm is numerically sensitive)
  - Sinusoidal positional embeddings stay fp32
  - lm_head projection: cast hidden states back to fp32 before matmul
    (256k vocab softmax overflows in fp16)
  - KV cache allocated in fp16 (automatically, from k_new.dtype)

Usage:
  python examples/nllb_fp16.py --model_path "D:\\models\\nllb-600M-bc-bm-250817.zip" \\
      --src_lang eng_Latn --tgt_lang fra_Latn \\
      --text "The United Nations Chief says there is no military solution in Syria"
"""

import argparse, time
from typing import List, Optional, Union

import numpy as np
from tinygrad import Tensor, TinyJit, Variable, dtypes, Device
from tinygrad.nn.state import get_state_dict, load_state_dict, torch_load
from tinygrad.helpers import Timing
from tinygrad.uop.ops import UOp

from extra.models.nllb import NLLBConfig, NLLBModel, remap_weights
from examples.nllb import NLLBTokenizer, load_weights_from_zip, load_weights_from_hf


# ---------------------------------------------------------------------------
# Mixed-precision model wrapper
# ---------------------------------------------------------------------------

def cast_to_fp16(model: NLLBModel) -> NLLBModel:
  """
  Cast all weights to fp16 except:
  - LayerNorm weight/bias (keep fp32 for stability)
  - Sinusoidal positional embedding weights (keep fp32, small, not perf critical)
  """
  sd = get_state_dict(model)
  n_cast = 0
  n_skip = 0
  for name, t in sd.items():
    # Skip layer norm parameters
    if "layer_norm" in name or "layernorm" in name:
      n_skip += 1
      continue
    # Skip positional embedding weights (sinusoidal, not trained)
    if "embed_positions" in name:
      n_skip += 1
      continue
    # Cast everything else to fp16
    t.replace(t.cast(dtypes.half).realize())
    n_cast += 1
  print(f"  cast {n_cast} tensors to fp16, kept {n_skip} in fp32")
  return model


class NLLBModelFP16(NLLBModel):
  """
  NLLBModel with mixed-precision decode:
  - All forward pass runs in fp16 (weights are fp16)
  - Before the lm_head matmul, cast hidden states back to fp32
  - This prevents overflow in the 256k-vocab softmax
  """

  def decode(
    self,
    decoder_input_ids: Tensor,
    encoder_hidden_states: Tensor,
    encoder_attention_mask: Optional[Tensor] = None,
    start_pos: Union[int, UOp] = 0,
  ) -> Tensor:
    hidden = self.decoder(decoder_input_ids, encoder_hidden_states,
                          encoder_attention_mask, start_pos=start_pos)
    # Cast to fp32 before the large vocab projection for numerical stability
    hidden_f32 = hidden.cast(dtypes.float)
    weight_f32 = self.shared.weight.cast(dtypes.float)
    logits = hidden_f32 @ weight_f32.T
    return logits


# ---------------------------------------------------------------------------
# Greedy decode (same logic as examples/nllb.py, works with any NLLBModel)
# ---------------------------------------------------------------------------

def greedy_decode(
  model: NLLBModel,
  input_ids: List[int],
  attention_mask: List[int],
  forced_bos_token_id: int,
  eos_token_id: int = 2,
  max_new_tokens: int = 200,
) -> tuple[List[int], dict]:
  Tensor.training = False
  timings = {}

  src_ids  = Tensor(np.array([input_ids],      dtype=np.int32))
  src_mask = Tensor(np.array([attention_mask],  dtype=np.int32))

  t0 = time.perf_counter()
  enc_out = model.encode(src_ids, src_mask).realize()
  timings["encode_ms"] = (time.perf_counter() - t0) * 1000

  generated = [eos_token_id, forced_bos_token_id]
  dec_in0   = Tensor(np.array([generated], dtype=np.int32))

  t0 = time.perf_counter()
  logits0    = model.decode(dec_in0, enc_out, src_mask, start_pos=0)
  next_token = int(logits0[0, -1].argmax(axis=-1).numpy())
  timings["first_decode_ms"] = (time.perf_counter() - t0) * 1000
  generated.append(next_token)

  if next_token == eos_token_id or max_new_tokens <= 1:
    return generated[2:], timings

  max_start  = max_new_tokens + len(generated) - 1
  decode_jit = TinyJit(lambda tok, sp: model.decode(tok, enc_out, src_mask, start_pos=sp))

  t0 = time.perf_counter()
  for step in range(1, max_new_tokens):
    sp      = Variable("start_pos", 1, max_start).bind(len(generated) - 1)
    dec_in  = Tensor(np.array([[generated[-1]]], dtype=np.int32))
    logits  = decode_jit(dec_in, sp)
    next_token = int(logits[0, -1].argmax(axis=-1).numpy())
    generated.append(next_token)
    if next_token == eos_token_id:
      break
  timings["jit_decode_ms"] = (time.perf_counter() - t0) * 1000
  timings["jit_tokens"]    = len(generated) - 3  # exclude prompt + first step

  return generated[2:], timings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(description="NLLB-200 fp16 mixed-precision experiment")
  parser.add_argument("--text",       type=str, default="The United Nations Chief says there is no military solution in Syria")
  parser.add_argument("--src_lang",   type=str, default="eng_Latn")
  parser.add_argument("--tgt_lang",   type=str, default="fra_Latn")
  parser.add_argument("--max_tokens", type=int, default=200)
  parser.add_argument("--model_path", type=str, default=None)
  parser.add_argument("--model_id",   type=str, default="facebook/nllb-200-distilled-600M")
  args = parser.parse_args()

  print(f"Device: {Device.DEFAULT}")
  print(f"Source: [{args.src_lang}] {args.text}")
  print(f"Target: {args.tgt_lang}")
  print()

  # Tokenizer
  source = args.model_path if args.model_path else args.model_id
  tokenizer = NLLBTokenizer(source)

  # Build model as fp16 subclass
  config = NLLBConfig()
  model  = NLLBModelFP16(config, max_cache_len=args.max_tokens + 10)

  # Load weights (fp32 from file)
  print("Loading weights ...")
  if args.model_path:
    load_weights_from_zip(model, args.model_path)
  else:
    load_weights_from_hf(model, args.model_id)

  # Cast to mixed precision
  print("Casting to mixed fp16/fp32 ...")
  cast_to_fp16(model)
  print()

  # Tokenize
  input_ids, attention_mask = tokenizer.encode(args.text, args.src_lang)
  forced_bos = tokenizer.get_lang_id(args.tgt_lang)
  print(f"Input tokens ({len(input_ids)}): {input_ids}")
  print(f"Forced BOS:  {forced_bos} ({args.tgt_lang})")
  print()

  # Decode
  t_total = time.perf_counter()
  output_ids, timings = greedy_decode(
    model, input_ids, attention_mask,
    forced_bos_token_id=forced_bos,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=args.max_tokens,
  )
  total_ms = (time.perf_counter() - t_total) * 1000

  translation = tokenizer.decode(output_ids)
  print(f"Translation: {translation}")
  print()

  # Timing report
  jit_tokens = timings.get("jit_tokens", 0)
  jit_ms     = timings.get("jit_decode_ms", 0)
  tok_per_s  = jit_tokens / (jit_ms / 1000) if jit_ms > 0 else 0

  print("--- Timing ---")
  print(f"  Encode:        {timings['encode_ms']:.1f} ms")
  print(f"  First decode:  {timings['first_decode_ms']:.1f} ms")
  print(f"  JIT decode:    {jit_ms:.1f} ms  ({jit_tokens} tokens)")
  print(f"  Total:         {total_ms:.1f} ms")
  print(f"  Throughput:    {tok_per_s:.2f} tok/s")


if __name__ == "__main__":
  main()
