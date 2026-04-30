# NLLB-200 (No Language Left Behind) — tinygrad implementation
# Architecture: M2M100 encoder-decoder transformer
# Reference: https://huggingface.co/facebook/nllb-200-distilled-600M
#
# Weight keys match HuggingFace M2M100ForConditionalGeneration:
#   model.shared.weight                         (vocab_size, d_model)
#   model.encoder.embed_positions.weights        (max_pos+2, d_model)  -- sinusoidal, non-trainable
#   model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}
#   model.encoder.layers.{i}.self_attn_layer_norm.{weight,bias}
#   model.encoder.layers.{i}.fc1/fc2.{weight,bias}
#   model.encoder.layers.{i}.final_layer_norm.{weight,bias}
#   model.encoder.layer_norm.{weight,bias}
#   (same pattern for decoder, plus encoder_attn layers)
#   lm_head.weight  (tied to model.shared.weight)

import math
from dataclasses import dataclass
from typing import Optional, List, Union

from tinygrad import nn, Tensor, dtypes
from tinygrad.uop.ops import UOp
from tinygrad.nn.state import load_state_dict
from tinygrad.helpers import fetch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NLLBConfig:
  vocab_size: int = 256206
  d_model: int = 1024
  encoder_layers: int = 12
  decoder_layers: int = 12
  encoder_attention_heads: int = 16
  decoder_attention_heads: int = 16
  encoder_ffn_dim: int = 4096
  decoder_ffn_dim: int = 4096
  max_position_embeddings: int = 1024
  dropout: float = 0.1
  attention_dropout: float = 0.1
  activation_dropout: float = 0.0
  pad_token_id: int = 1
  bos_token_id: int = 0
  eos_token_id: int = 2
  decoder_start_token_id: int = 2
  scale_embedding: bool = True


# ---------------------------------------------------------------------------
# Sinusoidal Positional Embedding
# (offset=2, positions include padding_idx and a spare slot)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEmbedding:
  """Fixed sinusoidal positional embeddings used by M2M100/NLLB."""

  OFFSET = 2  # positions 0,1 reserved; real positions start at padding_idx+1

  def __init__(self, num_positions: int, embedding_dim: int, padding_idx: int = 1):
    self.embedding_dim = embedding_dim
    self.padding_idx = padding_idx
    # weights shape: (num_positions + OFFSET, embedding_dim)
    self.weights = SinusoidalPositionalEmbedding._get_embedding(
      num_positions + self.OFFSET, embedding_dim, padding_idx
    )

  @staticmethod
  def _get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: int) -> Tensor:
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    # (half_dim,)
    inv_freq = Tensor.arange(half_dim).float() * -emb
    inv_freq = inv_freq.exp()
    # (num_embeddings, half_dim)
    pos = Tensor.arange(num_embeddings).float().unsqueeze(1) * inv_freq.unsqueeze(0)
    # interleave sin/cos: (num_embeddings, embedding_dim)
    emb = Tensor.cat(pos.sin(), pos.cos(), dim=1)
    if embedding_dim % 2 == 1:
      emb = Tensor.cat(emb, Tensor.zeros(num_embeddings, 1), dim=1)
    # zero out padding position
    # We use a simple mask since tinygrad doesn't have index_put
    mask = Tensor.ones(num_embeddings, 1)
    # set row padding_idx to zero via multiplication
    row_mask = (Tensor.arange(num_embeddings) != padding_idx).float().unsqueeze(1)
    emb = emb * row_mask
    return emb

  def __call__(self, input_ids: Tensor, past_key_values_length: int = 0) -> Tensor:
    """
    input_ids: (batch, seq_len)
    Returns: (batch, seq_len, embedding_dim)
    """
    bsz, seq_len = input_ids.shape
    # Create position ids: non-padding tokens get incrementing positions
    # mask: 1 where not padding, 0 where padding
    mask = (input_ids != self.padding_idx).cast(dtypes.int32)
    # cumsum gives 1-based incremental positions for non-pad tokens
    incremental = mask.cumsum(axis=1) + past_key_values_length
    position_ids = incremental * mask + self.padding_idx  # pad positions stay at padding_idx
    # gather embeddings: (bsz*seq_len,) -> (bsz, seq_len, d_model)
    flat_ids = position_ids.flatten()  # (bsz*seq_len,)
    # index into weights table
    embeds = self.weights[flat_ids]    # (bsz*seq_len, d_model)
    return embeds.reshape(bsz, seq_len, self.embedding_dim)


# ---------------------------------------------------------------------------
# Multi-Head Attention (self-attention and cross-attention)
# Supports KV caching for the decoder.
# ---------------------------------------------------------------------------

class M2M100Attention:
  """
  Multi-head attention for M2M100/NLLB.
  - is_decoder=True: enable KV caching for self-attention
  - cross_attention=True: this is an encoder-decoder cross-attention layer
    (encoder outputs are the key/value source; KV is cached after first step)

  KV cache uses the assign-based pattern (same as gpt2.py) so it is compatible
  with TinyJit + symbolic Variable start_pos.
  """

  def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
               is_decoder: bool = False, cross_attention: bool = False,
               max_cache_len: int = 0):
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.scaling = self.head_dim ** -0.5
    self.dropout = dropout
    self.is_decoder = is_decoder
    self.cross_attention = cross_attention
    self.max_cache_len = max_cache_len

    self.q_proj = nn.Linear(embed_dim, embed_dim)
    self.k_proj = nn.Linear(embed_dim, embed_dim)
    self.v_proj = nn.Linear(embed_dim, embed_dim)
    self.out_proj = nn.Linear(embed_dim, embed_dim)

  def _shape(self, x: Tensor) -> Tensor:
    """(batch, seq, embed) -> (batch, heads, seq, head_dim)"""
    bsz, seq = x.shape[0], x.shape[1]
    return x.reshape(bsz, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

  def __call__(
    self,
    hidden_states: Tensor,
    key_value_states: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    start_pos: Union[int, UOp] = 0,
  ) -> Tensor:
    """
    hidden_states:     (batch, tgt_seq, embed_dim)
    key_value_states:  (batch, src_seq, embed_dim) — encoder output for cross-attn
    attention_mask:    additive float mask broadcastable to (B, H, tgt, src)
    start_pos:         int or Variable — decoder step position for self-attn KV cache writes
    """
    bsz, tgt_len, _ = hidden_states.shape

    q = self._shape(self.q_proj(hidden_states))  # (B, H, tgt, head_dim)

    if self.cross_attention:
      # Cross-attention: keys/values come from encoder output.
      # key_value_states is provided on the first decoder step (start_pos==0),
      # then we reuse the cached KV for subsequent steps.
      if key_value_states is not None:
        k = self._shape(self.k_proj(key_value_states))
        v = self._shape(self.v_proj(key_value_states))
        if not hasattr(self, 'cache_k'):
          self.cache_k = k.contiguous().realize()
          self.cache_v = v.contiguous().realize()
        else:
          self.cache_k.assign(k.contiguous()).realize()
          self.cache_v.assign(v.contiguous()).realize()
      else:
        k, v = self.cache_k, self.cache_v
    else:
      # Self-attention with assign-based KV cache
      k_new = self._shape(self.k_proj(hidden_states))  # (B, H, tgt_len, head_dim)
      v_new = self._shape(self.v_proj(hidden_states))

      if self.is_decoder and self.max_cache_len > 0:
        if not hasattr(self, 'cache_k'):
          self.cache_k = Tensor.zeros(bsz, self.num_heads, self.max_cache_len, self.head_dim, dtype=k_new.dtype).contiguous().realize()
          self.cache_v = Tensor.zeros(bsz, self.num_heads, self.max_cache_len, self.head_dim, dtype=v_new.dtype).contiguous().realize()
        # Write new tokens into the cache at start_pos
        self.cache_k[:, :, start_pos:start_pos+tgt_len, :].assign(k_new).realize()
        self.cache_v[:, :, start_pos:start_pos+tgt_len, :].assign(v_new).realize()
        # Read back all tokens up to start_pos+tgt_len (symbolic shrink — JIT friendly)
        k = self.cache_k[:, :, :start_pos+tgt_len, :]
        v = self.cache_v[:, :, :start_pos+tgt_len, :]
      else:
        k, v = k_new, v_new

    # scaled dot-product attention
    attn_out = Tensor.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                                    dropout_p=self.dropout if Tensor.training else 0.0)
    # (B, H, tgt, head_dim) -> (B, tgt, embed_dim)
    attn_out = attn_out.permute(0, 2, 1, 3).reshape(bsz, tgt_len, self.embed_dim)
    return self.out_proj(attn_out)


# ---------------------------------------------------------------------------
# Encoder Layer
# ---------------------------------------------------------------------------

class M2M100EncoderLayer:
  def __init__(self, config: NLLBConfig):
    d = config.d_model
    self.self_attn = M2M100Attention(d, config.encoder_attention_heads,
                                     dropout=config.attention_dropout)
    self.self_attn_layer_norm = nn.LayerNorm(d)
    self.fc1 = nn.Linear(d, config.encoder_ffn_dim)
    self.fc2 = nn.Linear(config.encoder_ffn_dim, d)
    self.final_layer_norm = nn.LayerNorm(d)
    self.dropout = config.dropout
    self.activation_dropout = config.activation_dropout

  def __call__(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
    # Pre-norm self-attention
    residual = x
    x = self.self_attn_layer_norm(x)
    x = self.self_attn(x, attention_mask=attention_mask)
    x = x.dropout(self.dropout)
    x = residual + x
    # Pre-norm FFN
    residual = x
    x = self.final_layer_norm(x)
    x = self.fc1(x).relu()
    x = x.dropout(self.activation_dropout)
    x = self.fc2(x)
    x = x.dropout(self.dropout)
    x = residual + x
    return x


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class M2M100DecoderLayer:
  def __init__(self, config: NLLBConfig, max_cache_len: int = 0):
    d = config.d_model
    self.self_attn = M2M100Attention(d, config.decoder_attention_heads,
                                     dropout=config.attention_dropout,
                                     is_decoder=True, max_cache_len=max_cache_len)
    self.self_attn_layer_norm = nn.LayerNorm(d)
    self.encoder_attn = M2M100Attention(d, config.decoder_attention_heads,
                                        dropout=config.attention_dropout,
                                        is_decoder=True, cross_attention=True)
    self.encoder_attn_layer_norm = nn.LayerNorm(d)
    self.fc1 = nn.Linear(d, config.decoder_ffn_dim)
    self.fc2 = nn.Linear(config.decoder_ffn_dim, d)
    self.final_layer_norm = nn.LayerNorm(d)
    self.dropout = config.dropout
    self.activation_dropout = config.activation_dropout

  def __call__(
    self,
    x: Tensor,
    encoder_hidden_states: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    encoder_attention_mask: Optional[Tensor] = None,
    start_pos: Union[int, UOp] = 0,
  ) -> Tensor:
    # Self-attention (causal)
    residual = x
    x = self.self_attn_layer_norm(x)
    x = self.self_attn(x, attention_mask=attention_mask, start_pos=start_pos)
    x = x.dropout(self.dropout)
    x = residual + x
    # Cross-attention
    if encoder_hidden_states is not None or hasattr(self.encoder_attn, 'cache_k'):
      residual = x
      x = self.encoder_attn_layer_norm(x)
      x = self.encoder_attn(x, key_value_states=encoder_hidden_states,
                             attention_mask=encoder_attention_mask, start_pos=start_pos)
      x = x.dropout(self.dropout)
      x = residual + x
    # FFN
    residual = x
    x = self.final_layer_norm(x)
    x = self.fc1(x).relu()
    x = x.dropout(self.activation_dropout)
    x = self.fc2(x)
    x = x.dropout(self.dropout)
    x = residual + x
    return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class M2M100Encoder:
  def __init__(self, config: NLLBConfig, embed_tokens: nn.Embedding):
    self.embed_tokens = embed_tokens
    self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
    self.embed_positions = SinusoidalPositionalEmbedding(
      config.max_position_embeddings, config.d_model, config.pad_token_id
    )
    self.layers = [M2M100EncoderLayer(config) for _ in range(config.encoder_layers)]
    self.layer_norm = nn.LayerNorm(config.d_model)
    self.dropout = config.dropout

  def __call__(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
    """
    input_ids:      (batch, src_len)
    attention_mask: (batch, src_len) binary mask; 1=attend, 0=ignore
    Returns:        (batch, src_len, d_model)
    """
    x = self.embed_tokens(input_ids) * self.embed_scale
    x = x + self.embed_positions(input_ids)
    x = x.dropout(self.dropout)

    # Convert binary attention_mask to additive float mask
    if attention_mask is not None:
      # (batch, 1, 1, src_len) — broadcast over heads and query positions
      enc_mask = (1.0 - attention_mask.cast(dtypes.float32)).unsqueeze(1).unsqueeze(2) * -1e9
    else:
      enc_mask = None

    for layer in self.layers:
      x = layer(x, enc_mask)
    x = self.layer_norm(x)
    return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class M2M100Decoder:
  def __init__(self, config: NLLBConfig, embed_tokens: nn.Embedding, max_cache_len: int = 0):
    self.embed_tokens = embed_tokens
    self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
    self.embed_positions = SinusoidalPositionalEmbedding(
      config.max_position_embeddings, config.d_model, config.pad_token_id
    )
    self.layers = [M2M100DecoderLayer(config, max_cache_len=max_cache_len)
                   for _ in range(config.decoder_layers)]
    self.layer_norm = nn.LayerNorm(config.d_model)
    self.dropout = config.dropout
    self.pad_token_id = config.pad_token_id

  def __call__(
    self,
    input_ids: Tensor,
    encoder_hidden_states: Optional[Tensor] = None,
    encoder_attention_mask: Optional[Tensor] = None,
    start_pos: Union[int, UOp] = 0,
  ) -> Tensor:
    """
    input_ids:             (batch, tgt_len)  — token ids for this step
    encoder_hidden_states: (batch, src_len, d_model) — encoder output
    encoder_attention_mask:(batch, src_len) — binary mask for encoder tokens
    start_pos:             int or Variable, current decoding position (for KV cache)
    Returns:               (batch, tgt_len, d_model)
    """
    # Concrete value needed for shapes/triu; symbolic used for cache slicing
    start_pos_val = start_pos if isinstance(start_pos, int) else start_pos.val
    bsz, tgt_len = input_ids.shape
    x = self.embed_tokens(input_ids) * self.embed_scale
    x = x + self.embed_positions(input_ids, past_key_values_length=start_pos_val)
    x = x.dropout(self.dropout)

    # Causal mask — use concrete value for triu diagonal (JIT traces one graph per shape)
    # When decoding one token at a time, the mask is all-zeros (no masking needed) so skip it.
    if tgt_len > 1:
      total_len = start_pos_val + tgt_len
      causal_mask = Tensor.full((tgt_len, total_len), float("-inf")).triu(start_pos_val + 1)
      causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    else:
      causal_mask = None

    # Encoder cross-attention mask
    if encoder_attention_mask is not None:
      cross_mask = (1.0 - encoder_attention_mask.cast(dtypes.float32)).unsqueeze(1).unsqueeze(2) * -1e9
    else:
      cross_mask = None

    for layer in self.layers:
      x = layer(x, encoder_hidden_states=encoder_hidden_states,
                 attention_mask=causal_mask, encoder_attention_mask=cross_mask,
                 start_pos=start_pos)
    x = self.layer_norm(x)
    return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class NLLBModel:
  """
  NLLB-200 translation model (M2M100 architecture).

  Usage:
    model = NLLBModel(config, max_cache_len=200)
    load_state_dict(model, weights)

    # encode once
    enc_out = model.encode(input_ids, attention_mask)
    # decode autoregressively
    logits = model.decode(decoder_input_ids, enc_out, attention_mask, start_pos=0)
  """

  def __init__(self, config: NLLBConfig = NLLBConfig(), max_cache_len: int = 200):
    self.config = config
    # Shared embedding (encoder + decoder share weights with lm_head)
    self.shared = nn.Embedding(config.vocab_size, config.d_model)
    self.encoder = M2M100Encoder(config, self.shared)
    self.decoder = M2M100Decoder(config, self.shared, max_cache_len=max_cache_len)
    # lm_head weight is tied to shared.weight (no separate parameter)
    # We do the projection manually in decode() using shared.weight

  def encode(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
    """Run the encoder. Call once per input sequence."""
    return self.encoder(input_ids, attention_mask)

  def decode(
    self,
    decoder_input_ids: Tensor,
    encoder_hidden_states: Tensor,
    encoder_attention_mask: Optional[Tensor] = None,
    start_pos: Union[int, UOp] = 0,
  ) -> Tensor:
    """
    One decoder step.
    Returns logits: (batch, tgt_len, vocab_size)
    """
    hidden = self.decoder(decoder_input_ids, encoder_hidden_states,
                          encoder_attention_mask, start_pos=start_pos)
    # lm_head: linear projection with tied weights (no bias)
    logits = hidden @ self.shared.weight.T
    return logits


# ---------------------------------------------------------------------------
# Weight loading helper
# ---------------------------------------------------------------------------

def remap_weights(state_dict: dict) -> dict:
  """
  Remap HuggingFace M2M100ForConditionalGeneration weight keys to our tinygrad model.

  HF key structure:
    model.shared.weight
    model.encoder.embed_positions.weights    (sinusoidal; we regenerate, can skip)
    model.encoder.layers.{i}.self_attn.q_proj.weight / .bias
    model.encoder.layers.{i}.self_attn.k_proj.weight / .bias
    model.encoder.layers.{i}.self_attn.v_proj.weight / .bias
    model.encoder.layers.{i}.self_attn.out_proj.weight / .bias
    model.encoder.layers.{i}.self_attn_layer_norm.weight / .bias
    model.encoder.layers.{i}.fc1.weight / .bias
    model.encoder.layers.{i}.fc2.weight / .bias
    model.encoder.layers.{i}.final_layer_norm.weight / .bias
    model.encoder.layer_norm.weight / .bias
    (decoder same + encoder_attn instead of second self_attn)
    lm_head.weight    (tied to model.shared.weight — can skip)

  Our key structure (tinygrad get_state_dict naming):
    shared.weight
    encoder.embed_tokens.weight  (same tensor as shared)
    encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}
    encoder.layers.{i}.self_attn_layer_norm.{weight,bias}
    encoder.layers.{i}.fc1.{weight,bias}
    encoder.layers.{i}.fc2.{weight,bias}
    encoder.layers.{i}.final_layer_norm.{weight,bias}
    encoder.layer_norm.{weight,bias}
    decoder.embed_tokens.weight  (same tensor as shared)
    decoder.layers.{i}.self_attn.*
    decoder.layers.{i}.self_attn_layer_norm.*
    decoder.layers.{i}.encoder_attn.*
    decoder.layers.{i}.encoder_attn_layer_norm.*
    decoder.layers.{i}.fc1/fc2.*
    decoder.layers.{i}.final_layer_norm.*
    decoder.layer_norm.*
  """
  new = {}
  for k, v in state_dict.items():
    # Strip the top-level "model." prefix
    if k.startswith("model."):
      new_k = k[len("model."):]
    elif k == "lm_head.weight":
      # Tied to shared.weight — skip (we use shared.weight directly in decode())
      continue
    else:
      new_k = k

    # embed_positions are fixed/sinusoidal — skip (we regenerate them)
    if "embed_positions" in new_k:
      continue

    # embed_tokens keys: encoder.embed_tokens & decoder.embed_tokens are tied to shared
    # Map them all to shared.weight so load_state_dict sets them once
    if new_k in ("encoder.embed_tokens.weight", "decoder.embed_tokens.weight"):
      new_k = "shared.weight"

    new[new_k] = v
  return new
