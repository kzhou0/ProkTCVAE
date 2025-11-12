# modeling_base.py
# Single-file Bacformer-style Rotary Transformer + VAE
# Reconciles with:
#   - bacformer.modeling.config.BacformerConfig  (your provided config)
#   - bacformer.modeling.utils.create_4d_from_2d_attn_mask (your util)
#
# Key points:
# - Encoder is bidirectional; Decoder is causal.
# - Optional latent conditioning via add-to-embeddings and/or KV injection.
# - Works even if your config doesn't define latent-related fields (defaults applied).

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List, Literal, Tuple

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from bacformer.modeling.config import BacformerConfig, SPECIAL_TOKENS_DICT
from bacformer.modeling.utils import create_4d_from_2d_attn_mask


# =========================
# Output container
# =========================
@dataclass
class BacformerModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    attentions: Optional[List[torch.FloatTensor]] = None
    pooler_output: Optional[torch.FloatTensor] = None


# =========================
# Rotary helpers (LLaMA-style)
# =========================
def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: [B, L, nH, Dh]
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    freqs_cos = _reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = _reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


# =========================
# Attention utils
# =========================
def _make_causal_4d_mask(B: int, L: int, device: torch.device, dtype: torch.dtype = torch.bool) -> torch.Tensor:
    """Return [B, 1, L, L] boolean causal mask (True=visible)."""
    tril = torch.ones((L, L), dtype=torch.bool, device=device).tril()
    mask = tril[None, None, :, :].expand(B, 1, L, L).to(dtype)
    return mask


def _expand_k_with_z_slot(attn_mask_4d: Optional[torch.Tensor], visible_for_z: bool = True) -> Optional[torch.Tensor]:
    """Append one key column for latent z KV to [B,H,Lq,Lk] boolean mask."""
    if attn_mask_4d is None:
        return None
    B, H, Lq, Lk = attn_mask_4d.shape
    col = torch.ones((B, H, Lq, 1), dtype=attn_mask_4d.dtype, device=attn_mask_4d.device) if visible_for_z \
        else torch.zeros((B, H, Lq, 1), dtype=attn_mask_4d.dtype, device=attn_mask_4d.device)
    return torch.cat([col, attn_mask_4d], dim=-1)  # z-slot at K index 0


def _sdpa_with_weights(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, training=True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (attn_output, attn_weights)."""
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None or attn_mask.dtype == torch.bool
        causal = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(0)
        attn_bias.masked_fill_(~causal, float("-inf"))

    if attn_mask is not None:
        # boolean mask expected: False = masked
        assert attn_mask.dtype == torch.bool, "Provide boolean attention_mask for SDPA weights path."
        # broadcast-friendly addition: convert boolean -> additive bias
        bias = torch.zeros_like(attn_bias)
        # If attn_mask is [B,H,Lq,Lk], we assume B=H=1 here; callers broadcast per-batch/head internally.
        # For simplicity we take the first batch/head (typical when called per-sample/head).
        # If you pass full [B,H,Lq,Lk], pre-broadcast to [1,1,L,S] before calling this helper.
        am = attn_mask
        if am.dim() == 4:
            am = am[0, 0]
        bias.masked_fill_(~am, float("-inf"))
        attn_bias = attn_bias + bias

    attn_weight = (query @ key.transpose(-2, -1)) * scale_factor
    attn_weight = attn_weight + attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=training)
    attn_output = attn_weight @ value
    return attn_output, attn_weight


# =========================
# Rotary Self-Attention (+ optional latent KV)
# =========================
class RotarySelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_head = embed_dim // num_heads
        self.dropout_rate = dropout

        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,                              # [B, L, H]
        attn_mask: Optional[torch.Tensor],            # [B, heads, Lq, Lk] bool
        freqs_cos: torch.Tensor,                      # [L, Dh]
        freqs_sin: torch.Tensor,                      # [L, Dh]
        is_causal: bool = False,
        return_attn_weights: bool = False,
        z_k: Optional[torch.Tensor] = None,           # [B, heads, Z=1, Dh]
        z_v: Optional[torch.Tensor] = None,           # [B, heads, Z=1, Dh]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, Hdim = x.shape
        xq, xk, xv = self.q(x), self.k(x), self.v(x)
        # [B, L, heads, Dh]
        xq = xq.view(B, L, self.num_heads, self.dim_head)
        xk = xk.view(B, L, self.num_heads, self.dim_head)
        xv = xv.view(B, L, self.num_heads, self.dim_head)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        # [B, heads, L, Dh]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if z_k is not None and z_v is not None:
            xk = torch.cat([z_k, xk], dim=2)  # prepend latent slot
            xv = torch.cat([z_v, xv], dim=2)

        if return_attn_weights:
            att, aw = _sdpa_with_weights(
                xq, xk, xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=is_causal,
                training=self.training,
            )
        else:
            att = scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=is_causal,
            )
            aw = None

        out = att.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.out_proj(out), aw


# =========================
# Transformer Layer / Encoder
# =========================
class BacformerTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu"] = "gelu",
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.self_mha = RotarySelfAttention(
            embed_dim=hidden_size, num_heads=num_attention_heads, dropout=dropout
        )
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
        is_causal: bool = False,
        z_k: Optional[torch.Tensor] = None,
        z_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_outputs, attn_weights = self.self_mha(
            hidden_state,
            attn_mask=attention_mask,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            return_attn_weights=return_attn_weights,
            is_causal=is_causal,
            z_k=z_k,
            z_v=z_v,
        )
        x = self.norm1(hidden_state + self.dropout1(attn_outputs))
        ff_output = self.fc2(self.dropout2(self.activation(self.fc1(x))))
        x = self.norm2(x + self.dropout3(ff_output))
        return x, attn_weights


class BacformerTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu"] = "gelu",
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BacformerTransformerLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
        is_causal: bool = False,
        z_k: Optional[torch.Tensor] = None,
        z_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        attn_weights_arr: List[Optional[torch.Tensor]] = []
        for layer in self.layers:
            hidden_state, aw = layer(
                hidden_state=hidden_state,
                attention_mask=attention_mask,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                return_attn_weights=return_attn_weights,
                is_causal=is_causal,
                z_k=z_k,
                z_v=z_v,
            )
            attn_weights_arr.append(aw)
        return hidden_state, attn_weights_arr


# =========================
# Embeddings (DNA/protein-ready)
# =========================
class GenomicEmbeddings(nn.Module):
    """Construct the protein embeddings from protein sequence, position embeddings and sequence type embeddings."""

    def __init__(self, config: BacformerConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.token_type_embeddings = nn.Embedding(
            num_embeddings=config.max_token_type_embeddings + 1,
            embedding_dim=config.hidden_size,
            padding_idx=config.max_token_type_embeddings,
        )

        self.special_tokens_embeddings = nn.Embedding(
            num_embeddings=config.num_special_tokens,
            embedding_dim=config.hidden_size,
        )
        self.prot_emb_token_id = config.prot_emb_token_id
        self.pad_token_id = config.pad_token_id

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        protein_embeddings: torch.Tensor = None,
        special_tokens_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None,  # used for causal protein family modeling
        property_ids: torch.Tensor = None,  # used for conditional fine-tuning for desired property
    ) -> torch.Tensor:
        """Forward pass for protein embeddings."""
        bs, seq_length, dim = protein_embeddings.shape

        # pass the pooled ESM protein embeddings through a linear layer
        protein_embeddings = self.linear(protein_embeddings.type_as(self.linear.weight))
        protein_embeddings = torch.where(
            special_tokens_mask.unsqueeze(-1).repeat(1, 1, dim) == self.prot_emb_token_id,
            protein_embeddings,
            self.special_tokens_embeddings(special_tokens_mask),
        )

        if token_type_ids is not None:
            protein_embeddings += self.token_type_embeddings(token_type_ids)

        protein_embeddings = self.LayerNorm(protein_embeddings)
        protein_embeddings = self.dropout(protein_embeddings)
        return protein_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,                 # [B, L]
        special_tokens_mask: Optional[torch.Tensor] = None,       # [B, L]
        token_type_ids: Optional[torch.Tensor] = None,            # [B, L]
        external_embeddings: Optional[torch.Tensor] = None,       # [B, L, H]
    ) -> torch.Tensor:
        if external_embeddings is not None:
            embs = self.linear(external_embeddings.type_as(self.linear.weight))
        else:
            assert input_ids is not None, "Either input_ids or external_embeddings must be provided."
            embs = self.token_embeddings(input_ids)

        if special_tokens_mask is not None:
            B, L, H = embs.shape
            embs = torch.where(
                (special_tokens_mask.unsqueeze(-1).expand(B, L, H) == self.prot_emb_token_id),
                embs,
                self.special_tokens_embeddings(special_tokens_mask),
            )

        if token_type_ids is not None:
            embs = embs + self.token_type_embeddings(token_type_ids)

        embs = self.LayerNorm(embs)
        embs = self.dropout(embs)
        return embs


# =========================
# Pooler
# =========================
class BacformerPooler(nn.Module):
    def __init__(self, config: BacformerConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if padding_mask is not None:
            denom = padding_mask.sum(dim=1).clamp(min=1).unsqueeze(1)
            mean_hidden_states = torch.einsum("bij,bi->bj", hidden_states, padding_mask.to(hidden_states.dtype)) / denom
        else:
            mean_hidden_states = hidden_states.mean(dim=1)
        return self.activation(self.dense(mean_hidden_states))


# =========================
# PreTrained base
# =========================
class BacformerPreTrainedModel(PreTrainedModel):
    """Weight init consistent with your config."""
    config_class = BacformerConfig
    base_model_prefix = "bacformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GenomicEmbeddings", "BacformerTransformerLayer"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# =========================
# Encoder (bidirectional) -> mean/logvar
# =========================
class BacformerEncoder(nn.Module):
    def __init__(self, config: BacformerConfig):
        super().__init__()
        self.config = config
        self.encoder = BacformerTransformerEncoder(
            num_hidden_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            activation="gelu",
            layer_norm_eps=config.layer_norm_eps,
        )
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, int(config.max_position_embeddings * 1.5)
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: Optional[bool] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        return_attn_weights = self.config.return_attn_weights if return_attn_weights is None else return_attn_weights
        B, L, _ = hidden_states.shape
        last_hidden_state, attn_weights = self.encoder(
            hidden_state=hidden_states,
            attention_mask=attention_mask,
            freqs_cos=self.freqs_cos[:L, :].to(hidden_states.device, hidden_states.dtype),
            freqs_sin=self.freqs_sin[:L, :].to(hidden_states.device, hidden_states.dtype),
            return_attn_weights=return_attn_weights,
            is_causal=is_causal,
        )
        return last_hidden_state, attn_weights


class BacformerVAEEncoder(nn.Module):
    def __init__(self, config: BacformerConfig):
        super().__init__()
        self.config = config
        self.embeddings = GenomicEmbeddings(config)
        self.encoder = BacformerEncoder(config)
        self.pooler = BacformerPooler(config)
        # Handle missing VAE fields gracefully
        latent_size = int(getattr(config, "latent_size", 256))
        self.to_mean = nn.Linear(config.hidden_size, latent_size)
        self.to_logvar = nn.Linear(config.hidden_size, latent_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        external_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.embeddings(
            input_ids=input_ids,
            special_tokens_mask=special_tokens_mask,
            token_type_ids=token_type_ids,
            external_embeddings=external_embeddings,
        )
        attn_4d = None
        if attention_mask is not None:
            attn_4d = create_4d_from_2d_attn_mask(attention_mask, self.config.num_attention_heads).bool()
        h, _ = self.encoder(hidden_states=x, attention_mask=attn_4d, is_causal=False)
        pooled = self.pooler(h, padding_mask=attention_mask)
        mean, logvar = self.to_mean(pooled), self.to_logvar(pooled)
        return mean, logvar, h


# =========================
# Decoder (causal) with z-add and/or z-KV
# =========================
class ZInputAdaptor(nn.Module):
    def __init__(self, hidden_size: int, latent_size: int):
        super().__init__()
        self.proj = nn.Linear(latent_size, hidden_size, bias=False)
    def forward(self, token_embs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return token_embs + self.proj(z).unsqueeze(1)


class ZKVProj(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, latent_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.proj_k = nn.Linear(latent_size, hidden_size, bias=False)
        self.proj_v = nn.Linear(latent_size, hidden_size, bias=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = z.size(0)
        k = self.proj_k(z).view(B, self.num_heads, 1, self.head_dim)
        v = self.proj_v(z).view(B, self.num_heads, 1, self.head_dim)
        return k, v


class BacformerVAEDecoder(nn.Module):
    def __init__(self, config: BacformerConfig):
        super().__init__()
        self.config = config
        self.embeddings = GenomicEmbeddings(config)
        self.encoder = BacformerTransformerEncoder(
            num_hidden_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            activation="gelu",
            layer_norm_eps=config.layer_norm_eps,
        )
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, int(config.max_position_embeddings * 1.5)
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Flags (with safe defaults if not in config)
        self.add_input: bool = bool(getattr(config, "add_input", True))
        self.add_attn: bool = bool(getattr(config, "add_attn", True))
        self.attn_proj_vary: bool = bool(getattr(config, "attn_proj_vary", False))

        latent_size = int(getattr(config, "latent_size", 256))

        if self.add_input:
            self.z_in = ZInputAdaptor(config.hidden_size, latent_size)
        if self.add_attn:
            if self.attn_proj_vary:
                self.z_kv_layers = nn.ModuleList(
                    [ZKVProj(config.hidden_size, config.num_attention_heads, latent_size) for _ in range(config.num_hidden_layers)]
                )
            else:
                self.z_kv = ZKVProj(config.hidden_size, config.num_attention_heads, latent_size)

        # LM head (tie weights with token_embeddings if you want; left separate here)
        self.lm_head = nn.Linear(config.hidden_size, config.protein_clusters_vocab_size, bias=False)

    def _build_causal_mask(self, attention_mask_2d: Optional[torch.Tensor], L: int) -> Optional[torch.Tensor]:
        """Build [B,H,L,L] boolean mask combining padding & causality."""
        if attention_mask_2d is None:
            return None
        B = attention_mask_2d.size(0)
        pad = create_4d_from_2d_attn_mask(attention_mask_2d, self.config.num_attention_heads).bool()  # [B,H,1,L]
        causal = _make_causal_4d_mask(B, L, attention_mask_2d.device)  # [B,1,L,L]
        pad = pad.expand(-1, -1, L, -1)  # [B,H,L,L]
        return (pad & causal)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        external_embeddings: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        assert z is not None, "Decoder requires latent z"
        x = self.embeddings(
            input_ids=input_ids,
            special_tokens_mask=special_tokens_mask,
            token_type_ids=token_type_ids,
            external_embeddings=external_embeddings,
        )
        if self.add_input:
            x = self.z_in(x, z)

        B, L, _ = x.shape
        mask = self._build_causal_mask(attention_mask, L)

        last = x
        attn_list: List[Optional[torch.Tensor]] = []
        for i, layer in enumerate(self.encoder.layers):
            z_k = z_v = None
            if self.add_attn:
                if self.attn_proj_vary:
                    z_k, z_v = self.z_kv_layers[i](z)
                else:
                    z_k, z_v = self.z_kv(z)
            layer_mask = _expand_k_with_z_slot(mask, visible_for_z=True) if mask is not None else None

            last, a = layer(
                hidden_state=last,
                attention_mask=layer_mask,
                freqs_cos=self.freqs_cos[:last.size(1), :].to(last.device, last.dtype),
                freqs_sin=self.freqs_sin[:last.size(1), :].to(last.device, last.dtype),
                return_attn_weights=return_attn_weights,
                is_causal=True,
                z_k=z_k,
                z_v=z_v,
            )
            attn_list.append(a)

        logits = self.lm_head(last)
        return logits, attn_list


# =========================
# Top-level VAE model
# =========================
class BacformerVAE(BacformerPreTrainedModel):
    """Posterior/prior encoders + causal decoder with latent conditioning."""
    def __init__(self, config: BacformerConfig):
        super().__init__(config)
        self.config = config

        # Defaults if not present in your config
        self.latent_size: int = int(getattr(config, "latent_size", 256))
        self.beta_kl: float = float(getattr(config, "beta_kl", 1.0))
        self.learn_prior: bool = bool(getattr(config, "learn_prior", False))

        self.posterior = BacformerVAEEncoder(config)
        self.prior = BacformerVAEEncoder(config) if self.learn_prior else None
        self.decoder = BacformerVAEDecoder(config)

        self.post_init()

    @staticmethod
    def reparam(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mean + eps * std

    @staticmethod
    def kl(mean1: torch.Tensor, logvar1: torch.Tensor, mean2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
        # KL( N(mean1, diag(exp(logvar1))) || N(mean2, diag(exp(logvar2))) )
        var1 = logvar1.exp()
        var2 = logvar2.exp()
        term = (var1 / var2) + ((mean2 - mean1) ** 2) / var2
        kl = 0.5 * (logvar2 - logvar1 - 1 + term).sum(dim=-1)
        return kl.mean()

    def forward(
        self,
        # posterior (y) inputs
        y_input_ids: Optional[torch.Tensor] = None,
        y_special_tokens_mask: Optional[torch.Tensor] = None,
        y_token_type_ids: Optional[torch.Tensor] = None,
        y_attention_mask: Optional[torch.Tensor] = None,
        # prior (x) inputs (only if learn_prior=True)
        x_input_ids: Optional[torch.Tensor] = None,
        x_special_tokens_mask: Optional[torch.Tensor] = None,
        x_token_type_ids: Optional[torch.Tensor] = None,
        x_attention_mask: Optional[torch.Tensor] = None,
        # decoder inputs (teacher forcing)
        dec_input_ids: Optional[torch.Tensor] = None,
        dec_special_tokens_mask: Optional[torch.Tensor] = None,
        dec_token_type_ids: Optional[torch.Tensor] = None,
        dec_attention_mask: Optional[torch.Tensor] = None,
        # flags
        from_prior: bool = False,
        from_mean: bool = False,
        labels: Optional[torch.Tensor] = None,  # [B, L] with pad index masked out
        return_attn_weights: bool = False,
        return_dict: Optional[bool] = None,
    ) -> BacformerModelOutput | tuple:
        return_dict = self.config.return_dict if return_dict is None else return_dict

        # Posterior over z from y
        post_mean, post_logvar, _ = self.posterior(
            input_ids=y_input_ids,
            special_tokens_mask=y_special_tokens_mask,
            token_type_ids=y_token_type_ids,
            attention_mask=y_attention_mask,
        )

        # Prior (conditional) or standard normal
        if self.learn_prior:
            assert x_input_ids is not None and x_attention_mask is not None, "learn_prior=True requires x_* inputs"
            prior_mean, prior_logvar, _ = self.prior(
                input_ids=x_input_ids,
                special_tokens_mask=x_special_tokens_mask,
                token_type_ids=x_token_type_ids,
                attention_mask=x_attention_mask,
            )
        else:
            prior_mean = torch.zeros_like(post_mean)
            prior_logvar = torch.zeros_like(post_logvar)

        mean, logvar = (prior_mean, prior_logvar) if from_prior else (post_mean, post_logvar)
        z = mean if from_mean else self.reparam(mean, logvar)

        # Decode
        logits, attns = self.decoder(
            input_ids=dec_input_ids,
            special_tokens_mask=dec_special_tokens_mask,
            token_type_ids=dec_token_type_ids,
            attention_mask=dec_attention_mask,
            z=z,
            return_attn_weights=return_attn_weights,
        )

        loss = None
        if labels is not None:
            ce = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            kl = self.kl(post_mean, post_logvar, prior_mean, prior_logvar)
            loss = ce + self.beta_kl * kl

        if not return_dict:
            # tuple for classic HF pattern
            return (loss, logits, attns)

        return BacformerModelOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=None,
            attentions=attns,
            pooler_output=None,
        )
