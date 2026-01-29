from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import RMSNorm
import math

class ImageProjModel(nn.Module):
    """Projection Model"""

    def __init__(self, in_channels=64, inner_dim=3072):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(in_channels, inner_dim),
            torch.nn.GELU(),
            torch.nn.Linear(inner_dim, inner_dim),
        )
        self.norm = torch.nn.LayerNorm(inner_dim)

    def forward(self, hidden_states):

        hidden_states = self.proj(hidden_states)
        return self.norm(hidden_states)

class MakeupFluxAttnProcessor(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, inner_dim=3072):
        super().__init__()

        rank=1024
        """ref"""
        self.ref_to_k_lora_down = nn.Linear(inner_dim, rank, bias=False)
        self.ref_to_k_lora_up = nn.Linear(rank, inner_dim, bias=False)
        self.ref_to_v_lora_down = nn.Linear(inner_dim, rank, bias=False)
        self.ref_to_v_lora_up = nn.Linear(rank, inner_dim, bias=False)
        nn.init.kaiming_uniform_(self.ref_to_k_lora_down.weight, a=math.sqrt(5)) 
        nn.init.zeros_(self.ref_to_k_lora_up.weight)
        nn.init.kaiming_uniform_(self.ref_to_v_lora_down.weight, a=math.sqrt(5)) 
        nn.init.zeros_(self.ref_to_v_lora_up.weight)

        self.ref_norm_k = RMSNorm(128, eps=1e-5, elementwise_affine=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        cond_image_rotary_emb: Optional[torch.Tensor] = None,
        ref_hidden_states: torch.Tensor = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if ref_hidden_states is not None:
            ref_hidden_states_key_proj = self.ref_to_k_lora_up(self.ref_to_k_lora_down(ref_hidden_states))
            ref_hidden_states_value_proj = self.ref_to_v_lora_up(self.ref_to_v_lora_down(ref_hidden_states))

            ref_hidden_states_key_proj = ref_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            ref_hidden_states_value_proj = ref_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if self.ref_norm_k is not None:
                ref_hidden_states_key_proj = self.ref_norm_k(ref_hidden_states_key_proj)
            
            if cond_image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb

                ref_hidden_states_key_proj = apply_rotary_emb(ref_hidden_states_key_proj, cond_image_rotary_emb[0])
            # 跟query拼起来
            key = torch.cat([key, ref_hidden_states_key_proj], dim=2)
            value = torch.cat([value, ref_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states