import functools
import quarot
import quarot.transformers
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention,\
      LlamaForCausalLM, apply_rotary_pos_emb, LlamaMLP, \
        eager_attention_forward, ALL_ATTENTION_FUNCTIONS, \
            FlashAttentionKwargs, Unpack
from transformers.models.clip.modeling_clip import CLIPMLP, CLIPFlashAttention2, CLIPAttention
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from typing import Optional, Tuple, Callable
from transformers import Cache
import logging
from llava.model import LlavaLlamaForCausalLM
from e2e.quantized_llava.module_quant import OnlineHadamard, Linear4bit, Quantizer

ALL_LAYERNORM_LAYERS.append(quarot.nn.RMSNorm)

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class QuarotFP16clipAttention(CLIPAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = torch.nn.Identity()
        self.o_proj_hadamard = torch.nn.Identity()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()
        hidden_states = self.quantizer(hidden_states)
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = self.o_proj_hadamard(attn_output.transpose(-1, -2)).transpose(-1, -2)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

class Quarotonlinev1clipAttention(QuarotFP16clipAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_proj = torch.nn.Sequential(
            (self.config.hidden_size),
            Quantizer(),
            Linear4bit.from_float(self.q_proj),
        )
        self.k_proj = torch.nn.Sequential(
            OnlineHadamard(self.config.hidden_size),
            Quantizer(),
            Linear4bit.from_float(self.k_proj),
        )
        self.v_proj = torch.nn.Sequential(
            OnlineHadamard(self.config.hidden_size),
            Quantizer(),
            Linear4bit.from_float(self.v_proj),
        )
        self.o_proj_hadamard = OnlineHadamard(self.config.num_attention_heads)
        self.out_proj = torch.nn.Sequential(
            Quantizer(),
            Linear4bit.from_float(self.out_proj)
        )

class QuarotonlineclipAttention(QuarotFP16clipAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # here implement ln online
        self.quantizer = torch.nn.Sequential(
            OnlineHadamard(self.config.hidden_size),
            Quantizer(),
        )
        self.q_proj = Linear4bit.from_float(self.q_proj)
        self.k_proj = Linear4bit.from_float(self.k_proj)
        self.v_proj = Linear4bit.from_float(self.v_proj)
        self.o_proj_hadamard = OnlineHadamard(self.config.num_attention_heads)
        self.out_proj = torch.nn.Sequential(
            Quantizer(),
            Linear4bit.from_float(self.out_proj)
        )

class QuarotonlineclipMLP(CLIPMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.quantizer = Quantizer()
        self.fc1 = torch.nn.Sequential(
            OnlineHadamard(self.config.hidden_size),
            Quantizer(),
            Linear4bit.from_float(self.fc1)
        )
        self.fc2 = torch.nn.Sequential(
            OnlineHadamard(self.config.intermediate_size),
            Quantizer(),
            Linear4bit.from_float(self.fc2)
        )

    def forward(self, x):
        # x = self.quantizer(x)
        return super().forward(x)
    

class QuarotMMprojector(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Identity()
    
    def forward(self, x):
        return self.linear(x)
    
    def from_float(self, projector, config):
        # for module in projector:
        fc1 = projector[0]
        act = projector[1]
        fc2 = projector[2]
        self.linear = torch.nn.Sequential(
            OnlineHadamard(config.mm_hidden_size),
            Quantizer(),
            Linear4bit.from_float(fc1),
            act,
            OnlineHadamard(config.hidden_size),
            Quantizer(),
            Linear4bit.from_float(fc2),
        )
        # just a wrapper, just return the linear
        return self.linear

class QuarotFP16LlamaAttention(LlamaAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = torch.nn.Identity()
        self.o_proj_hadamard = torch.nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        hidden_states = self.quantizer(hidden_states)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states.contiguous(), value_states.contiguous(), self.layer_idx, cache_kwargs)


        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logging.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = self.o_proj_hadamard(attn_output.transpose(-1, -2)).transpose(-1, -2)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class QuarotLlamaAttention(QuarotFP16LlamaAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = Quantizer()
        self.q_proj = Linear4bit.from_float(self.q_proj)
        self.k_proj = Linear4bit.from_float(self.k_proj)
        self.v_proj = Linear4bit.from_float(self.v_proj)
        self.o_proj_hadamard = OnlineHadamard(self.config.num_attention_heads)
        self.o_proj = torch.nn.Sequential(
            Quantizer(),
            Linear4bit.from_float(self.o_proj)
        )

class QuarotLlamaMLP(LlamaMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = Quantizer()
        self.up_proj = Linear4bit.from_float(self.up_proj)
        self.gate_proj = Linear4bit.from_float(self.gate_proj)
        self.down_proj = torch.nn.Sequential(
            OnlineHadamard(self.intermediate_size),
            Quantizer(),
            Linear4bit.from_float(self.down_proj)
        )

    def forward(self, x):
        x = self.quantizer(x)
        return super().forward(x)


class QuarotFP16LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = QuarotFP16LlamaAttention(config=config, layer_idx=layer_idx)
        self.cache_dtype = "float16"
        self._expected_max_length = None

        
    def build_cache(self, batch_size, page_size, max_length):
        device = self.model.layers[0].self_attn.v_proj.weight.device
        dtype = self.cache_dtype or self.model.layers[0].self_attn.v_proj.weight.dtype
        
        num_heads = self.config.num_key_value_heads
        model_dim = self.config.hidden_size
        head_dim = model_dim // num_heads
        disable_quant = self.cache_dtype == "float16" 
        return quarot.transformers.MultiLayerPagedKVCache4Bit(
            batch_size=batch_size,
            page_size=page_size, 
            max_seq_len=max_length, 
            device=device, 
            n_layers=len(self.model.layers),
            num_heads=num_heads,
            head_dim=head_dim,
            disable_quant=disable_quant,
            hadamard_dtype=None if disable_quant else torch.float16
        )

    def _get_logits_processor(self, generation_config, *args, **kwargs):
        # This is a hack to get the max length from generation_config.
        # Doing it here because max_length might not be set before this 
        # method is called.
        self._expected_max_length = generation_config.max_length # This value will be reset at the next forward call
        return super()._get_logits_processor(generation_config, *args, **kwargs)


    def forward(self, input_ids, *args, past_key_values=None, **kwargs):
        if past_key_values is None:
            max_length = self._expected_max_length or input_ids.shape[1]
            self._expected_max_length = None # Reset this value.
            past_key_values = self.build_cache(
                input_ids.shape[0], 
                page_size=max_length,  # For now working with single page per batch.
                max_length=max_length)
        out = super().forward(input_ids, *args, past_key_values=past_key_values, **kwargs)
        return out
    


class QuarotLlavaLlamaForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self,config, quant_part='all', *args, **kwargs):
        super(QuarotLlavaLlamaForCausalLM, self).__init__(config, *args, **kwargs)
        self.quant_part = quant_part
    def init_quant(self):
        if self.quant_part == 'all':
            config = self.config
            for layer_idx, layer in enumerate(self.model.layers):
                layer.self_attn = QuarotLlamaAttention(config=config, layer_idx=layer_idx)
                layer.input_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                layer.post_attention_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                layer.mlp = QuarotLlamaMLP(config=config)
            config = self.model.vision_tower.config
            for layer_idx , layer in enumerate(self.model.vision_tower.vision_tower.vision_model.encoder.layers):
                layer.self_attn = QuarotonlineclipAttention(config=config)
                # layer.layer_norm1 =
                layer.mlp = QuarotonlineclipMLP(config = config)
            self.model.mm_projector = QuarotMMprojector().from_float(self.model.mm_projector,config = self.config)
        elif self.quant_part == 'vit':
            config = self.model.vision_tower.config
            for layer_idx , layer in enumerate(self.model.vision_tower.vision_tower.vision_model.encoder.layers):
                layer.self_attn = QuarotonlineclipAttention(config=config)
                # layer.layer_norm1 =
                layer.mlp = QuarotonlineclipMLP(config = config)
            self.model.mm_projector = QuarotMMprojector().from_float(self.model.mm_projector,config = self.config)
        elif self.quant_part == 'lm':
            config = self.config
            for layer_idx, layer in enumerate(self.model.model.layers):
                layer.self_attn = QuarotLlamaAttention(config=config)
                layer.input_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                layer.post_attention_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                layer.mlp = QuarotLlamaMLP(config=config)
        self.cache_dtype = "int4"

# class QuarotLlamallavaForCausalLM(QuarotFP16LlamaForCausalLM):
#     def __init__(self, config):
#         super().__init__(config)
#         assert config._attn_implementation == "flash_attention_2"
#         self.norm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         for layer_idx, layer in enumerate(self.model.layers):
#             layer.self_attn = QuarotLlamaAttention(config=config, layer_idx=layer_idx)
#             layer.input_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#             layer.post_attention_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#             layer.mlp = QuarotLlamaMLP(config=config)
#         self.cache_dtype = "int4"

