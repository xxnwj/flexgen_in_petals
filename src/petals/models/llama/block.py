"""
LLaMA intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
See commit history for authorship.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.llama.modeling_llama import (
    # LlamaAttention,
    LlamaConfig,
    # LlamaDecoderLayer,
    # LlamaMLP,
    LlamaRMSNorm,
    repeat_kv,
    rotate_half,
)




from petals.utils.cuda_graphs import make_inference_graphed_callable
# from petals.flexgen_utils.utils import ExecutionEnv
from petals.flexgen_utils.ExecutionEnv import ExecutionEnv
from petals.flexgen_utils.compression import CompressionConfig
from petals.flexgen_utils.policy import Policy
from petals.flexgen_utils.pytorch_backend import fix_recursive_import
from petals.flexgen_utils.utils import ValueHolder, array_1d, array_2d
from petals.models.llama.flex_llama import FLEX_LlamaAttention, FLEX_LlamaMLP, LlamaDecoderLayer
from petals.models.llama.llama_config import get_llama_config
import os

fix_recursive_import()



def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

#### ##### weight_specs = [
            # ((v, h), dtype, path + "embed_tokens.weight"),
            # ]
def init_weight_list(weight_specs, policy, env):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))
        ret.append(weight)
    return ret
    


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"





class OptimizedLlamaAttention(FLEX_LlamaAttention):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args,  **kwargs)
    # def __init__(self, *args, env, policy, layer_id, **kwargs):
    #     super().__init__(*args, env, policy, layer_id, **kwargs)
        self._rotary_graph = None
        
        # self.env = env
        # self.layer_id = layer_id
        # self.policy = policy
        # self.compute = self.env.gpu

        # self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
        #                         else self.compute)
        # self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
        #                           else self.env.gpu)
        
        # self.task = None
        
        
        
        

    def _optimized_apply_rotary(self, query_states, key_states, cos, sin):
        if self._rotary_graph is None:
            self._rotary_graph = make_inference_graphed_callable(
                apply_rotary_pos_emb, sample_args=(query_states, key_states, cos, sin)
            )
        return self._rotary_graph(query_states, key_states, cos, sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert not output_attentions
        if position_ids is None:
            past_seen_tokens = past_key_value[0].shape[2] if past_key_value is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

        if q_len == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
            query_states, key_states = self._optimized_apply_rotary(query_states, key_states, cos, sin)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class OptimizedLlamaDecoderLayer(LlamaDecoderLayer):  # used in block_utils.py return config.block_class(config)
    def __init__(self, config: LlamaConfig,layer_id: int, env: ExecutionEnv, policy: Policy, weight_home: array_1d, path: str, ):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        print('OptimizedLlamaDecoderLayer config ', config)
        self.config = config
        self.num_heads = config.num_attention_heads
        # self.self_attn = OptimizedLlamaAttention(config=config, layer_idx=0)
        # self.mlp = LlamaMLP(config=config)
        ########---------------------------------------------
        self.self_attn = OptimizedLlamaAttention(config=config, env=env, policy=policy, layer_id=self.layer_id )
        #layer_idx only matters for KV caching, and we re-implement it in Petals
        self.mlp = FLEX_LlamaMLP(config=config, env=env, policy=policy,layer_id=self.layer_id )
         ########---------------------------------------------
        # self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.pre_attn_graph = None
        self.post_attn_graph = None
        
        
        
        self.llama_config = get_llama_config('huggyllama/llama-7b')
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches
        layers = []
        layers.append(self.self_attn)
        layers.append(self.mlp)
        # layers.append(self.input_layernorm)
        # layers.append(self.post_attention_layernorm)
        
        self.layers = layers
        self.num_layers = len(layers)
        # print('block.py, class OptimizedLlamaDecoderLayer(LlamaDecoderLayer): self.mlp ', self.mlp)
        # dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
        # dev_choices = [env.disk, env.cpu, env.gpu]
        
        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()
        
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        print('before init_all_weights OptimizedLlamaDecoderLayer self.config', self.config)
        # Initialize weights and apply final processing
        self.init_all_weights()
        print('OptimizedLlamaDecoderLayer self.config', self.config)
        
    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)
            
    def init_weight(self, j):
        print('self.llama_config ', self.llama_config)
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.llama_config.name}-np")))
        check_path = os.path.join(expanded_path, "embed_tokens.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_llama_weights(self.llama_config.name, self.path)
        self.layers[j].init_weight(self.weight_home[j], expanded_path)
        
    def _optimized_input_layernorm(self, hidden_states):
        if self.pre_attn_graph is None:
            self.pre_attn_graph = make_inference_graphed_callable(
                self.input_layernorm.forward, sample_args=(hidden_states,)
            )
        return self.pre_attn_graph(hidden_states)

    def _optimized_output_layernorm(self, hidden_states):
        if self.post_attn_graph is None:
            self.post_attn_graph = make_inference_graphed_callable(
                self.post_attention_layernorm.forward, sample_args=(hidden_states,)
            )
        return self.post_attn_graph(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        if hidden_states.size(1) == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
            hidden_states = self._optimized_input_layernorm(hidden_states)
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids, # i, the iterator of sequence length
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        if hidden_states.size(1) == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
            hidden_states = self._optimized_output_layernorm(hidden_states)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
        # import pdb; pdb.set_trace() ###### <------
        hidden_states = self.mlp(hidden_states) ###### <------
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def set_task(self, task):
        self.self_attn.set_task(task)
        self.mlp.set_task(task)


class WrappedLlamaBlock(OptimizedLlamaDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        batch_size, seq_length, _ = hidden_states.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        past_key_value = layer_past
        if past_key_value is not None:
            past_key_values_length = past_key_value[0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            past_key_value = self._reorder_cache_from_bloom_to_llama(past_key_value, batch_size, past_key_values_length)

        assert position_ids is None

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=hidden_states,
            past_key_values_length=past_key_values_length,
        )

        outputs = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )

        if use_cache:
            present_key_value = outputs[-1]
            present_key_value = self._reorder_cache_from_llama_to_bloom(
                present_key_value, batch_size, seq_length_with_past
            )
            outputs = outputs[:-1] + (present_key_value,)

        return outputs

    def _reorder_cache_from_bloom_to_llama(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        key_states = key_states.permute(0, 2, 1)
        key_states = key_states.view(
            batch_size, self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        value_states = value_states.view(*key_states.shape)
        return (key_states, value_states)

    def _reorder_cache_from_llama_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        value_states = value_states.view(
            batch_size * self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        key_states = key_states.view(*value_states.shape)
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)


###############################################################################################
    def load_weight(self, i, j, k, overlap=True):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return
        # Load from weight_home to weight_read_buf
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()

    def init_cache(self, j, k):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if i == 0:  # prefill, no cache
            return
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def store_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos - 1:pos])
        else:  # load from the last layer
            val = self.hidden[i][j - 1][k].pop().move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        # Store to hidden states buffers
        if j == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos + 1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos + 1] = ids
        else:  # move to home
            x = self.hidden[i][j][k]
            if x.val:  # x may already be moved due to overlapping
                x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
                               self.weight_read_buf[j], self.attention_mask[k],
                               self.cache_write_buf[j][k], i, k)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)


