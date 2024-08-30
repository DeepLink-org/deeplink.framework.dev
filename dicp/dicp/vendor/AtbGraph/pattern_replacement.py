import functools
import torch
from dicp.dynamo_bridge.op_transformer import (
    BackendPatternBase,
    PatternMatcherPass,
    register_backend_patterns,
)
atb_pattern_matcher = PatternMatcherPass()

torch_patterns_cls_list_1 = []
register_torch_pattern_1 = functools.partial(
    register_backend_patterns, torch_patterns_cls_list_1)

torch_patterns_cls_list_2 = []
register_torch_pattern_2 = functools.partial(
    register_backend_patterns, torch_patterns_cls_list_2)


torch_patterns_cls_list_3 = []
register_torch_pattern_3 = functools.partial(
    register_backend_patterns, torch_patterns_cls_list_3)


aten = torch.ops.aten
atb = torch.ops.atb
infer_ext = torch.ops.infer_ext

@register_torch_pattern_1
class TorchAddRmsNorm(BackendPatternBase):
    @staticmethod
    def pattern(arg0, arg1, gamma, epsilon):
        add = torch.ops.atb.add.default(arg0, arg1)
        norm = torch.ops.infer_ext.rms_norm.default(add, gamma, epsilon)
        return norm

    @staticmethod
    def replacement(arg0, arg1, gamma, epsilon):
        add_and_norm = torch.ops.atb.add_and_rms_norm.default(arg0, arg1, gamma, epsilon)
        return add_and_norm

@register_torch_pattern_1
class TorchViewAndRope(BackendPatternBase):
    @staticmethod
    def pattern(query, key, cos, sin, seqlen, view_size):
        query = aten.view(query, [-1, view_size])
        key = aten.view(key, [-1, view_size])
        rope = atb.rope.default(query, key, cos, sin, seqlen)
        return rope

    @staticmethod
    def replacement(query, key, cos, sin, seqlen, view_size):
        rope = atb.view_and_rope.default(query, key, cos, sin, seqlen, view_size)
        return rope


@register_torch_pattern_1
class TorchFillKVCacaheAndContextAttention(BackendPatternBase):
    @staticmethod
    def pattern(query, key, value, k_cache, v_cache, kv_start_indices_1d, kv_seqlens_int, mask, num_heads, num_kv_heads, kv_head_size, block_size, hidden_size, llama_num_heads, llama_head_dim):
        query = aten.view.default(query, [-1, llama_num_heads, llama_head_dim])
        key = aten.view.default(key, [-1, llama_num_heads, llama_head_dim])
        k_cache = aten.view.default(k_cache, [-1, block_size, num_kv_heads, kv_head_size])
        v_cache = aten.view.default(v_cache, [-1, block_size, num_kv_heads, kv_head_size])
        fill_kv_cache = infer_ext.fill_kv_cache.default(key, value, k_cache, v_cache, kv_start_indices_1d)
        getitem_1 = fill_kv_cache[0]
        getitem_2 = fill_kv_cache[1]
        
        query = aten.view.default(query, [-1, hidden_size])
        key = aten.view.default(key, [-1, hidden_size])
        value = aten.view.default(value, [-1, hidden_size])
        attn_out = atb.context_attention.default(query, key, value, getitem_1, getitem_2, kv_seqlens_int, mask, num_heads, num_kv_heads)
        return attn_out

    @staticmethod
    def replacement(query, key, value, k_cache, v_cache, kv_start_indices_1d, kv_seqlens_int, mask, num_heads, num_kv_heads, kv_head_size, block_size):
        out = atb.atb_context_attention.default(query,
                                                          key,
                                                          value,
                                                          k_cache,
                                                          v_cache,
                                                          kv_start_indices_1d,
                                                          kv_seqlens_int,
                                                          mask,
                                                          num_heads,
                                                          num_kv_heads,
                                                          kv_head_size,
                                                          block_size)
        return out


@register_torch_pattern_2
class TorchFused(BackendPatternBase):
    @staticmethod
    def pattern(query, key, value, k_cache, v_cache, kv_start_indices_1d, kv_seqlens_int, mask, num_heads, num_kv_heads, kv_head_size, block_size,
                view_shape,
                linear_weight):
        atb_context_attention = atb.atb_context_attention.default(query,
                                                                            key,
                                                                            value,
                                                                            k_cache,
                                                                            v_cache,
                                                                            kv_start_indices_1d,
                                                                            kv_seqlens_int,
                                                                            mask,
                                                                            num_heads,
                                                                            num_kv_heads,
                                                                            kv_head_size,
                                                                            block_size,)
        view = torch.ops.aten.view.default(atb_context_attention, view_shape)
        linear = torch.ops.atb.linear.default(view, linear_weight, None, False, True)
        return linear

    @staticmethod
    def replacement(query, key, value, k_cache, v_cache, kv_start_indices_1d, kv_seqlens_int, mask, num_heads, num_kv_heads, kv_head_size, block_size,
                view_shape,
                linear_weight):
        out = torch.ops.atb.fused_op.default(query, key, value, k_cache, v_cache, kv_start_indices_1d, kv_seqlens_int, mask, num_heads, num_kv_heads, kv_head_size, block_size,
                view_shape,
                linear_weight)
        return out

@register_torch_pattern_3
class TorchLLamaPrefill1(BackendPatternBase):
    @staticmethod
    def pattern(
        hidden_states,
        rms_norm_1_gamma, eps_1, # rms_norm
        qkv_weight,
        all_heads, # self.num_heads + self.num_kv_heads + self.num_kv_heads
        llama_head_dim, # self.head_dim
        llama_hidden_size, # self.num_heads * self.head_dim
        llama_num_heads, # self.num_heads
        llama_num_kv_heads, # self.num_kv_heads
        cos, sin,  # rope
        k_cache, v_cache,
        block_size,        # self.block_size
        attn_num_heads,    # self.num_heads
        attn_num_kv_heads, # self.num_kv_heads
        attn_v_head_size,  # self.v_head_size
        attn_hidden_size_q, # self.num_heads * self.v_head_size
        attn_hidden_size_kv, # self.num_kv_heads * self.v_head_size
        kv_start_indices_1d,
        kv_seqlens_int,
        mask,
        o_weight,
        rms_norm_2_gamma, eps_2,
        gate_up,
        down,
        rms_norm_3_gamma, eps_3,
    ):
        rms_norm = infer_ext.rms_norm.default(hidden_states, rms_norm_1_gamma, eps_1)
        linear = atb.linear.default(rms_norm, qkv_weight, None, False, True)
        view = aten.view.default(linear, [-1, all_heads, llama_head_dim])
        split = aten.split_with_sizes.default(view, [llama_num_heads, llama_num_kv_heads, llama_num_kv_heads], 1)
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        
        view_1 = aten.view.default(getitem, [-1, llama_hidden_size])
        view_2 = aten.view.default(getitem_1, [-1, llama_hidden_size])
        rope = atb.rope.default(view_1, view_2, cos, sin, kv_seqlens_int)
        getitem_3 = rope[0]
        getitem_4 = rope[1]
        
        view_3 = aten.view.default(getitem_3, [-1, llama_num_heads, llama_head_dim])
        view_4 = aten.view.default(getitem_4, [-1, llama_num_heads, llama_head_dim])
        
        view_5 = aten.view.default(k_cache, [-1, block_size, attn_num_kv_heads, attn_v_head_size])
        view_6 = aten.view.default(v_cache, [-1, block_size, attn_num_kv_heads, attn_v_head_size])
        fill_kv_cache = infer_ext.fill_kv_cache.default(view_4, getitem_2, view_5, view_6, kv_start_indices_1d)
        getitem_5 = fill_kv_cache[0]
        getitem_6 = fill_kv_cache[1]

        view_7 = aten.view.default(view_3, [-1, attn_hidden_size_q])
        view_8 = aten.view.default(view_4, [-1, attn_hidden_size_kv])
        view_9 = aten.view.default(getitem_2, [-1, attn_hidden_size_kv])
        context_attention = atb.context_attention.default(view_7, view_8, view_9, getitem_5, getitem_6, kv_seqlens_int, mask, attn_num_heads, attn_num_kv_heads)        
        
        view_10 = aten.view.default(context_attention, [1, -1, llama_hidden_size])
        linear_1 = atb.linear.default(view_10, o_weight, None, False, True)
        
        add = atb.add.default(linear_1, hidden_states)
        rms_norm_1 = infer_ext.rms_norm.default(add, rms_norm_2_gamma, eps_2)
        
        mlp_gate = atb.mlp_gate.default(rms_norm_1, gate_up, down)
        add_1 = atb.add.default(mlp_gate, add)
        rms_norm_2 = infer_ext.rms_norm.default(add_1, rms_norm_3_gamma, eps_3)
        return rms_norm_2

    @staticmethod
    def replacement(
        hidden_states,
        rms_norm_1_gamma, eps_1, # rms_norm
        qkv_weight,
        all_heads, # self.num_heads + self.num_kv_heads + self.num_kv_heads
        llama_head_dim, # self.head_dim
        llama_hidden_size, # self.num_heads * self.head_dim
        llama_num_heads, # self.num_heads
        llama_num_kv_heads, # self.num_kv_heads
        cos, sin,  # rope
        k_cache, v_cache,
        block_size,        # self.block_size
        attn_num_heads,    # self.num_heads
        attn_num_kv_heads, # self.num_kv_heads
        attn_v_head_size,  # self.v_head_size
        attn_hidden_size_q, # self.num_heads * self.v_head_size
        attn_hidden_size_kv, # self.num_kv_heads * self.v_head_size
        kv_start_indices_1d,
        kv_seqlens_int,
        mask,
        o_weight,
        rms_norm_2_gamma, eps_2,
        gate_up,
        down,
        rms_norm_3_gamma, eps_3,
    ):
        out = torch.ops.atb.llama_prefill_and_norm.default(hidden_states,
                                                            rms_norm_1_gamma,
                                                            qkv_weight,
                                                            cos,
                                                            sin,
                                                            k_cache,
                                                            v_cache,
                                                            kv_start_indices_1d,
                                                            kv_seqlens_int,
                                                            mask,
                                                            o_weight,
                                                            rms_norm_2_gamma,
                                                            gate_up,
                                                            down,
                                                            rms_norm_3_gamma,
                                                            eps_1,
                                                            llama_num_heads,
                                                            llama_num_kv_heads,
                                                            llama_head_dim,
                                                            block_size)
        return out

@register_torch_pattern_3
class TorchLLamaPrefill2(BackendPatternBase):
    @staticmethod
    def pattern(
        hidden_states,
        rms_norm_1_gamma, eps_1, # rms_norm
        qkv_weight,
        all_heads, # self.num_heads + self.num_kv_heads + self.num_kv_heads
        llama_head_dim, # self.head_dim
        llama_hidden_size, # self.num_heads * self.head_dim
        llama_num_heads, # self.num_heads
        llama_num_kv_heads, # self.num_kv_heads
        cos, sin,  # rope
        k_cache, v_cache,
        block_size,        # self.block_size
        attn_num_heads,    # self.num_heads
        attn_num_kv_heads, # self.num_kv_heads
        attn_v_head_size,  # self.v_head_size
        attn_hidden_size_q, # self.num_heads * self.v_head_size
        attn_hidden_size_kv, # self.num_kv_heads * self.v_head_size
        kv_start_indices_1d,
        kv_seqlens_int,
        mask,
        o_weight,
        rms_norm_2_gamma, eps_2,
        gate_up,
        down,
    ):
        rms_norm = infer_ext.rms_norm.default(hidden_states, rms_norm_1_gamma, eps_1)
        linear = atb.linear.default(rms_norm, qkv_weight, None, False, True)
        view = aten.view.default(linear, [-1, all_heads, llama_head_dim])
        split = aten.split_with_sizes.default(view, [llama_num_heads, llama_num_kv_heads, llama_num_kv_heads], 1)
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        
        view_1 = aten.view.default(getitem, [-1, llama_hidden_size])
        view_2 = aten.view.default(getitem_1, [-1, llama_hidden_size])
        rope = atb.rope.default(view_1, view_2, cos, sin, kv_seqlens_int)
        getitem_3 = rope[0]
        getitem_4 = rope[1]
        
        view_3 = aten.view.default(getitem_3, [-1, llama_num_heads, llama_head_dim])
        view_4 = aten.view.default(getitem_4, [-1, llama_num_heads, llama_head_dim])
        
        view_5 = aten.view.default(k_cache, [-1, block_size, attn_num_kv_heads, attn_v_head_size])
        view_6 = aten.view.default(v_cache, [-1, block_size, attn_num_kv_heads, attn_v_head_size])
        fill_kv_cache = infer_ext.fill_kv_cache.default(view_4, getitem_2, view_5, view_6, kv_start_indices_1d)
        getitem_5 = fill_kv_cache[0]
        getitem_6 = fill_kv_cache[1]

        view_7 = aten.view.default(view_3, [-1, attn_hidden_size_q])
        view_8 = aten.view.default(view_4, [-1, attn_hidden_size_kv])
        view_9 = aten.view.default(getitem_2, [-1, attn_hidden_size_kv])
        context_attention = atb.context_attention.default(view_7, view_8, view_9, getitem_5, getitem_6, kv_seqlens_int, mask, attn_num_heads, attn_num_kv_heads)        
        
        view_10 = aten.view.default(context_attention, [1, -1, llama_hidden_size])
        linear_1 = atb.linear.default(view_10, o_weight, None, False, True)
        
        add = atb.add.default(linear_1, hidden_states)
        rms_norm_1 = infer_ext.rms_norm.default(add, rms_norm_2_gamma, eps_2)
        
        mlp_gate = atb.mlp_gate.default(rms_norm_1, gate_up, down)
        add_1 = atb.add.default(mlp_gate, add)
        return add_1

    @staticmethod
    def replacement(
        hidden_states,
        rms_norm_1_gamma, eps_1, # rms_norm
        qkv_weight,
        all_heads, # self.num_heads + self.num_kv_heads + self.num_kv_heads
        llama_head_dim, # self.head_dim
        llama_hidden_size, # self.num_heads * self.head_dim
        llama_num_heads, # self.num_heads
        llama_num_kv_heads, # self.num_kv_heads
        cos, sin,  # rope
        k_cache, v_cache,
        block_size,        # self.block_size
        attn_num_heads,    # self.num_heads
        attn_num_kv_heads, # self.num_kv_heads
        attn_v_head_size,  # self.v_head_size
        attn_hidden_size_q, # self.num_heads * self.v_head_size
        attn_hidden_size_kv, # self.num_kv_heads * self.v_head_size
        kv_start_indices_1d,
        kv_seqlens_int,
        mask,
        o_weight,
        rms_norm_2_gamma, eps_2,
        gate_up,
        down,
    ):
        out = torch.ops.atb.llama_prefill.default(hidden_states,
                                                  rms_norm_1_gamma,
                                                  qkv_weight,
                                                  cos,
                                                  sin,
                                                  k_cache,
                                                  v_cache,
                                                  kv_start_indices_1d,
                                                  kv_seqlens_int,
                                                  mask,
                                                  o_weight,
                                                  rms_norm_2_gamma,
                                                  gate_up,
                                                  down,
                                                  eps_1,
                                                  llama_num_heads,
                                                  llama_num_kv_heads,
                                                  llama_head_dim,
                                                  block_size)
        return out



@register_torch_pattern_3
class TorchLLamaDecode1(BackendPatternBase):
    @staticmethod
    def pattern(
        hidden_states,
        rms_norm_1_gamma, eps_1, # rms_norm
        qkv_weight,
        all_heads, # self.num_heads + self.num_kv_heads + self.num_kv_heads
        llama_head_dim, # self.head_dim
        llama_hidden_size, # self.num_heads * self.head_dim
        llama_num_heads, # self.num_heads
        llama_num_kv_heads, # self.num_kv_heads
        cos, sin,  # rope
        k_cache, v_cache,
        block_size,        # self.block_size
        attn_num_heads,    # self.num_heads
        attn_num_kv_heads, # self.num_kv_heads
        attn_v_head_size,  # self.v_head_size
        kv_start_indices_1d,
        kv_seqlens_int,
        mask,
        block_offsets,
        o_weight,
        rms_norm_2_gamma, eps_2,
        gate_up,
        down,
        rms_norm_3_gamma, eps_3,
    ):
        rms_norm = infer_ext.rms_norm.default(hidden_states, rms_norm_1_gamma, eps_1)
        linear = atb.linear.default(rms_norm, qkv_weight, None, False, True)
        view = aten.view.default(linear, [-1, all_heads, llama_head_dim])
        split = aten.split_with_sizes.default(view, [llama_num_heads, llama_num_kv_heads, llama_num_kv_heads], 1)
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        
        view_1 = aten.view.default(getitem, [-1, llama_hidden_size])
        view_2 = aten.view.default(getitem_1, [-1, llama_hidden_size])
        rope = atb.rope.default(view_1, view_2, cos, sin, kv_seqlens_int)
        getitem_3 = rope[0]
        getitem_4 = rope[1]
        
        view_3 = aten.view.default(getitem_3, [-1, llama_num_heads, llama_head_dim])
        view_4 = aten.view.default(getitem_4, [-1, llama_num_heads, llama_head_dim])
        
        view_5 = aten.view.default(k_cache, [-1, block_size, attn_num_kv_heads, attn_v_head_size])
        view_6 = aten.view.default(v_cache, [-1, block_size, attn_num_kv_heads, attn_v_head_size])
        fill_kv_cache = infer_ext.fill_kv_cache.default(view_4, getitem_2, view_5, view_6, kv_start_indices_1d)
        getitem_5 = fill_kv_cache[0]
        getitem_6 = fill_kv_cache[1]

        paged_attention = atb.paged_attention_decode.default(view_3, getitem_5, getitem_6, block_offsets, kv_seqlens_int, mask, attn_num_heads, attn_num_kv_heads)        
        
        view_10 = aten.view.default(paged_attention, [1, -1, llama_hidden_size])
        linear_1 = atb.linear.default(view_10, o_weight, None, False, True)
        
        add = atb.add.default(linear_1, hidden_states)
        rms_norm_1 = infer_ext.rms_norm.default(add, rms_norm_2_gamma, eps_2)
        
        mlp_gate = atb.mlp_gate.default(rms_norm_1, gate_up, down)
        add_1 = atb.add.default(mlp_gate, add)
        rms_norm_2 = infer_ext.rms_norm.default(add_1, rms_norm_3_gamma, eps_3)
        return rms_norm_2

    @staticmethod
    def replacement(
        hidden_states,
        rms_norm_1_gamma, eps_1, # rms_norm
        qkv_weight,
        all_heads, # self.num_heads + self.num_kv_heads + self.num_kv_heads
        llama_head_dim, # self.head_dim
        llama_hidden_size, # self.num_heads * self.head_dim
        llama_num_heads, # self.num_heads
        llama_num_kv_heads, # self.num_kv_heads
        cos, sin,  # rope
        k_cache, v_cache,
        block_size,        # self.block_size
        attn_num_heads,    # self.num_heads
        attn_num_kv_heads, # self.num_kv_heads
        attn_v_head_size,  # self.v_head_size
        kv_start_indices_1d,
        kv_seqlens_int,
        mask,
        block_offsets,
        o_weight,
        rms_norm_2_gamma, eps_2,
        gate_up,
        down,
        rms_norm_3_gamma, eps_3,
    ):
        out = torch.ops.atb.llama_decode_and_norm.default(hidden_states,
                                                            rms_norm_1_gamma,
                                                            qkv_weight,
                                                            cos,
                                                            sin,
                                                            k_cache,
                                                            v_cache,
                                                            kv_start_indices_1d,
                                                            kv_seqlens_int,
                                                            mask,
                                                            block_offsets,
                                                            o_weight,
                                                            rms_norm_2_gamma,
                                                            gate_up,
                                                            down,
                                                            rms_norm_3_gamma,
                                                            eps_1,
                                                            llama_num_heads,
                                                            llama_num_kv_heads,
                                                            llama_head_dim,
                                                            block_size)
        return out


@register_torch_pattern_3
class TorchLLamaDecode2(BackendPatternBase):
    @staticmethod
    def pattern(
        hidden_states,
        rms_norm_1_gamma, eps_1, # rms_norm
        qkv_weight,
        all_heads, # self.num_heads + self.num_kv_heads + self.num_kv_heads
        llama_head_dim, # self.head_dim
        llama_hidden_size, # self.num_heads * self.head_dim
        llama_num_heads, # self.num_heads
        llama_num_kv_heads, # self.num_kv_heads
        cos, sin,  # rope
        k_cache, v_cache,
        block_size,        # self.block_size
        attn_num_heads,    # self.num_heads
        attn_num_kv_heads, # self.num_kv_heads
        attn_v_head_size,  # self.v_head_size
        kv_start_indices_1d,
        kv_seqlens_int,
        mask,
        block_offsets,
        o_weight,
        rms_norm_2_gamma, eps_2,
        gate_up,
        down
    ):
        rms_norm = infer_ext.rms_norm.default(hidden_states, rms_norm_1_gamma, eps_1)
        linear = atb.linear.default(rms_norm, qkv_weight, None, False, True)
        view = aten.view.default(linear, [-1, all_heads, llama_head_dim])
        split = aten.split_with_sizes.default(view, [llama_num_heads, llama_num_kv_heads, llama_num_kv_heads], 1)
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        
        view_1 = aten.view.default(getitem, [-1, llama_hidden_size])
        view_2 = aten.view.default(getitem_1, [-1, llama_hidden_size])
        rope = atb.rope.default(view_1, view_2, cos, sin, kv_seqlens_int)
        getitem_3 = rope[0]
        getitem_4 = rope[1]
        
        view_3 = aten.view.default(getitem_3, [-1, llama_num_heads, llama_head_dim])
        view_4 = aten.view.default(getitem_4, [-1, llama_num_heads, llama_head_dim])
        
        view_5 = aten.view.default(k_cache, [-1, block_size, attn_num_kv_heads, attn_v_head_size])
        view_6 = aten.view.default(v_cache, [-1, block_size, attn_num_kv_heads, attn_v_head_size])
        fill_kv_cache = infer_ext.fill_kv_cache.default(view_4, getitem_2, view_5, view_6, kv_start_indices_1d)
        getitem_5 = fill_kv_cache[0]
        getitem_6 = fill_kv_cache[1]

        paged_attention = atb.paged_attention_decode.default(view_3, getitem_5, getitem_6, block_offsets, kv_seqlens_int, mask, attn_num_heads, attn_num_kv_heads)        
        
        view_10 = aten.view.default(paged_attention, [1, -1, llama_hidden_size])
        linear_1 = atb.linear.default(view_10, o_weight, None, False, True)
        
        add = atb.add.default(linear_1, hidden_states)
        rms_norm_1 = infer_ext.rms_norm.default(add, rms_norm_2_gamma, eps_2)
        
        mlp_gate = atb.mlp_gate.default(rms_norm_1, gate_up, down)
        add_1 = atb.add.default(mlp_gate, add)
        return add_1

    @staticmethod
    def replacement(
        hidden_states,
        rms_norm_1_gamma, eps_1, # rms_norm
        qkv_weight,
        all_heads, # self.num_heads + self.num_kv_heads + self.num_kv_heads
        llama_head_dim, # self.head_dim
        llama_hidden_size, # self.num_heads * self.head_dim
        llama_num_heads, # self.num_heads
        llama_num_kv_heads, # self.num_kv_heads
        cos, sin,  # rope
        k_cache, v_cache,
        block_size,        # self.block_size
        attn_num_heads,    # self.num_heads
        attn_num_kv_heads, # self.num_kv_heads
        attn_v_head_size,  # self.v_head_size
        kv_start_indices_1d,
        kv_seqlens_int,
        mask,
        block_offsets,
        o_weight,
        rms_norm_2_gamma, eps_2,
        gate_up,
        down
    ):
        out = torch.ops.atb.llama_decode.default(hidden_states,
                                                            rms_norm_1_gamma,
                                                            qkv_weight,
                                                            cos,
                                                            sin,
                                                            k_cache,
                                                            v_cache,
                                                            kv_start_indices_1d,
                                                            kv_seqlens_int,
                                                            mask,
                                                            block_offsets,
                                                            o_weight,
                                                            rms_norm_2_gamma,
                                                            gate_up,
                                                            down,
                                                            eps_1,
                                                            llama_num_heads,
                                                            llama_num_kv_heads,
                                                            llama_head_dim,
                                                            block_size)
        return out



# @register_torch_pattern_3
# class TorchLLamaPrefill3(BackendPatternBase):
#     @staticmethod
#     def pattern(
#         hidden_states,
#         rms_norm_1_gamma, eps_1,
#         qkv_weight,
#         all_heads,
#         llama_head_dim, # self.head_dim
#         llama_num_heads, # self.num_heads
#         llama_hidden_size,
#         llama_num_kv_heads, # self.num_kv_heads
#         cos, sin,  # rope
#         k_cache, v_cache,
#         block_size,        # self.block_size
#         attn_num_heads,    # self.num_heads
#         attn_num_kv_heads, # self.num_kv_heads
#         attn_v_head_size,  # self.v_head_size
#         attn_hidden_size_q, # self.num_heads * self.v_head_size
#         attn_hidden_size_kv, # self.num_kv_heads * self.v_head_size
#         kv_start_indices_1d,
#         kv_seqlens_int,
#         mask,
#         o_weight,
#         rms_norm_2_gamma, eps_2,
#         gate_up,
#         down,
#     ):
#         # residual2 = atb.add.default(hidden_states, residual)
#         rms_norm = infer_ext.rms_norm.default(hidden_states, rms_norm_1_gamma, eps_1)
#         linear = atb.linear.default(rms_norm, qkv_weight, None, False, True)
#         view = aten.view.default(linear, [-1, all_heads, llama_head_dim])
#         split = aten.split_with_sizes.default(view, [llama_num_heads, llama_num_kv_heads, llama_num_kv_heads], 1)
#         getitem = split[0]
#         getitem_1 = split[1]
#         getitem_2 = split[2]
        
#         view_1 = aten.view.default(getitem, [-1, llama_hidden_size])
#         view_2 = aten.view.default(getitem_1, [-1, llama_hidden_size])
#         rope = atb.rope.default(view_1, view_2, cos, sin, kv_seqlens_int)
#         getitem_3 = rope[0]
#         getitem_4 = rope[1]

#         view_3 = aten.view.default(getitem_3, [-1, llama_num_heads, llama_head_dim])
#         view_4 = aten.view.default(getitem_4, [-1, llama_num_heads, llama_head_dim])
        
#         view_5 = aten.view.default(k_cache, [-1, block_size, attn_num_kv_heads, attn_v_head_size])
#         view_6 = aten.view.default(v_cache, [-1, block_size, attn_num_kv_heads, attn_v_head_size])
#         fill_kv_cache = infer_ext.fill_kv_cache.default(view_4, getitem_2, view_5, view_6, kv_start_indices_1d)
#         getitem_5 = fill_kv_cache[0]
#         getitem_6 = fill_kv_cache[1]

#         view_7 = aten.view.default(view_3, [-1, attn_hidden_size_q])
#         view_8 = aten.view.default(view_4, [-1, attn_hidden_size_kv])
#         view_9 = aten.view.default(getitem_2, [-1, attn_hidden_size_kv])
#         context_attention = atb.context_attention.default(view_7, view_8, view_9, getitem_5, getitem_6, kv_seqlens_int, mask, attn_num_heads, attn_num_kv_heads)     

#         view_10 = aten.view.default(context_attention, [1, -1, llama_hidden_size])
#         linear_1 = atb.linear.default(view_10, o_weight, None, False, True)
        
#         add = atb.add.default(linear_1, hidden_states)
#         rms_norm_1 = infer_ext.rms_norm.default(add, rms_norm_2_gamma, eps_2)
        
#         mlp_gate = atb.mlp_gate.default(rms_norm_1, gate_up, down)
#         add_1 = atb.add.default(mlp_gate, add)
#         return add_1

#     @staticmethod
#     def replacement(
#         hidden_states,
#         rms_norm_1_gamma, eps_1,
#         qkv_weight,
#         all_heads,
#         llama_head_dim, # self.head_dim
#         llama_num_heads, # self.num_heads
#         llama_hidden_size,
#         llama_num_kv_heads, # self.num_kv_heads
#         cos, sin,  # rope
#         k_cache, v_cache,
#         block_size,        # self.block_size
#         attn_num_heads,    # self.num_heads
#         attn_num_kv_heads, # self.num_kv_heads
#         attn_v_head_size,  # self.v_head_size
#         attn_hidden_size_q, # self.num_heads * self.v_head_size
#         attn_hidden_size_kv, # self.num_kv_heads * self.v_head_size
#         kv_start_indices_1d,
#         kv_seqlens_int,
#         mask,
#         o_weight,
#         rms_norm_2_gamma, eps_2,
#         gate_up,
#         down,
#     ):
#         add = torch.ops.atb.debug.default(
#                                           hidden_states,
#                                           None,
#                                           rms_norm_1_gamma,
#                                           qkv_weight,
#                                           cos,
#                                           sin,
#                                           k_cache,
#                                           v_cache,
#                                           kv_start_indices_1d,
#                                           kv_seqlens_int,
#                                           mask,
#                                           o_weight,
#                                           rms_norm_2_gamma,
#                                           gate_up,
#                                           down,
#                                           eps_2,
#                                           attn_num_heads,
#                                           attn_num_kv_heads,
#                                           llama_head_dim,
#                                           block_size)
#         return add
