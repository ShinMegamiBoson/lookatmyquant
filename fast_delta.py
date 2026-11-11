"""Qwen3.5's linear-attention layers without their 63-step loop. transformers' torch_chunk_gated_delta_rule (what runs when
the fla package is not installed) inverts each 64-token chunk's unit lower-triangular matrix I − A one row at a time: 63
steps of a few small calls each, whatever the batch, in 24 of the 4B's 32 layers. On a B200 that makes a training pass
take about 0.8 s whether it holds 4 windows or 8 (finetune.py --profile). Here the same inverse is one
triangular solve; everything else is transformers' function as it stands, so the two agree to rounding
(experiments in the docstring of install()).

    import fast_delta; fast_delta.install()   # before the model runs; only the chunk rule (training and scoring whole
                                              # windows) changes, not generation's token-by-token rule
"""

import torch
import torch.nn.functional as F
import transformers.models.qwen3_5.modeling_qwen3_5 as M

original = M.torch_chunk_gated_delta_rule


def chunk_rule(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query, key = M.l2norm(query, dim=-1, eps=1e-6), M.l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, g = F.pad(beta, (0, pad_size)), F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta, k_beta = value * beta.unsqueeze(-1), key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    A = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)  # strictly lower-triangular
    # the loop this replaces leaves T with T = A + A T below the diagonal, and then adds I: (I − A)⁻¹
    eye = torch.eye(chunk_size, dtype=A.dtype, device=A.device)
    attn = torch.linalg.solve_triangular(eye - A, eye.expand_as(A), upper=False, unitriangular=True)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
                            if initial_state is None else initial_state.to(value))
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_new = v_i - k_cumdecay[:, :, i] @ last_recurrent_state
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (last_recurrent_state * g[:, :, i, -1, None, None].exp()
                                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new)
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])[:, :, :sequence_length]
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype), last_recurrent_state


def install():
    """From here on the model's linear-attention layers use chunk_rule for whole windows."""
    M.torch_chunk_gated_delta_rule = chunk_rule
