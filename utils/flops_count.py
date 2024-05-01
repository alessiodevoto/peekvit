import numpy as np
import torch
from ptflops import get_model_complexity_info


"""
Experimental code for counting flops in a model.
We use the ptflops library to count flops, see https://github.com/sovrasov/flops-counter.pytorch/tree/master.
The library defines hooks that can be registered to a model to count flops.
We have to define custom hooks for residual layers, since the library does not support them. 
"""


def count_masked_tokens(batch: torch.Tensor, per_sequence: bool = False):
    if not batch.dim() == 3 or batch.shape[1] == 1:
        return torch.tensor(0)
    # TODO: we have class tokens, registers and possibly budget tokens, which are never masked out
    # here we assume that the first and last two tokens are special tokens, so we exclude them from the count
    # in the future we should make this more general and pass the indices of the tokens to exclude
    # batch = batch[:, 2:-2, :]
    masked = torch.sum(batch, dim=-1) == 0
    return masked.sum(-1) if per_sequence else masked.sum()


# this is the hook for a standard linear layer:
# https://github.com/sovrasov/flops-counter.pytorch/blob/6a00a4fa13053f2891a7ce00405142f4ec201fbc/ptflops/pytorch_ops.py#L33
# the difference is that we should exclude from the sequence the tokens that are masked out, i.e. all zeros
def res_linear_flops_counter_hook(module, input, output):
    input = input[0]
    output_last_dim = output.shape[-1]
    input_last_dim = input.shape[-1]
    pre_last_dims_prod = np.prod(
        input.shape[0:-1], dtype=np.int64
    ) - count_masked_tokens(input, per_sequence=False)
    bias_flops = output_last_dim if module.bias is not None else 0
    module.__flops__ += int(
        torch.sum((input_last_dim * output_last_dim + bias_flops) * pre_last_dims_prod)
    )

    if not hasattr(module, "avg_sparsity"):
        module.avg_sparsity = torch.tensor(
            0.0
        )  # count_masked_tokens(input, per_sequence=False)
    else:
        masked_tokens = count_masked_tokens(input, per_sequence=False)
        module.avg_sparsity = module.avg_sparsity + masked_tokens.to(
            module.avg_sparsity.device
        )


# this is standard the hook for a standard multihead attention layer:
# https://github.com/sovrasov/flops-counter.pytorch/blob/6a00a4fa13053f2891a7ce00405142f4ec201fbc/ptflops/pytorch_ops.py#L174
# the difference is that we should exclude from the sequence the tokens that are masked out, i.e. all zeros
def res_multihead_attention_counter_hook(multihead_attention_module, input, output):
    flops = 0

    q, k, v = input

    batch_first = (
        multihead_attention_module.batch_first
        if hasattr(multihead_attention_module, "batch_first")
        else False
    )
    if batch_first:
        len_idx = 1
    else:
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    # we change len to exclude the tokens that are masked out
    qlen = q.shape[len_idx] - count_masked_tokens(q, per_sequence=True)
    klen = k.shape[len_idx] - count_masked_tokens(k, per_sequence=True)
    vlen = v.shape[len_idx] - count_masked_tokens(v, per_sequence=True)

    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.embed_dim

    if multihead_attention_module.kdim is None:
        assert kdim == qdim
    if multihead_attention_module.vdim is None:
        assert vdim == qdim

    # Q scaling
    flops += qlen * qdim

    # Initial projections
    flops += (
        (qlen * qdim * qdim)  # QW
        + (klen * kdim * kdim)  # KW
        + (vlen * vdim * vdim)  # VW
    )

    if multihead_attention_module.in_proj_bias is not None:
        flops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_flops = (
        (qlen * klen * qk_head_dim)  # QK^T
        + (qlen * klen)  # softmax
        + (qlen * klen * v_head_dim)  # AV
    )

    flops += num_heads * head_flops

    # final projection, bias is always enableds
    flops += qlen * vdim * (vdim + 1)

    # flops *= batch_size
    multihead_attention_module.__flops__ += int(flops.sum())

    if not hasattr(multihead_attention_module, "avg_sparsity"):
        multihead_attention_module.avg_sparsity = torch.tensor(
            0.0
        )  # count_masked_tokens(q, per_sequence=False)
    else:
        num_masked_tokens = count_masked_tokens(q, per_sequence=False)
        multihead_attention_module.avg_sparsity = (
            multihead_attention_module.avg_sparsity
            + num_masked_tokens.to(multihead_attention_module.avg_sparsity.device)
        )


def compute_flops(
    model,
    input_batch,
    custom_modules_hooks: dict = {
        torch.nn.MultiheadAttention: res_multihead_attention_counter_hook,
        torch.nn.Linear: res_linear_flops_counter_hook,
    },
    as_strings=True,
    print_per_layer_stat=True,
    verbose=True,
    flops_units="Mac",
    output_file=None,
):

    input_shape = tuple(input_batch.shape)
    macs, params = get_model_complexity_info(
        model,
        input_shape,
        ost=output_file,
        input_constructor=lambda _: input_batch,
        as_strings=as_strings,
        print_per_layer_stat=print_per_layer_stat,
        verbose=verbose,
        custom_modules_hooks=custom_modules_hooks,
        flops_units=flops_units,
    )

    # flops = 2 * macs
    if macs is None:
        print(
            "Something went wrong with the flops count: macs is None. Returning zero flops"
        )
        return torch.tensor(0.0), params
    else:
        return macs * 2, params


# usage
"""macs, params = get_model_complexity_info(
    resvit, (1,3,224,224), 
    input_constructor = lambda x : torch.randn(x),
    as_strings=True,
    print_per_layer_stat=True, 
    verbose=True, 
    custom_modules_hooks={torch.nn.MultiheadAttention: res_multihead_attention_counter_hook, torch.nn.Linear: res_linear_flops_counter_hook},
    flops_units='MMac'
    )"""


####################################################### old ##################################################################
