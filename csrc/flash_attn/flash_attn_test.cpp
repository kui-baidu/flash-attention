#include <iostream>
#include "flash_attn.h"

int main() {
    flash_attn_fwd(
            nullptr, // const void *q,              // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
            nullptr, // const void *k,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            nullptr, // const void *v,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            nullptr, // void *out,                  // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            nullptr, // const void *cu_seqlens_q,   // int32, batch_size+1, starting offset of each sequence
            nullptr, // const void *cu_seqlens_k,   // int32, batch_size+1, starting offset of each sequence
            0, // const int total_q,
            0, // const int total_k,
            0, // const int batch_size,
            0, // const int num_heads,
            0, // const int head_size,
            0, // const int max_seqlen_q_,
            0, // const int max_seqlen_k_,
            0.0f, // const float p_dropout,
            0.0f, // const float softmax_scale,
            true, // const bool zero_tensors,
            true, // const bool is_causal,
            true, // const bool is_bf16,
            0, // const int num_splits,        // SMs per attention matrix, can be 1
            nullptr, // void *softmax_lse_ptr,       // softmax log_sum_exp
            nullptr, // void *softmax_ptr,
            nullptr, // cudaStream_t stream,
            0 // int seed // TODO
    );

    std::cout << "done" << std::endl;

    return 0;
}
