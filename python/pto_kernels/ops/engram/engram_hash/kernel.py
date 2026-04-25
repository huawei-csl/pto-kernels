from ptodsl import pto, to_ir_module
from ptodsl import scalar as s

from pto_kernels.ops.tilekernels_common import (
    iadd,
    imul,
    int64_type,
    irem,
    ixor,
    load_scalar,
    ptr,
    sext,
    store_scalar,
    trunc,
)


const = s.const


def _meta_data():
    i64 = int64_type()
    return {
        "ptr_i32": ptr(pto.int32),
        "ptr_i64": ptr(i64),
        "i32": pto.int32,
        "i64": i64,
    }


def build_engram_hash(
    max_ngram_size: int = 3,
    num_ngram_layers: int = 2,
    num_embed_table_per_ngram: int = 8,
):
    """Build scalar/token-parallel Engram n-gram hash index generation."""
    if max_ngram_size < 2:
        raise ValueError("max_ngram_size must be at least 2")
    if num_ngram_layers <= 0:
        raise ValueError("num_ngram_layers must be positive")
    if num_embed_table_per_ngram <= 0:
        raise ValueError("num_embed_table_per_ngram must be positive")

    num_out_cols = (max_ngram_size - 1) * num_embed_table_per_ngram

    def tilekernels_engram_hash_kernel(
        ngram_token_ids_ptr: "ptr_i32",
        multipliers_ptr: "ptr_i64",
        vocab_sizes_ptr: "ptr_i32",
        offsets_ptr: "ptr_i32",
        output_ptr: "ptr_i32",
        num_tokens_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_layers = const(num_ngram_layers)
        i64 = int64_type()
        num_tokens = s.index_cast(num_tokens_i32)
        total_rows = num_tokens * c_layers
        zero_i64 = s.index_cast(c0, i64)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(total_rows, nblocks)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, total_rows)

            for work in pto.range(row_start, row_end, c1):
                layer = work // num_tokens
                token = work % num_tokens
                hash_value = zero_i64
                for ngram_idx in range(max_ngram_size):
                    token_i32 = load_scalar(
                        pto.int32,
                        ngram_token_ids_ptr,
                        token * const(max_ngram_size) + const(ngram_idx),
                    )
                    multiplier = load_scalar(
                        i64,
                        multipliers_ptr,
                        layer * const(max_ngram_size) + const(ngram_idx),
                    )
                    token_i64 = sext(token_i32, i64)
                    hash_value = ixor(hash_value, imul(token_i64, multiplier))

                    if ngram_idx > 0:
                        ngram_offset = const(
                            (ngram_idx - 1) * num_embed_table_per_ngram
                        )
                        for table_idx in range(num_embed_table_per_ngram):
                            col = ngram_offset + const(table_idx)
                            vocab_i32 = load_scalar(
                                pto.int32,
                                vocab_sizes_ptr,
                                layer
                                * const(
                                    (max_ngram_size - 1)
                                    * num_embed_table_per_ngram
                                )
                                + col,
                            )
                            offset_i32 = load_scalar(
                                pto.int32,
                                offsets_ptr,
                                layer * const(num_out_cols) + col,
                            )
                            hash_mod_i64 = irem(hash_value, sext(vocab_i32, i64))
                            out_i32 = iadd(trunc(hash_mod_i64, pto.int32), offset_i32)
                            store_scalar(
                                output_ptr,
                                layer * (num_tokens * const(num_out_cols))
                                + token * const(num_out_cols)
                                + col,
                                out_i32,
                            )

    tilekernels_engram_hash_kernel.__name__ = (
        "tilekernels_engram_hash_"
        f"n{max_ngram_size}_l{num_ngram_layers}_t{num_embed_table_per_ngram}"
    )
    return to_ir_module(meta_data=_meta_data)(tilekernels_engram_hash_kernel)
