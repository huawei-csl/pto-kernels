#!/bin/bash

# Unit test of tile-lang on test cases from https://github.com/fla-org/flash-linear-attention/blob/main/tests/ops/test_gated_delta.py#L89-L100

# Example given by tilelang-ascend
python chunk_gated_delta_rule_varlen.py --B 1 --T 204 --H 8 --Hg 4 --K 128 --V 128
python chunk_gated_delta_rule_varlen.py --T 204 --H 8 --Hg 4 --K 128 --V 128 --varlen true

# non-GVA (HV == H)
# (B, T, H, HV, D, scale, gate_logit_norm, mask_p, use_qk_l2norm, dtype)
# (2, 75, 4, 4, 64, 1, 0.01, 0, False, torch.float16),
# (2, 500, 3, 3, 60, 1, 1, 0, False, torch.float16),
# (2, 1000, 3, 3, 64, 0.1, 1, 0.5, False, torch.float16),
# (3, 1024, 4, 4, 100, 1, 0.1, 0, False, torch.float16),
# (4, 1024, 4, 4, 128, 0.1, 1, 0, True, torch.float16),
# (2, 1500, 4, 4, 128, 0.1, 10, 0, False, torch.float16),
# (4, 2048, 8, 8, 64, 0.1, 1, 0, False, torch.float16),

python chunk_gated_delta_rule_varlen.py --B 2 --T 75 --H 4 --Hg 4 --K 64 --V 64 #PASS
# python chunk_gated_delta_rule_varlen.py --B 2 --T 500 --H 3 --Hg 3 --K 60 --V 60 # error: static assertion failed due to requirement '(Loc == TileType::Vec) || (1024 == TileConfig::fractalMxSize) || (60 == 1) || (Rows % InnerRows == 0)': Layout rows must be divisible by inner box row
python chunk_gated_delta_rule_varlen.py --B 2 --T 1000 --H 3 --Hg 3 --K 64 --V 64 # PASS
# python chunk_gated_delta_rule_varlen.py --B 3 --T 1024 --H 4 --Hg 4 --K 100 --V 100 # FAIL: error: static assertion failed due to requirement '(Loc == TileType::Vec) || (1024 == TileConfig::fractalMxSize) || (100 == 1) || (Rows % InnerRows == 0)': Layout rows must be divisible by inner box rows
# python chunk_gated_delta_rule_varlen.py --B 4 --T 1024 --H 4 --Hg 4 --K 128 --V 128 # PASS
python chunk_gated_delta_rule_varlen.py --B 2 --T 1500 --H 4 --Hg 4 --K 128 --V 128 # FAIL(accuracy): Mismatched elements: 1295770 / 3145728 (41.2%)
python chunk_gated_delta_rule_varlen.py --B 4 --T 2048 --H 8 --Hg 8 --K 64 --V 64

################
# GVA (HV > H) #
################
# (B, T, H, HV, D, scale, gate_logit_norm, mask_p, use_qk_l2norm, dtype)
# (2, 256, 2, 4, 64, 1, 1, 0, False, torch.float16),
# (2, 512, 2, 8, 64, 1, 0.1, 0, True, torch.float16),
# (2, 1024, 4, 8, 128, 0.1, 1, 0, False, torch.float16),

# Qwen3.6-27B shape https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json#L88-L91
python chunk_gated_delta_rule_varlen.py --varlen true --seqlens 7,32,159,256,50 --H 48 --Hg 16 --K 128 --V 128 # PASS -- dumps to `chunk_gated_delta_rule_varlen_H48.cpp`
python chunk_gated_delta_rule_varlen.py --varlen true --seqlens 512,512 --H 48 --Hg 16 --K 128 --V 128 #  1.8% mismatch, due to accumulating error by too many steps?
python chunk_gated_delta_rule_varlen.py --varlen true --seqlens 2048,2048 --H 48 --Hg 16 --K 128 --V 128 # 27.2% mismatch

# Qwen3.5-9B shape https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json#L54-L57
python chunk_gated_delta_rule_varlen.py --varlen true --seqlens 7,32,159,256,50 --H 32 --Hg 16 --K 128 --V 128 # PASS -- dumps to `chunk_gated_delta_rule_varlen_H32.cpp`
python chunk_gated_delta_rule_varlen.py --varlen true --seqlens 512,512 --H 32 --Hg 16 --K 128 --V 128 # 1.6% mismatch, due to accumulating error by too many steps?
python chunk_gated_delta_rule_varlen.py --varlen true --seqlens 1024,1024 --H 32 --Hg 16 --K 128 --V 128  # 1.8 mismatch
