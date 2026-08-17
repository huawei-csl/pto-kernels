# mxfp4_quant_a5 - MXFP4 block quantization on Ascend A5

bf16 -> 4-bit E2M1 nibbles plus one E8M0 scale per 32 elements, on the Ascend 950 / A5 (`dav-c310`) vector core, JIT-compiled with `bisheng` and loaded via `ctypes`. `K` is a template parameter over 26 widths; one `.so` holds an instantiation per width and the launcher dispatches on it, so there is no rebuild per size.

## Ours is faster than `torch_npu` at every supported width

![bf16 to MXFP4 bandwidth on Ascend A5, ours against torch_npu](https://raw.githubusercontent.com/Mocchibird/pto-kernels-plots/main/mxfp4_quant_a5/mxfp4_bandwidth_k.png)

| K | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| ours (GB/s) | **2759** | **3278** | **3278** | **3268** | **3232** | **3274** |
| `torch_npu` (GB/s) | 1393 | 2770 | 2756 | 3102 | 3015 | 3128 |
| ratio | **1.98x** | **1.18x** | **1.19x** | **1.05x** | **1.07x** | **1.05x** |

Between **1.05x** and **1.98x**, at batch 65,536, and the
output is **bit-identical** to the vendor op at every shape. From K=256 up it settles
at **1.05x-1.19x**; the 1.98x at K=128 is the
widest gap because a launch that small is dominated by per-call cost, where
`torch_npu` must allocate its two outputs and we are handed ours.

Bandwidth counts every byte the operation moves: `2K` read plus `K/2 + K/32` written,
2.53125 B/element, the same formula for both arms. Figures are steady-state
throughput -- 40 launches per wall-clock bracket, 9 brackets, median of 3 sweeps --
not single-launch latency.

> **Against PTO's own quantizer.** `benchmark.py` also builds this source a second
> time with `-DMXFP4_TQUANT`, swapping our four compute passes for PTO 9.1.0's
> `TQuant_MXFP4_E2M1` tile op and leaving tiling, buffering and every
> `TLOAD`/`TSTORE` identical. On that matched launch ours is **on par or a little
> ahead at every width**, with bit-identical output. Measured separately on
> CANN 9.1.0-beta.3; the CSVs are in the plots repo.

## Reproducing

On a real A5 with a CANN toolkit sourced. The default width list is 64-2048, so pass
the widths above explicitly to sweep the same shapes:

```bash
./run_benchmark.sh --axis k --ks 128,256,512,1024,2048,4096 --tag 1
```

That writes `build/pairs_k_1.csv`. The figure above was measured in an earlier run
whose CSV ships beside it in the plots repo, so the numbers there are checkable
directly; a fresh sweep on a different toolkit or part will not reproduce them
row-for-row, and rows are only comparable within one measurement.

Each arm is gated bit-exact against `torch_npu` before it is timed, so a wrong kernel cannot report a fast number. PTO gained its MXFP4 quantizer in 9.1.0, so on 9.0.0 the `TQuant` arm is skipped with a message and the `torch_npu` comparison still runs; PTO 9.1.0 shipped two `TQuant_MXFP4_E2M1_Impl` signatures and `benchmark.py` compiles the variant both ways, keeping whichever the local headers accept.
