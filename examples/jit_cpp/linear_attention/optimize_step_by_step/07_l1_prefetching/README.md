# Step 07: L1 Prefetching

This step corresponds to commit `26aac37`.

What changed:
- a second `H`-state L1 tile is kept on the cube side
- the next accumulated hidden-state tile is prefetched while the current output path is already loading `Q`, `V`, and masked attention

This is the current best educational endpoint in the optimization ladder.

Important benchmarking note:
- the small `--quick` benchmark only uses `(8/16, 20, 1024, 128, 128)` shapes, so it typically reports around the mid-`60 TFLOP/s` range
- the kernel in this directory is intentionally kept identical to the current main example, so the full benchmark table reaches the same large-shape performance class

To reproduce the main-example style result, run:

```bash
python benchmark_linear_attention.py --warmup 2 --repeats 5
```

On this machine, that full-table run validated:
- `77.71 TFLOP/s` / `565.43 GiB/s` at `(12, 20, 8192, 128, 128)`

For comparison, the current main example measured immediately afterwards with the same command reached:
- `77.97 TFLOP/s` / `567.34 GiB/s` at `(24, 20, 6144, 128, 128)`

That small difference is normal run-to-run noise. The important point is that this final tutorial step reaches the same `~78 TFLOP/s` performance class as the main example when both are benchmarked with the full table.

Run those large-shape benchmarks one process at a time so the NPU is not oversubscribed by multiple concurrent benchmark jobs.
