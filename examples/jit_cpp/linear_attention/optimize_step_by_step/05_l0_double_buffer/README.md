# Step 05: L0 Double Buffer

This step corresponds to commit `7b811b0`.

What changed:
- the cube-side matmul helper splits `K=128` into `2 x 64` phases
- L0A/L0B ping-pong buffering hides part of the extract latency behind cube compute
