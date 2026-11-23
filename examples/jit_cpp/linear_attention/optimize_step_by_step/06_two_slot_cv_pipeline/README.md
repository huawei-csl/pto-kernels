# Step 06: Two-Slot Cube-Vector Pipeline

This step corresponds to commit `3350511`.

What changed:
- the per-core workspaces are doubled to two slots
- cube can prepare chunk `i + 1` while vector finishes chunk `i`
- explicit cross-core handshakes keep the staged buffers safe when `B * H` exceeds the core count
