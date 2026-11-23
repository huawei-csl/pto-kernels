# Step 04: Increase Chunk Size To 128

This step corresponds to commit `bd954f9`.

What changed:
- the kernel is reworked to fit `C=128, D=128` within the validated on-chip memory budget
- arithmetic intensity improves, which moves the kernel from the `~30 TFLOP/s` class toward the `~50 TFLOP/s` class
