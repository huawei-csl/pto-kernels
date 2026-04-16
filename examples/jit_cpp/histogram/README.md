# Ascend PTO Histogram Implementation Examples

This directory contains a series of implementations demonstrating the evolution and optimization of a histogram kernel using the PTO library for Ascend NPUs (specifically targeting the A2/910B architecture).

## Implementation Evolution

The implementations are organized into steps, each introducing new concepts or optimizations:

- **Step 0: Count Less Than (`step0_count_less_than`)**: The foundational algorithm that counts how many elements in a tile are less than a given pivot value using vector comparisons and reductions. The algorithm is implemented using atomic operations and using a two phase kernels. The atomic operations didn't always behave as expected, see note below.
- **Step 1: Naive Histogram (`step1_naive_histogram`)**: Expands the logic to a full histogram by looping over all bins. For each bin, it calculates the count of elements falling within that range.
- **Step 2: Double Buffering (`step2_double_buffering`)**: Introduces double buffering (ping-pong) for data loading from Global Memory (GM) to Unified Buffer (UB), allowing computation and data movement to overlap.
- **Step 3: Scatter Index to GM (`step3_scatter_index_to_gm`)**: A significant algorithmic shift. Instead of looping over bins and performing scalar updates, this implementation calculates the bin index for each element in parallel using vector operations and uses `MSCATTER` with `AtomicAdd` to update the histogram directly in Global Memory. This avoids the slow scalar-to-vector synchronization.

## Included Files

- `bench_kernels.py`: A comprehensive benchmarking suite to compare the performance of different implementations.
- `plot_kernels.py`: A utility script to visualize the benchmarking results.
- `run_histogram.py`: A script for functional testing and verification of the kernels.
- `jit_util_histogram.py`: Utility functions for JIT-compiling the C++ kernels into Python-callable operators.
- `kernel_utils.h`: Common helper functions used across the different kernel implementations.

## Usage

### Testing
To verify the correctness of the implementations, run:
```bash
python run_histogram.py
```
*Note: You can modify the parameters (such as `num_bins`, `total_length`, `tile_size` or which implementation to use) directly inside the script.*

### Benchmarking
To compare the performance of the different steps:
```bash
python bench_kernels.py
```
The script supports various arguments for configuring the benchmark range and parameters. Use `--help` to see all options.

### Plotting
After running the benchmarks, you can generate plots using:
```bash
python plot_kernels.py
```

## Torch Operator Integration

A production-ready Torch operator implementation is provided in the repository's core source tree:
- **Kernel Implementation**: `csrc/kernel/kernel_histogram.cpp`
- **Host/C++ Wrapper**: `csrc/host/torch_histogram.h`

## Known Issues and Observations

During development and optimization, several architectural and library-specific details were noted:

- **`TCMPS` Ambiguity**: The documentation is ambiguous regarding whether the valid column number of the mask tile must exactly match the source tile or if it can be different. The bits are packed, the example shows allocation of a tile that is using reduced number of valid rows, however this is in contrast with the description of the operation and in fact doesn't work.
- **Temporary Tiles**: Several instructions such as `TSEL`, `TXOR`, `TGATHER` and others mention an extra temporary tile parameter that is not actually required. Operations that do require it (like `TROWSUM`) don't mention any constraints that this tile should have.
- **`AtomicAdd` in `TSTORE`**: The `AtomicAdd` parameter in the `TSTORE` instruction was found to be unreliable in some configurations, requiring the use of a two-phase algorithm (local reduction followed by a final global reduction) for stability.
- **`MSCATTER` on A2/A3**: The `MSCATTER` implementation is not supported on A2/A3 architectures, thus at the time of writing Step 3 is provided only as an illustration of the next direction of implementation.
- Occasionally the two-phase algorithm will not launch the first or second phase with no error provided. Whether this is an implementation issue or a driver/toolkit issue is still under investigation.
