# GPU bisection solver

This module contains the CUDA node solver, procedural-field bridge, and source-only
review harness for BinocMesher. It does not automatically replace the main meshing
pipeline. Callers provide CSR nodes, original field parameters, and output buffers.

The implementation supports GPU-resident rounds (`R0`), resident rounds replayed
with CUDA Graphs (`R1`), one-thread-per-node fusion (`L0`), and cooperative node
fusion (`L1_8`, `L1_32`). `BM_L132_BLOCK_THREADS` selects 256, 128, or 64 threads per
block for `L1_32`; each node continues to use exactly 32 threads. The default is 256.
All variants keep the same arithmetic, precision, field functions and output
contract. No performance winner is assumed by the source defaults.

- [Build instructions](BUILD.md): existing CUDA installation, explicit upstream
  include path, architecture, host compiler, and external output directory.
- [C ABI and lifecycle](API.md): public declarations in `include/solver_api.h` and
  `src/field_types.h`; Python bindings are also in `src/check_gpu_solver.py`.
- [Review harness](HARNESS.md): natural-input correctness, native output adaptation,
  finite paired performance comparisons, and separate Nsight Compute collection.

Build from the repository root, after making the existing Infinigen submodule
headers available:

```bash
python -B gpu_accelerate/build.py \
  --infinigen-include "$PWD/infinigen_binocmesher/infinigen" \
  --output /path/outside/repository/cuda-build \
  --blocks 256 128 64
```

Use `--host-compiler` and `--nvcc` when the CUDA installation is not on PATH.
`--debug` emits solver libraries with allocation guards, without enabling CUDA
`-G` or changing the arithmetic. Linux/WSL with a GCC-compatible host compiler is
the supported build path. The architecture defaults to `sm_120` and is configurable.

The four CUDA/header sources are preserved from the tested implementation. The
build entry and workspace/tool paths are portable packaging changes. Earlier local
experiment conclusions are not bundled with this source publication.

No scene parameters, native captures, checkpoint files, measured timing tables,
profiler reports, compiled binaries, environments, or copied third-party trees are
included. Natural-field validation therefore requires separately supplied inputs;
analytic diagnostic fixtures cannot establish real procedural-field coverage.
The included field headers reference the existing Infinigen dependency rather than
vendoring it. Upstream license notices and dependency licensing remain applicable.
