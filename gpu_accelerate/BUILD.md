Build the CUDA field bridge and solver shared libraries with Python 3.10+ and an existing CUDA toolkit. Actual compilation targets Linux/WSL with a GCC-compatible host compiler. `--help` and `--dry-run` also work without a configured CUDA environment. The builder uses only the Python standard library and installs nothing.

Supply the **inner** Infinigen directory: it must contain `terrain/source/common/` and `infinigen_gpl/bnodes/`. Preserve the upstream revision used by your application; this repository does not vendor those field implementations. CUDA must support the requested architecture and host compiler. The default architecture is `sm_120`.

From the repository root, first inspect the commands:

```bash
python3 gpu_accelerate/build.py \
  --infinigen-include /path/to/infinigen/infinigen \
  --output ../gpu-build-release \
  --dry-run
```

`--dry-run` prints JSON and makes no filesystem writes or compiler calls. Missing source/header paths appear in `missing_inputs`; the plan does not prove that the compiler can build them.

Build all three fixed block variants:

```bash
python3 gpu_accelerate/build.py \
  --nvcc /path/to/cuda/bin/nvcc \
  --host-compiler /path/to/g++ \
  --infinigen-include /path/to/infinigen/infinigen \
  --output ../gpu-build-release
```

`--nvcc` defaults to `nvcc` on `PATH`; omit `--host-compiler` to let nvcc select its default host compiler. Caller-supplied relative paths are resolved from the current working directory. Package source paths are resolved from `build.py`, so invoking the script from another directory works.

The output must be a new or empty directory outside the checkout. A default build compiles the bridge once and produces:

- `libfield_bridge.so`
- `libgpu_solver_b256.so`
- `libgpu_solver_b128.so`
- `libgpu_solver_b64.so`

Use `--blocks 128 64` to build a subset or `--arch sm_120` to select an architecture. Every solver keeps 32 threads per node; the block macro changes only the L1_32 launch configuration. Other solver modes retain their source-defined launch configuration.

For device guards, use a separate output directory and add `--debug`. This produces `libgpu_solver_b256_debug.so`, `libgpu_solver_b128_debug.so`, and `libgpu_solver_b64_debug.so` for the default block list, plus the same ordinary field bridge. These guard builds retain `-O3`; they do not enable nvcc `-G`, and their timing API is intentionally unavailable.

Both build types retain the original strict floating-point options:

```text
-O3 -std=c++17 --fmad=false --ftz=false
--prec-div=true --prec-sqrt=true
-Xcompiler=-fPIC,-fno-fast-math,-ffp-contract=off -Xptxas=-v
```

The compiler runs with the output directory as its working directory. Its temporary directory, command records, stdout/stderr logs, `BUILD_PLAN.json`, `COMMANDS.jsonl`, and `STATUS.json` all stay under that output directory. Environment variables that inject extra nvcc flags are removed only from the child compiler environment. A failed compilation stops the build and preserves its evidence; no automatic retry or driver modification occurs.

This is a **source build**, not application integration or a new GPU validation run. The package does not include captured checkpoints, frozen field parameters, real-scene inputs, profiling reports, or performance trial data. A successful compilation alone does not establish numerical equivalence or acceleration on your inputs.