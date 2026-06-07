# Validation

The project requires Zig 0.16.0 or newer within the 0.16 release line.

## Local checks

Run these checks for source, examples, and build integration:

```sh
zig version
zig fmt --check build.zig src/*.zig examples/*.zig
zig build test --summary all
zig build examples launch --summary all
```

Focused optimized tests are useful for runtime and semantic changes:

```sh
zig test -OReleaseSafe src/cuda_driver.zig
zig test -OReleaseSafe src/execution.zig
zig test -OReleaseSafe src/semantics.zig
zig test -OReleaseSafe src/arch_validation.zig
zig test -OReleaseSafe src/memory_model.zig
```

## CUTLASS checks

These checks require `nvidia-cutlass-dsl` in the selected Python environment:

```sh
python3 tools/cutlass_mlir_bridge.py discover --json
zig build verify-cutlass-parse -Dcutlass-python=python3
zig build verify-cutlass-pipeline -Dcutlass-python=python3
zig build verify-kernel-builders-parse -Dcutlass-python=python3
zig build verify-upstream-parity-parse -Dcutlass-python=python3 -j1
```

The aggregate upstream parity check imports CUTLASS once per fixture and may be
slow. Individual fixtures can be checked with:

```sh
python3 tools/cutlass_mlir_bridge.py parse \
  --input testdata/golden/upstream/hello_world.mlir
```

## Environment-dependent checks

The following checks are not part of the default test suite:

- `zig build verify-cutlass-kernel-cubin` requires a compatible CUTLASS/CUDA
  toolchain and writes a CUBIN artifact.
- Actual CUDA module loading and kernel launch require a CUDA-capable host.
- Full-project `ReleaseSafe` builds may take substantially longer than focused
  module tests.

Historical test counts are intentionally omitted because they become stale as
the suite changes. Treat the current command output as the source of truth.
