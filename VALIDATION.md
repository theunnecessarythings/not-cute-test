# Validation

Validation was run from a clean integrated tree with Zig 0.16 provided by the Python `ziglang` package.

```sh
python3 -m ziglang fmt --check build.zig src/*.zig examples/*.zig
python3 -m ziglang build test --summary all
python3 -m ziglang build examples audit cutlass-bridge cutlass-fixtures cutlass-emission cutlass-routed cutlass-full-tiled harness runtime-plan exec --summary all
```

Observed result:

```text
161/161 Debug tests passed
examples build passed
audit/cutlass-bridge/cutlass-fixtures/cutlass-emission/cutlass-routed/cutlass-full-tiled/harness/runtime-plan builds passed
```

CUTLASS DSL parser bridge spot checks were also run after installing `nvidia-cutlass-dsl`:

```sh
python3 tools/cutlass_mlir_bridge.py discover --json
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/cutlass_routed_tensor_vector.mlir
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/tiled_emit_full_tiled_copy.mlir
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/tiled_emit_full_tiled_mma.mlir
```

Observed result:

```text
CUTLASS DSL discovery passed
cutlass_routed_tensor_vector parsed through CUTLASS DSL bridge
tiled_emit_full_tiled_copy parsed through CUTLASS DSL bridge
tiled_emit_full_tiled_mma parsed through CUTLASS DSL bridge
```

## Not yet validated

- Full-project ReleaseSafe build remains too slow for this sandbox.
- `cute-to-nvvm` lowering and cubin/object generation are not validated here.
- CUDA driver/runtime module loading, function lookup, argument packing, and `cuLaunchKernel` wiring are implemented and compiled. Actual GPU launch is not validated in this sandbox.

## API/architecture parity integration validation

The current integrated artifact adds source-derived API and architecture manifests:

```sh
python3 -m ziglang fmt --check build.zig src/*.zig examples/*.zig
python3 -m ziglang build test --summary all
python3 -m ziglang build examples audit api-audit cutlass-bridge cutlass-fixtures cutlass-emission cutlass-routed cutlass-full-tiled harness runtime-plan exec --summary all
python3 tools/cutlass_mlir_bridge.py discover --json
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/cutlass_routed_tensor_vector.mlir
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/tiled_emit_full_tiled_copy.mlir
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/tiled_emit_full_tiled_mma.mlir
```

Observed result in this sandbox:

```text
161/161 Debug tests passed
all listed CLI/example build targets passed
CUTLASS DSL discovery passed
three CUTLASS parser spot checks passed
```

Full-project ReleaseSafe still did not complete inside the sandbox timeout. Debug/default validation is the reliable local target at this point.
## Integrated API closure validation

Validated with 156/156 Debug tests, all example/CLI build targets, CUTLASS DSL discovery through `nvidia-cutlass-dsl`, and parser checks for routed tensor vector, full tiled copy, and full tiled MMA fixtures.


## Runtime execution wiring validation

Additional commands run for this revision:

```sh
python3 -m ziglang build exec --summary all
python3 -m ziglang test -OReleaseSafe src/cuda_driver.zig
python3 -m ziglang test -OReleaseSafe src/execution.zig
./zig-out/bin/not-cute-exec
```

Observed result:

```text
exec build passed
cuda_driver.zig direct ReleaseSafe tests passed
execution.zig direct ReleaseSafe tests passed
not-cute-exec emitted dry-run launch JSON
```

## Latest integrated pipeline/API/architecture pass

The library now includes `compile_pipeline.zig`, `pipeline_verify.zig`,
`semantics.zig`, and `arch_op_exact.zig`. These add CUTLASS bridge artifact
planning/extraction commands, sharded parser/pipeline verification, deeper
shape/stride/coordinate semantics, and stricter architecture operation
validation for cp.async/TMA/WGMMA/tcgen05-style descriptors. See
`docs/PIPELINE_API_ARCH_IMPLEMENTATION.md`.

### Integrated pipeline/API/architecture pass validation

Validated in this environment with Zig 0.16 via `python3 -m ziglang`:

```sh
python3 -m ziglang fmt --check build.zig src/*.zig examples/*.zig
python3 -m ziglang build test --summary all
python3 -m ziglang build examples audit api-audit cutlass-bridge cutlass-fixtures \
  cutlass-emission cutlass-routed cutlass-full-tiled harness runtime-plan exec \
  compile-pipeline pipeline-verify --summary all
```

Result: `170/170 Debug tests passed`; all listed CLI/example build targets succeeded.

CUTLASS DSL bridge checks:

```sh
python3 tools/cutlass_mlir_bridge.py discover --json
python3 tools/cutlass_mlir_bridge.py verify --input testdata/cutlass/cutlass_routed_tensor_vector.mlir --pipeline 'builtin.module(canonicalize)' --enable-verifier
python3 tools/cutlass_mlir_bridge.py verify --input testdata/cutlass/cutlass_routed_copy_atom.mlir --pipeline 'builtin.module(canonicalize)' --enable-verifier
python3 tools/cutlass_mlir_bridge.py verify --input testdata/cutlass/cutlass_routed_mma_atom.mlir --pipeline 'builtin.module(canonicalize)' --enable-verifier
python3 tools/cutlass_mlir_bridge.py verify --input testdata/cutlass/tiled_emit_full_tiled_copy.mlir --pipeline 'builtin.module(canonicalize)' --enable-verifier
python3 tools/cutlass_mlir_bridge.py verify --input testdata/cutlass/tiled_emit_full_tiled_mma.mlir --pipeline 'builtin.module(canonicalize)' --enable-verifier
python3 tools/cutlass_mlir_bridge.py compile-artifact --input testdata/cutlass/tiled_emit_full_tiled_mma.mlir --work-dir /tmp/notcute_art3 --function tiled_mma --pipeline "builtin.module(cute-to-nvvm{cubin-format=bin cubin-chip='sm_90' dump-cubin-path='/tmp/notcute_art3/tiled_mma'})" --enable-verifier
```

Result: CUTLASS discovery passed, all five canonicalize pipeline checks passed, and `compile-artifact` produced a lowered MLIR snapshot through `cute-to-nvvm`. The toy tiled-MMA fixture did not emit a CUBIN in this sandbox; production CUBIN extraction still depends on full kernel-shaped MLIR and the local CUTLASS/CUDA toolchain.

Direct optimized checks for new semantic modules:

```sh
python3 -m ziglang test -OReleaseSafe src/semantics.zig
python3 -m ziglang test -OReleaseSafe src/arch_op_exact.zig
```

Both passed. Full-project ReleaseSafe still exceeds this sandbox's timeout budget.

## Latest integrated pipeline/API/architecture validation

Validated in the sandbox after the completion pass:

```sh
python3 -m ziglang build test --summary all
python3 -m ziglang build examples audit api-audit cutlass-bridge cutlass-fixtures \
  cutlass-emission cutlass-routed cutlass-full-tiled harness runtime-plan exec \
  compile-pipeline pipeline-verify --summary all
python3 -m ziglang build verify-cutlass-kernel-cubin --summary all
python3 -m ziglang test -OReleaseSafe src/semantics.zig
python3 -m ziglang test -OReleaseSafe src/arch_op_exact.zig
```

Results: 177/177 Debug tests passed.  The CUTLASS `cute-to-nvvm` kernel fixture
emitted an extensionless ELF CUBIN at the requested `dump-cubin-path`.


## Kernel builders and memory model integrated pass

Added `src/kernel_builders.zig` for full-module Zig-native kernel builders and `src/memory_model.zig` for host/device/managed/external buffer ownership, DLPack-like interop, tensor views, and host↔device transfer planning. Build targets: `kernel-builders`, `memory-model`, and `verify-kernel-builders-parse`.


## Kernel builders and memory model validation

Validated after adding integrated kernel builders and coherent memory ownership APIs:

```sh
python3 -m ziglang fmt --check build.zig src/*.zig examples/*.zig
python3 -m ziglang build test --summary all
python3 -m ziglang build examples audit api-audit cutlass-bridge cutlass-fixtures cutlass-emission cutlass-routed cutlass-full-tiled harness runtime-plan exec compile-pipeline pipeline-verify kernel-builders memory-model --summary all
python3 -m ziglang test -OReleaseSafe src/memory_model.zig
```

Result: 184/184 Debug tests passed; all listed CLIs/examples built; `memory_model.zig` ReleaseSafe direct tests passed.

CUTLASS parser checks passed for all generated kernel-builder MLIR fixtures individually:

- copy kernel
- vector-copy kernel
- tiled-copy kernel
- MMA microkernel
- GEMM mainloop
- epilogue kernel
- SM80 GEMM skeleton
- SM90 TMA/WGMMA skeleton
- SM100 tcgen05 skeleton

The generated tiled-copy kernel also passed `cute-to-nvvm` and emitted a real CUBIN at `zig-cache/not-cute-artifacts/builder_tiled_copy/tiled_copy_kernel` in this sandbox. Actual CUDA launch remains deferred.


## Upstream parity validation

Validated in this artifact:

```sh
python3 -m ziglang build test --summary all
python3 -m ziglang build examples kernel-builders memory-model upstream-parity --summary all
python3 -m ziglang build audit api-audit cutlass-bridge cutlass-fixtures cutlass-emission cutlass-routed cutlass-full-tiled harness runtime-plan exec compile-pipeline pipeline-verify --summary all
```

Result: 188/188 Debug unit tests passed, and all listed CLI/example targets built.

The aggregate `verify-upstream-parity-parse` target is available, but in this sandbox aggregate CUTLASS imports may exceed the timeout. Each upstream golden was parsed individually with `tools/cutlass_mlir_bridge.py parse --input ...` and passed.


Upstream parity parser checks run individually in this sandbox:

```sh
for f in testdata/golden/upstream/*.mlir; do
  python3 tools/cutlass_mlir_bridge.py parse --input "$f"
done
```

Result: all 9 upstream parity goldens parsed through CUTLASS DSL. The aggregate build target exists as `zig build verify-upstream-parity-parse`, but each invocation imports CUTLASS and can exceed this sandbox timeout when run as one aggregate target.
