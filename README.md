# not-cute

A Zig-native, zero-default-dependency porting work-in-progress for NVIDIA CuteDSL/CuTe concepts.

This repository is organized as a usable Zig library with domain modules. The public surface is grouped by domain:

- `layout`, `tuple`, `layout_algebra`, `layout_core`, `basis`, `morphism`
- `mlir_text`, `mlir_ops`, `mlir_harness`
- `typing`, `tensor`, `tensor_ssa`
- `atom`, `nvgpu`, `arch`, `arch_catalog`
- `copy_mma`, `algorithm`, `math`
- `runtime`, `runtime_plan`, `export_`, `jit`, `testing`, `experimental`
- `cutlass_bridge`, `cutlass_fixtures`, `cutlass_emit`, `cutlass_routed`, `tiled_emit`, `integration_audit`

The library defaults to Zig-only tests and deterministic textual MLIR generation. CUTLASS/MLIR validation is optional and uses an external Python bridge when `nvidia-cutlass-dsl` is installed.

## Build

```sh
python3 -m pip install ziglang
python3 -m ziglang fmt --check build.zig src/*.zig examples/*.zig
python3 -m ziglang build test --summary all
python3 -m ziglang build examples harness runtime-plan --summary all
```

Current local validation for this integrated artifact:

```text
161/161 Debug tests passed
examples build passed
harness/runtime-plan/exec/CUTLASS helper CLIs build passed
```


## Runtime execution wiring

This revision adds `src/cuda_driver.zig` and `src/execution.zig` for real CUDA Driver API dynamic loading, cubin/module loading, function lookup, argument packing, `cuLaunchKernel` orchestration, execution manifests, and generated C wrappers. Actual GPU launch is intentionally not validated in this sandbox, but the code path is compiled, unit-tested, and exposed through:

```sh
python3 -m ziglang build exec --summary all
./zig-out/bin/not-cute-exec
```

See `docs/RUNTIME_EXECUTION_WIRING.md`.

## Optional CUTLASS parser bridge

Install the real CUTLASS DSL package:

```sh
python3 -m pip install nvidia-cutlass-dsl
python3 tools/cutlass_mlir_bridge.py discover --json
```

Then run parser checks, either individually:

```sh
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/cutlass_routed_tensor_vector.mlir
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/tiled_emit_full_tiled_copy.mlir
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/tiled_emit_full_tiled_mma.mlir
```

or through build targets:

```sh
python3 -m ziglang build verify-cutlass-tensor -Dcutlass-python=python3
python3 -m ziglang build verify-cutlass-copy -Dcutlass-python=python3
python3 -m ziglang build verify-cutlass-mma -Dcutlass-python=python3
python3 -m ziglang build verify-cutlass-negative -Dcutlass-python=python3
```

## Module map

See `docs/MODULE_MAP.md` for the integrated source layout.

## Current honesty boundary

This is **not** a full production-complete CuteDSL replacement. It is an integrated Zig library containing the work so far: layout algebra, textual MLIR construction, atoms/traits, tensor/SSA descriptors, copy/MMA lowering descriptors, runtime/export planning, examples, CUTLASS parser bridge, and parser-aligned Cute MLIR fixtures.

Remaining work is tracked in `docs/REMAINING_FULL_PORT.md`. Major gaps include real `cute-to-nvvm` pass-pipeline success for generated kernels, CUDA module loading/launch, exhaustive source API parity, exact architecture-specific op validation/lowering, and upstream-scale regression coverage.

## API and architecture audit

The integrated library now includes source-derived API and architecture manifests generated from the uploaded CuteDSL tree:

- `src/api_surface.zig` / `docs/api_surface_manifest.json`
- `src/arch_exact.zig` / `docs/arch_nvgpu_manifest.json`
- `zig build api-audit` builds `not-cute-api-audit` for quick count checks.

This is intentionally a parity guardrail, not a claim that every Python behavior has been reimplemented. It makes missing API and arch-op behavior explicit and testable inside the Zig package.


## Integrated core/API and architecture coverage

This revision adds real integrated modules rather than phase checkpoints:

- `src/core.zig` for source-named static `cutlass.cute.core` layout/tuple helpers.
- `src/arch_ops.zig` for typed `cute.arch` / `cute.nvgpu` copy and MMA constructors.
- `src/api_surface.zig` regenerated against the updated public Zig names.

Current API audit:

```text
source_public_records=1327
zig_name_matches=297
implemented_cute_records=254
cute_records=721
arch_nvgpu_records=380
```

Run:

```sh
python3 -m ziglang build test --summary all
python3 -m ziglang build api-audit --summary all
./zig-out/bin/not-cute-api-audit
```
## Latest integrated API closure pass

The current artifact adds source-name compatibility modules for NVVM wrappers, tensor/atom APIs, CUDA runtime descriptors, compiler options, tree utilities, nvgpu aliases, and remaining `cutlass.cute.*` wrappers. The API audit now reports 841 / 1,327 public source-name matches and 721 / 721 `cutlass.cute.*` name matches. See `docs/API_CLOSURE_CONTINUATION.md`.

## Latest integrated pipeline/API/architecture pass

The library now includes `compile_pipeline.zig`, `pipeline_verify.zig`,
`semantics.zig`, and `arch_op_exact.zig`. These add CUTLASS bridge artifact
planning/extraction commands, sharded parser/pipeline verification, deeper
shape/stride/coordinate semantics, and stricter architecture operation
validation for cp.async/TMA/WGMMA/tcgen05-style descriptors. See
`docs/PIPELINE_API_ARCH_IMPLEMENTATION.md`.

## CUTLASS compile artifact path

The integrated bridge now supports real `cute-to-nvvm` artifact extraction for a
kernel-shaped fixture.  With `nvidia-cutlass-dsl` installed, run:

```sh
python3 -m ziglang build verify-cutlass-kernel-cubin --summary all
```

The target compiles `testdata/cutlass/kernel_tiled_copy.mlir` through the CUTLASS
DSL `PassManager` and requires the dumped CUBIN artifact.  Runtime CUDA launch is
still a separate environment-dependent step.


## Kernel builders and memory model integrated pass

Added `src/kernel_builders.zig` for full-module Zig-native kernel builders and `src/memory_model.zig` for host/device/managed/external buffer ownership, DLPack-like interop, tensor views, and host↔device transfer planning. Build targets: `kernel-builders`, `memory-model`, and `verify-kernel-builders-parse`.


## Upstream example parity

The library now includes `src/upstream_parity.zig`, `examples/upstream/*.zig`, and `testdata/golden/upstream/*.mlir`. These map the packaged CUTLASS CuTeDSL notebooks and FFI tensor example into Zig-native examples with parser-checked MLIR goldens. Build with:

```sh
python3 -m ziglang build upstream-parity
python3 -m ziglang build verify-upstream-parity-parse -j1
```

`cuda_graphs` is represented as a launch/dry-run parity example because real CUDA graph capture requires an external framework/runtime stream.
