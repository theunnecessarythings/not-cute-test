# Integrated module map

The repository exposes cohesive library modules. The current source layout is organized by library responsibility.

## Core algebra

- `src/tuple.zig` — static tuple/tree utilities.
- `src/layout.zig` — layout data model and coordinate/index operations.
- `src/layout_algebra.zig` — grouping, selection, slicing, dicing, offsets, shape/stride helpers.
- `src/layout_core.zig` — higher-level layout composition, products/divides, inverses, common-layout helpers, tiling helpers.
- `src/basis.zig` — ratio, scaled basis, swizzle descriptors.
- `src/morphism.zig` — finite-set/tuple morphism validation helpers.

## MLIR text and tooling

- `src/mlir_text.zig` — allocation-aware textual MLIR builder.
- `src/mlir_ops.zig` — CUTLASS/Cute-related dialect operation catalog.
- `src/mlir_harness.zig` — golden text and structural verifier harness.

## Types, tensors, atoms, and lowering

- `src/typing.zig` — scalar/type descriptors.
- `src/tensor.zig` — tensor descriptor helpers.
- `src/tensor_ssa.zig` — Zig-native tensor metadata and SSA-value model.
- `src/atom.zig` — traits, atoms, tiled MMA/copy wrappers.
- `src/nvgpu.zig` — source-shaped nvgpu atom descriptors.
- `src/arch.zig` — architecture intrinsic textual hooks.
- `src/arch_catalog.zig` — architecture/dtype/op support catalog.
- `src/copy_mma.zig` — copy/MMA lowering integration over tensors and atoms.
- `src/algorithm.zig` — algorithm-level copy/GEMM planning helpers.
- `src/math.zig` — math operation wrappers.

## Runtime, export, and testing support

- `src/runtime.zig` — runtime descriptor types.
- `src/runtime_plan.zig` — compile/load/launch manifest and command planning.
- `src/cuda_driver.zig` — real CUDA Driver API ABI, dynamic loader, memory/module/function/launch wrappers.
- `src/execution.zig` — executable kernel wiring over compile plans, cubins, symbols, argument packs, and launch configs.
- `src/export.zig` — C-header/wrapper/export metadata.
- `src/jit.zig` — JIT signature/cache/artifact descriptors.
- `src/testing.zig` — testing/benchmark/autotune helper descriptors.
- `src/experimental.zig` — experimental memory/TMA descriptors.

## CUTLASS bridge and parser-aligned fixtures

- `src/cutlass_bridge.zig` — Python bridge invocation planning.
- `src/cutlass_bridge_exec.zig` — optional bridge execution wrapper.
- `src/cutlass_emit.zig` — parser-aligned tensor/copy/MMA emission helpers.
- `src/cutlass_routed.zig` — default-routed generated tensor/copy/MMA modules.
- `src/tiled_emit.zig` — full tiled-copy and tiled-MMA parser fixtures.

## Examples

- `examples/*.zig` — standalone public-API example binaries.


## Core/API architecture pass modules

- `core.zig`: integrated source-named static core API compatibility layer.
- `src/arch_atoms.zig` — source-named nvgpu/arch copy and MMA constructors.
## Newly integrated source-name API modules

- `arch_nvvm.zig` — NVVM wrapper intrinsic emission.
- `tensor_api.zig` — source-name tensor API layer.
- `atom_api.zig` — source-name atom/tiled helper layer.
- `runtime_cuda.zig` — CUDA runtime descriptor layer.
- `compiler_api.zig` — compile option/plan compatibility layer.
- `tree_utils.zig` — PyTree/tree utility compatibility layer.
- `cute_compat.zig` — remaining integrated `cutlass.cute.*` compatibility names.

## Latest integrated pipeline/API/architecture pass

The library includes `compile_pipeline.zig`, `semantics.zig`, and
`arch_catalog.zig`. These add CUTLASS bridge artifact planning, deeper
shape/stride/coordinate semantics, and stricter architecture operation
validation for cp.async/TMA/WGMMA/tcgen05-style descriptors. See
`docs/PIPELINE_API_ARCH_IMPLEMENTATION.md`.


## Kernel builders and memory model integrated pass

Added `src/kernel_builders.zig` for full-module Zig-native kernel builders and `src/memory_model.zig` for host/device/managed/external buffer ownership, DLPack-like interop, tensor views, and host↔device transfer planning. Build targets: `kernel-builders`, `memory-model`, and `verify-kernel-builders-parse`.

Development scripts and executable entry points live under `tools/`. They are
not exported by `src/root.zig`.
