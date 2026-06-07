# Integrated module map

The repository exposes cohesive library modules. The current source layout is organized by library responsibility.

## Core algebra

- `src/tuple.zig` — static tuple/tree utilities.
- `src/layout.zig` — layout data model, coordinate/index operations, algebra, products/divides, inverses, and tiling helpers.
- `src/basis.zig` — ratio, scaled basis, swizzle descriptors.
- `src/morphism.zig` — finite-set/tuple morphism validation helpers.

## MLIR text and tooling

- `src/mlir.zig` — textual MLIR builder, CUTLASS/Cute operation catalog, golden text, and structural verifier harness.

## Types, tensors, atoms, and lowering

- `src/typing.zig` — scalar/type descriptors.
- `src/tensor.zig` — tensor descriptors, metadata, SSA-value model, and source-name API helpers.
- `src/atom.zig` — traits, atoms, tiled MMA/copy wrappers.
- `src/nvgpu.zig` — source-shaped nvgpu atom descriptors.
- `src/arch.zig` — unified architecture primitives, catalog, atoms, and NVVM intrinsics.
- `src/copy_mma.zig` — copy/MMA lowering integration over tensors and atoms.
- `src/algorithm.zig` — algorithm-level copy/GEMM planning helpers.
- `src/math.zig` — math operation wrappers.

## Runtime, export, and testing support

- `src/runtime.zig` — runtime descriptors, CUDA compatibility helpers, and compile/load/launch planning.
- `src/cuda_driver.zig` — real CUDA Driver API ABI, dynamic loader, memory/module/function/launch wrappers.
- `src/execution.zig` — executable kernel wiring over compile plans, cubins, symbols, argument packs, and launch configs.
- `src/export.zig` — C-header/wrapper/export metadata.
- `src/jit.zig` — JIT signature/cache/artifact descriptors.
- `src/testing.zig` — testing/benchmark/autotune helper descriptors.
- `src/experimental.zig` — experimental memory/TMA descriptors.

## CUTLASS bridge and parser-aligned fixtures

- `src/cutlass.zig` — Python bridge planning, parser-aligned emission helpers, routed generated modules, and full tiled-copy/MMA fixtures.

## Examples

- `examples/*.zig` — standalone public-API example binaries.


## Core/API architecture pass modules

- `core.zig`: integrated source-named static core API compatibility layer.

## Newly integrated source-name APIs

- `tensor.zig` — source-name tensor API layer.
- `atom.zig` — source-name atom/tiled helper layer.
- `runtime.zig` — CUDA runtime descriptor layer.
- `tree_utils.zig` — PyTree/tree utility compatibility layer.
- `cute_compat.zig` — remaining integrated `cutlass.cute.*` compatibility names.

## Latest integrated pipeline/API/architecture pass

The library includes `compile_pipeline.zig`, `semantics.zig`, and
`arch.zig`. These add CUTLASS bridge artifact planning, deeper
shape/stride/coordinate semantics, and stricter architecture operation
validation for cp.async/TMA/WGMMA/tcgen05-style descriptors. See
`docs/PIPELINE_API_ARCH_IMPLEMENTATION.md`.


## Kernel builders and memory model integrated pass

Added `src/kernel_builders.zig` for full-module Zig-native kernel builders and compile-request helpers, and `src/memory_model.zig` for host/device/managed/external buffer ownership, DLPack-like interop, tensor views, and host↔device transfer planning. Build targets: `kernel-builders`, `memory-model`, and `verify-kernel-builders-parse`.

Development scripts and executable entry points live under `tools/`. They are
not exported by `src/root.zig`.ot.zig`.
