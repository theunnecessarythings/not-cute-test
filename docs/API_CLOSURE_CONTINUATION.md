# Integrated API/Architecture Closure Continuation

This pass continues the integrated library conversion without adding checkpoint/phase modules.
It preserves existing implementation modules and adds source-name compatibility surfaces on top of
real descriptors, MLIR builders, runtime plans, and tensor/layout types.

## Added modules

- `arch_nvvm.zig`: source-named wrappers for `cutlass.cute.arch.nvvm_wrappers` intrinsics.  The wrappers emit deterministic textual MLIR calls such as `cutlass.nvvm.lane_idx`, `cutlass.nvvm.shuffle_sync_op`, and synchronization/fence operations.
- `tensor_api.zig`: source-name tensor constructors and helpers over `tensor.zig` and `tensor_ssa.zig`.
- `atom_api.zig`: source-name atom/tiled-copy/tiled-MMA helpers over `atom.zig`.
- `runtime_cuda.zig`: CUDA runtime descriptor API mirroring the Python runtime names while keeping this Zig library driver-free by default.
- `compiler_api.zig`: source-name compile option and compiler descriptor API.
- `tree_utils.zig`: PyTree/tree utility compatibility layer over Zig layout trees.
- `nvgpu_aliases.zig`: source-derived nvgpu/arch operation aliases connected to `arch_manifest.zig` records.
- `cute_compat.zig`: remaining `cutlass.cute.*` source-name wrappers, including tuple/core/runtime/testing/export/experimental pipeline names.

## API audit movement

Before this pass:

- Public source records: 1,327
- Public Zig name matches: 297
- `cutlass.cute.*` matches: 254 / 721

After this pass:

- Public source records: 1,327
- Public Zig name matches: 841
- `cutlass.cute.*` matches: 721 / 721

The count is name-surface coverage, not a claim that every function has identical dynamic semantics or CUDA execution behavior.

## Validation performed

```sh
python3 -m ziglang fmt --check build.zig src/*.zig examples/*.zig
python3 -m ziglang build test --summary all
python3 -m ziglang build examples audit api-audit cutlass-bridge cutlass-fixtures cutlass-emission cutlass-routed cutlass-full-tiled harness runtime-plan --summary all
python3 tools/cutlass_mlir_bridge.py discover --json
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/cutlass_routed_tensor_vector.mlir
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/tiled_emit_full_tiled_copy.mlir
python3 tools/cutlass_mlir_bridge.py parse --input testdata/cutlass/tiled_emit_full_tiled_mma.mlir
```

Results:

- 156/156 Debug Zig tests passed.
- All listed CLI/example build targets passed.
- CUTLASS DSL discovery passed with `nvidia-cutlass-dsl` 4.5.2 in this sandbox.
- Three CUTLASS parser spot checks passed.

## Remaining known limitations

- Source-name coverage for `cutlass.cute.*` is now complete by audit name matching, but not every API has full Python-identical dynamic behavior.
- CUDA runtime functions are descriptor/plan APIs, not driver-backed execution.
- `cutlass.arch.nvvm_wrappers` emits textual MLIR intrinsic hooks; final lowering still depends on CUTLASS pass pipeline success.
- Non-Cute integration surfaces remain incomplete: Torch, JAX, TVM FFI, pipeline schedulers, distributed helpers, and utility modules.
- Full `cute-to-nvvm` pass-pipeline validation still needs to be made systematic and fixed where emitted modules fail.
