# Integrated compile pipeline, semantic API, and architecture exactness pass

This pass addresses the next four remaining implementation areas without adding
phase-named source files:

1. CUTLASS MLIR compile artifact extraction.
2. Sharded pass-pipeline verifier integration.
3. Deeper semantic parity behind source-name API wrappers.
4. More exact architecture/nvgpu operation behavior.

## New stable modules

- `src/compile_pipeline.zig`
  - `CompileRequest` for parser/canonicalize/`cute-to-nvvm`/LIR pipelines.
  - CUTLASS bridge `compile-artifact` command generation.
  - Expected artifact manifests for lowered MLIR, CUBIN, PTX, object, JSON, and diagnostics.
  - Conversion from compile artifact manifests to `execution.ArtifactSet`.

- `src/pipeline_verify.zig`
  - Sharded verifier manifest for layout/tensor/copy/MMA/tiled/negative cases.
  - Deterministic bridge invocations for parse-only and `builtin.module(canonicalize)` pipeline checks.
  - Shell-script emission for CI and local runs.

- `src/semantics.zig`
  - Static/dynamic extent model.
  - Shape, stride, coordinate, and layout-view semantics.
  - Broadcast compatibility, row/column-major stride generation, linearize/delinearize, subview offsets, tiling math.

- `src/arch_op_exact.zig`
  - Strongly typed SM architecture, element, memory-space, runtime-field, copy, and MMA descriptors.
  - Validation for cp.async, TMA, ldmatrix/stmatrix, WGMMA, and tcgen05-style families.
  - CUTLASS-style MLIR type spelling helpers.
  - Source-manifest summary checks against the generated arch/nvgpu audit.

## Python bridge additions

`tools/cutlass_mlir_bridge.py` now supports:

```sh
python3 tools/cutlass_mlir_bridge.py compile-artifact \
  --input testdata/cutlass/tiled_emit_full_tiled_mma.mlir \
  --work-dir zig-cache/not-cute-artifacts \
  --function kernel \
  --pipeline "builtin.module(cute-to-nvvm{cubin-format=bin cubin-chip='sm_90' dump-cubin-path='zig-cache/not-cute-artifacts/kernel'})" \
  --enable-verifier
```

The command writes a lowered MLIR snapshot and scans the work directory for
CUBIN/PTX/object outputs. It can also be configured to require artifacts with
`--expect-cubin`, `--expect-ptx`, or `--expect-object`.

## Corrected pipeline option behavior

The earlier runtime pipeline emitted `dump-dir=...` inside `cute-to-nvvm`. The
installed CUTLASS DSL pass parser rejects that option. The Zig pipeline writer
now follows the visible CuteDSL source behavior more closely: `dump-dir` is a
front-end/environment option, while the pass pipeline uses `dump-cubin-path` and
`dump-ptx-path` when file dumping is enabled.

## Validation status

Debug/unit validation passed with 170 tests. The CUTLASS bridge parsed and
canonicalized the routed tensor/copy/MMA/tiled fixtures individually. A
`cute-to-nvvm` bridge compile run on the toy tiled-MMA fixture successfully
lowered to LLVM textual MLIR and wrote the lowered snapshot. It did not produce a
CUBIN in this sandbox for that toy fixture; real kernel modules and a complete
CUTLASS runtime environment are still required for production CUBIN emission.

Full-project ReleaseSafe still exceeds the sandbox time budget, but the newly
added `semantics.zig` and `arch_op_exact.zig` pass direct ReleaseSafe tests.
