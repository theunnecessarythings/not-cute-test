# Pipeline, Semantics, and Architecture Completion Pass

This integrated pass finishes the previously incomplete 1–4 wiring to the extent
that it can be validated in this non-GPU sandbox:

1. **Compile artifact extraction** now recognizes the actual CUTLASS pass-manager
   CUBIN dump path.  Upstream `dump-cubin-path` writes to the exact base path
   such as `work_dir/kernel`, not always to `kernel.cubin`; the Python bridge now
   classifies extensionless ELF files as CUBIN artifacts.
2. **Kernel-shaped MLIR fixture** `testdata/cutlass/kernel_tiled_copy.mlir` is a
   real `gpu.module` / `gpu.func kernel` using CUTLASS Cute tiled copy syntax.
   It parses, runs through `cute-to-nvvm`, and emits an ELF CUBIN in this sandbox
   through the installed `nvidia-cutlass-dsl` package.
3. **Semantic implementation depth** was expanded in `semantics.zig` with product
   scans, coordinate iteration, compactness/cosize checks, coordinate splitting,
   local-tile base-offset calculation, and logical/flat divide shape helpers.
4. **Architecture validation** was expanded in `arch_validation.zig` with typed
   constructors and validations for universal copy, cp.async CG/CA, TMA load/store,
   ldmatrix/stmatrix, tcgen05 load/store, SM80 MMA, SM89 FP8 MMA, SM90 WGMMA,
   and SM100 tcgen05 MMA-style descriptors.

## Validated CUTLASS artifact extraction

The following command was run successfully:

```sh
python3 tools/cutlass_mlir_bridge.py compile-artifact \
  --input testdata/cutlass/kernel_tiled_copy.mlir \
  --work-dir /tmp/notcute_kernel \
  --function tiled_copy_kernel \
  --pipeline "builtin.module(cute-to-nvvm{cubin-format=bin cubin-chip='sm_90' dump-cubin-path='/tmp/notcute_kernel/tiled_copy_kernel' preserve-line-info=true})" \
  --enable-verifier \
  --expect-cubin
```

The bridge reported:

```text
status: passed
lowered_mlir: /tmp/notcute_kernel/tiled_copy_kernel.lowered.mlir
cubin: /tmp/notcute_kernel/tiled_copy_kernel
cubin size: 4344 bytes
```

No CUDA kernel launch was performed here; this validates compilation and artifact
extraction only.

## Build targets

New/updated targets:

```sh
zig build verify-cutlass-kernel-cubin
zig build compile-pipeline
zig build pipeline-verify
```

`verify-cutlass-kernel-cubin` runs the real CUTLASS DSL PassManager over the
kernel-shaped fixture and requires a CUBIN artifact.
