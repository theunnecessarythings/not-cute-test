# Kernel builders and memory ownership

This integrated pass adds production-facing Zig APIs for the two previously missing runtime-adjacent domains.

## Kernel builders

`src/kernel_builders.zig` emits full MLIR modules with `gpu.module` / `gpu.func kernel`, not isolated operation snippets. The builders cover:

- copy kernel
- vector-copy kernel
- tiled-copy kernel
- MMA microkernel
- GEMM mainloop
- epilogue kernel
- SM80 GEMM builder
- SM90 TMA/WGMMA builder
- SM100 tcgen05 builder

The builders return or write complete parser-facing modules and also expose compile-request and launch-plan helpers so their output can flow into the existing CUTLASS bridge, CUBIN artifact extraction, and runtime execution wiring.

The architecture-specific builders currently produce complete skeletons using parser-aligned Cute tiled MMA forms plus architecture-specific validation/compile target metadata. GPU launch remains intentionally deferred.

## Memory model

`src/memory_model.zig` provides the coherent buffer ownership layer:

- owned/borrowed/external/managed ownership states
- host buffers with allocator-backed lifetime
- external and CUDA-driver-backed device buffers
- managed/external pointer descriptors
- alignment policy validation
- DLPack-like descriptors
- typed tensor views over runtime allocations
- host↔device, device↔host, device↔device transfer plans and driver hooks

These APIs are ready to wire into actual CUDA execution when a CUDA device and produced CUBIN are available.
