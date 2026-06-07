# Runtime execution wiring

This revision adds the CUDA execution wiring needed to move from compile/load/launch descriptors toward an executable runtime path.  The implementation is deliberately split into two layers:

- `src/cuda_driver.zig` provides the real CUDA Driver API ABI surface and dynamic loader.
- `src/execution.zig` connects compile plans, artifacts, kernel symbols, launch configs, argument packs, C wrapper generation, and CUDA-driver launch orchestration.

The code is ready to call into a CUDA-capable environment when these external pieces exist:

1. CUDA driver library such as `libcuda.so.1`.
2. CUTLASS/CuTe MLIR pipeline that produces a cubin/fatbin/object for the emitted MLIR.
3. A cubin/fatbin containing the expected kernel entry symbol.
4. Correct runtime argument storage whose addresses are passed to `cuLaunchKernel`.

## Added runtime APIs

`cuda_driver.zig` contains typed bindings for:

- `cuInit`
- `cuDriverGetVersion`
- `cuDeviceGetCount`
- `cuDeviceGet`
- `cuDeviceGetName`
- `cuDeviceComputeCapability`
- `cuDevicePrimaryCtxRetain`
- `cuDevicePrimaryCtxRelease`
- `cuCtxCreate_v2`
- `cuCtxDestroy_v2`
- `cuCtxSetCurrent`
- `cuCtxGetCurrent`
- `cuModuleLoad`
- `cuModuleLoadData`
- `cuModuleUnload`
- `cuModuleGetFunction`
- `cuLaunchKernel`
- `cuStreamCreate`
- `cuStreamDestroy_v2`
- `cuStreamSynchronize`
- `cuMemAlloc_v2`
- `cuMemFree_v2`
- `cuMemcpyHtoD_v2`
- `cuMemcpyDtoH_v2`
- `cuMemcpyDtoD_v2`
- `cuGetErrorString`

`execution.zig` contains:

- `ArtifactSet`
- `KernelBinding`
- `ExecutableKernel`
- `runDry`
- `launchWithCudaDriver`
- `writeExecutionCWrapper`
- `writeBuildRunbook`

## CLI

The `exec` build target installs `not-cute-exec`, a dry-run CLI that emits launch JSON for a default kernel plan:

```sh
python3 -m ziglang build exec --summary all
./zig-out/bin/not-cute-exec
```

## Validation in this sandbox

The CUDA driver path was compiled and unit-tested without requiring a GPU.  Actual `cuLaunchKernel` was not executed here because CUDA/GPU availability is not guaranteed.

Validated commands:

```sh
python3 -m ziglang build test --summary all
python3 -m ziglang build examples audit api-audit cutlass-bridge cutlass-fixtures cutlass-emission cutlass-routed cutlass-full-tiled harness runtime-plan exec --summary all
python3 -m ziglang test -OReleaseSafe src/cuda_driver.zig
python3 -m ziglang test -OReleaseSafe src/execution.zig
./zig-out/bin/not-cute-exec
```

Observed results:

- 161/161 Debug tests passed.
- All example/helper CLIs, including `exec`, built successfully.
- `cuda_driver.zig` direct ReleaseSafe tests passed.
- `execution.zig` direct ReleaseSafe tests passed.
- `not-cute-exec` emitted a dry-run JSON launch manifest.

## Boundary

This is no longer merely pseudo-code.  The driver ABI, dynamic symbol lookup, module loading, function lookup, argument packing, stream synchronization, memory copy helpers, and kernel launch call are present in Zig.  What remains unverified is end-to-end execution against a real generated cubin on a CUDA machine.
