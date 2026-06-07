const std = @import("std");
const mlir = @import("mlir.zig");
const compile_pipeline = @import("compile_pipeline.zig");

pub const Error = mlir.Error || compile_pipeline.Error;

pub fn writeEmptyGpuKernel(out: anytype, kernel_name: []const u8) Error!void {
    try mlir.validateSymbol(kernel_name);
    try out.append("module {\n");
    try out.append("  gpu.module @kernels {\n");
    try out.append("    gpu.func @");
    try out.append(kernel_name);
    try out.append("() kernel {\n");
    try out.append("      gpu.return\n");
    try out.append("    }\n");
    try out.append("  }\n");
    try out.append("}\n");
}

pub fn writeTiledCopyGpuKernel(out: anytype, kernel_name: []const u8) Error!void {
    try mlir.validateSymbol(kernel_name);
    try out.append("!copy_simt = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 32 b>, layout_copy_tv = <\"(1,1):(1,1)\">, tiler_mn = <\"[1:0;1:0]\">>\n");
    try out.append("!memref_gmem_f32_1x1 = !cute.memref<f32, gmem, \"(1,1):(1,1)\">\n");
    try out.append("!memref_gmem_f32_partition = !cute.memref<f32, gmem, \"((1,1),1,1):((0,0),0,0)\">\n");
    try out.append("module {\n");
    try out.append("  gpu.module @kernels {\n");
    try out.append("    gpu.func @");
    try out.append(kernel_name);
    try out.append("(%arg0: !memref_gmem_f32_1x1, %arg1: !memref_gmem_f32_1x1, %arg2: !cute.coord<\"0\">) kernel {\n");
    try out.append("      %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>\n");
    try out.append("      %tiled = cute.make_tiled_copy(%atom) : !copy_simt\n");
    try out.append("      %src_partitioned = cute.tiled.copy.partition_S(%tiled, %arg0, %arg2) : (!copy_simt, !memref_gmem_f32_1x1, !cute.coord<\"0\">) -> !memref_gmem_f32_partition\n");
    try out.append("      %dst_partitioned = cute.tiled.copy.partition_D(%tiled, %arg1, %arg2) : (!copy_simt, !memref_gmem_f32_1x1, !cute.coord<\"0\">) -> !memref_gmem_f32_partition\n");
    try out.append("      cute.copy(%tiled, %src_partitioned, %dst_partitioned) : (!copy_simt, !memref_gmem_f32_partition, !memref_gmem_f32_partition)\n");
    try out.append("      gpu.return\n");
    try out.append("    }\n");
    try out.append("  }\n");
    try out.append("}\n");
}

pub fn compileRequestForTiledCopyKernel(
    input_mlir: []const u8,
    work_dir: []const u8,
    arch: []const u8,
) compile_pipeline.CompileRequest {
    return compile_pipeline.defaultKernelCompileRequest(
        input_mlir,
        work_dir,
        "tiled_copy_kernel",
        arch,
    );
}

test "kernel_modules: writes kernel-shaped CUTLASS MLIR" {
    var out: mlir.TextBuffer(4096) = .{};
    try writeTiledCopyGpuKernel(&out, "tiled_copy_kernel");
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "gpu.module @kernels") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.slice(), "cute.copy") != null);
}

test "kernel_modules: compile request targets executable cubin dump path" {
    const req = compileRequestForTiledCopyKernel(
        "kernel.mlir",
        "zig-cache/not-cute/kernel",
        "sm_90",
    );
    var p: mlir.TextBuffer(256) = .{};
    try req.artifactPath(.cubin, &p);
    try std.testing.expectEqualStrings(
        "zig-cache/not-cute/kernel/tiled_copy_kernel",
        p.slice(),
    );
}
