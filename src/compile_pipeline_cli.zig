const std = @import("std");
const mlir = @import("mlir_text.zig");
const compile_pipeline = @import("compile_pipeline.zig");
const cutlass_bridge = @import("cutlass_bridge.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const cmd = args.next() orelse "plan";
    var out: mlir.TextBuffer(8192) = .{};
    if (std.mem.eql(u8, cmd, "plan")) {
        const input = args.next() orelse "testdata/cutlass/tiled_emit_full_tiled_mma.mlir";
        const req: compile_pipeline.CompileRequest = .{ .input_mlir = input, .work_dir = "zig-cache/not-cute-artifacts", .function_name = "kernel", .keep_ptx = true, .keep_cubin = true };
        const outcome = try compile_pipeline.planBridgeCompilation(.{}, req);
        try outcome.writeJson(&out);
    } else if (std.mem.eql(u8, cmd, "runbook")) {
        const input = args.next() orelse "testdata/cutlass/tiled_emit_full_tiled_mma.mlir";
        const req: compile_pipeline.CompileRequest = .{ .input_mlir = input, .work_dir = "zig-cache/not-cute-artifacts", .function_name = "kernel", .keep_ptx = true, .keep_cubin = true };
        const outcome = try compile_pipeline.planBridgeCompilation(cutlass_bridge.PythonBridgeConfig{}, req);
        try compile_pipeline.writeCompileRunbook(&out, outcome);
    } else {
        return error.InvalidArgs;
    }
    std.debug.print("{s}\n", .{out.slice()});
}
