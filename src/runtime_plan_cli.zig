const std = @import("std");
const runtime_plan = @import("runtime_plan.zig");
const mlir = @import("mlir_text.zig");

pub fn main(init: std.process.Init) !void {
    _ = init;
    var cmd: mlir.TextBuffer(4096) = .{};
    const plan: runtime_plan.CompilePlan = .{ .options = .{ .function_name = "kernel" }, .input_mlir = "kernel.mlir", .output_cubin = "kernel.cubin" };
    try plan.writeCuteOptCommand(&cmd);
    std.debug.print("{s}\n", .{cmd.slice()});
}
