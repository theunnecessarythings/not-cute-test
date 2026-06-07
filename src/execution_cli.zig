const std = @import("std");
const mlir = @import("mlir_text.zig");
const runtime = @import("runtime.zig");
const runtime_plan = @import("runtime_plan.zig");
const execution = @import("execution.zig");

pub fn main() !void {
    const mlir_path = "kernel.mlir";
    const cubin_path = "kernel.cubin";
    const kernel = "kernel";
    const cfg = try runtime.LaunchConfig.init(try runtime.Dim3.init(1, 1, 1), try runtime.Dim3.init(128, 1, 1), 0, runtime.Stream.default());
    const symbols = try runtime_plan.RuntimeSymbols.init("notcute", kernel);
    const launch: runtime_plan.LaunchPlan = .{
        .symbols = symbols,
        .module = try runtime.BinaryModule.init(cubin_path, .cubin),
        .config = cfg,
        .args = .{},
    };
    const compile: runtime_plan.CompilePlan = .{ .options = .{ .function_name = kernel }, .input_mlir = mlir_path, .output_cubin = cubin_path };
    const exe = try execution.makeExecutableKernel(&compile, &launch, mlir_path, cubin_path);
    var out: mlir.TextBuffer(8192) = .{};
    const report = try exe.dryRun();
    try report.writeJson(&out);
    std.debug.print("{s}", .{out.slice()});
}
