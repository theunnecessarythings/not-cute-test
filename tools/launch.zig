const std = @import("std");
const cute = @import("not_cute");
const runtime = cute.runtime;
const runtime_plan = cute.runtime_plan;
const execution = cute.execution;

fn stderr(message: []const u8) void {
    _ = std.os.linux.write(2, message.ptr, message.len);
}

fn run(init: std.process.Init.Minimal) !void {
    const raw_args = init.args.vector;

    if (raw_args.len < 3) {
        stderr("Usage: not-cute-launch <cubin_path> <kernel_name> [grid_x grid_y grid_z] [block_x block_y block_z] [cuda_driver_path]\n");
        std.process.exit(1);
    }

    const cubin_path = std.mem.span(raw_args[1]);
    const kernel_name = std.mem.span(raw_args[2]);

    var gx: u32 = 1;
    var gy: u32 = 1;
    var gz: u32 = 1;
    var bx: u32 = 128;
    var by: u32 = 1;
    var bz: u32 = 1;

    if (raw_args.len >= 6) {
        gx = try std.fmt.parseInt(u32, std.mem.span(raw_args[3]), 10);
        gy = try std.fmt.parseInt(u32, std.mem.span(raw_args[4]), 10);
        gz = try std.fmt.parseInt(u32, std.mem.span(raw_args[5]), 10);
    }
    if (raw_args.len >= 9) {
        bx = try std.fmt.parseInt(u32, std.mem.span(raw_args[6]), 10);
        by = try std.fmt.parseInt(u32, std.mem.span(raw_args[7]), 10);
        bz = try std.fmt.parseInt(u32, std.mem.span(raw_args[8]), 10);
    }
    const cuda_driver_path = if (raw_args.len >= 10) std.mem.span(raw_args[9]) else "libcuda.so.1";

    const cfg = try runtime.LaunchConfig.init(
        try runtime.Dim3.init(gx, gy, gz),
        try runtime.Dim3.init(bx, by, bz),
        0,
        runtime.Stream.default(),
    );

    const symbols = try runtime_plan.RuntimeSymbols.init("notcute", kernel_name);
    const launch: runtime_plan.LaunchPlan = .{
        .symbols = symbols,
        .module = try runtime.BinaryModule.init(cubin_path, .cubin),
        .config = cfg,
        .args = .{},
    };

    const compile: runtime_plan.CompilePlan = .{
        .tools = .{ .cuda_driver_library = cuda_driver_path },
        .options = .{ .function_name = kernel_name },
        .input_mlir = "dummy.mlir",
        .output_cubin = cubin_path,
    };

    const exe = try execution.makeExecutableKernel(
        &compile,
        &launch,
        "dummy.mlir",
        cubin_path,
    );

    stderr("Attempting CUDA kernel launch...\n");

    const result = try execution.launchCopyWithCudaDriver(std.heap.c_allocator, exe);

    _ = result;
    stderr("CUDA kernel launch succeeded.\n");
}

pub fn main(init: std.process.Init.Minimal) void {
    run(init) catch {
        stderr("CUDA kernel launch failed.\n");
        std.process.exit(1);
    };
}
