const std = @import("std");
const cute = @import("not_cute");

pub fn main() !void {
    const options: cute.kernel_builders.KernelOptions = .{
        .name = "gemm_mainloop",
        .kind = .gemm_mainloop,
        .arch = .sm90,
        .element = "f32",
        .tile_m = 64,
        .tile_n = 64,
        .tile_k = 16,
        .block_threads = 128,
        .grid_x = 80,
    };

    var module: cute.mlir.TextBuffer(32768) = .{};
    try cute.kernel_builders.writeKernelModule(&module, options);
    const launch = try options.launchConfig();
    const compile = try options.compileRequest(
        "gemm_mainloop.mlir",
        "zig-cache/not-cute-artifacts/gemm_mainloop",
    );
    var command: cute.mlir.TextBuffer(4096) = .{};
    try cute.compile_pipeline.bridgeCompileCommandText(
        .{ .python_exe = "python3", .bridge_script = "tools/cutlass_mlir_bridge.py" },
        compile,
        &command,
    );

    std.debug.print(
        \\// grid=({}, {}, {}) block=({}, {}, {})
        \\// compile: {s}
        \\{s}
    , .{
        launch.grid.x,
        launch.grid.y,
        launch.grid.z,
        launch.block.x,
        launch.block.y,
        launch.block.z,
        command.slice(),
        module.slice(),
    });
}
