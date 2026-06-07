const std = @import("std");
const root = @import("root.zig");
const mlir_harness_exec = @import("mlir_harness_exec.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const cmd = args.next() orelse {
        usage();
        return;
    };

    if (std.mem.eql(u8, cmd, "emit")) {
        const flag = args.next() orelse {
            usage();
            return error.InvalidArgs;
        };
        const out_dir = args.next() orelse {
            usage();
            return error.InvalidArgs;
        };
        if (!std.mem.eql(u8, flag, "--out-dir") or args.next() != null) {
            usage();
            return error.InvalidArgs;
        }
        try mlir_harness_exec.writeAllGeneratedCases(allocator, out_dir);
        return;
    }

    if (std.mem.eql(u8, cmd, "verify")) {
        const path = args.next() orelse {
            usage();
            return error.InvalidArgs;
        };
        if (args.next() != null) {
            usage();
            return error.InvalidArgs;
        }
        const result = try mlir_harness_exec.verifyMlirMaybe(allocator, .{ .enable_external_tools = true, .assume_tools_present = true }, path);
        switch (result.status) {
            .passed => return,
            .skipped => return error.ToolNotConfigured,
            .failed => return error.ToolFailed,
        }
    }

    usage();
    return error.InvalidArgs;
}

fn usage() void {
    std.debug.print(
        \\usage:
        \\  not-cute-mlir-harness emit --out-dir <dir>
        \\  not-cute-mlir-harness verify <file.mlir>
        \\
    , .{});
}
