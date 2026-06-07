const std = @import("std");
const examples_api = @import("examples_api.zig");
const mlir = @import("mlir_text.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const cmd = args.next() orelse {
        usage();
        return;
    };

    if (std.mem.eql(u8, cmd, "list")) {
        var out: mlir.TextBuffer(4096) = .{};
        try examples_api.writeExampleIndex(&out);
        std.debug.print("{s}", .{out.slice()});
        return;
    }

    if (std.mem.eql(u8, cmd, "emit")) {
        const name = args.next() orelse {
            usage();
            return error.InvalidArgs;
        };
        var out: mlir.TextBuffer(32768) = .{};
        try examples_api.renderExample(parseKind(name) orelse return error.InvalidArgs, &out);
        std.debug.print("{s}", .{out.slice()});
        return;
    }

    if (std.mem.eql(u8, cmd, "plan")) {
        const name = args.next() orelse "gemm_skeleton";
        var out: mlir.TextBuffer(4096) = .{};
        try examples_api.writeRuntimePlanForExample(parseKind(name) orelse return error.InvalidArgs, &out);
        std.debug.print("{s}", .{out.slice()});
        return;
    }

    usage();
    return error.InvalidArgs;
}

fn parseKind(name: []const u8) ?examples_api.ExampleKind {
    for (examples_api.all_examples) |kind| {
        if (std.mem.eql(u8, name, kind.stem())) return kind;
    }
    return null;
}

fn usage() void {
    std.debug.print(
        \\usage:
        \\  not-cute-examples list
        \\  not-cute-examples emit <layout_demo|tensor_demo|copy_demo|mma_demo|gemm_skeleton>
        \\  not-cute-examples plan [example]
        \\
    , .{});
}
