const std = @import("std");
const mlir = @import("mlir_text.zig");
const cutlass_bridge = @import("cutlass_bridge.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const cmd = args.next() orelse {
        usage();
        return;
    };

    const cfg: cutlass_bridge.PythonBridgeConfig = .{};
    var out: mlir.TextBuffer(8192) = .{};

    if (std.mem.eql(u8, cmd, "discover-command")) {
        const inv = try cutlass_bridge.discoveryInvocation(cfg);
        try inv.writeShell(&out);
    } else if (std.mem.eql(u8, cmd, "metadata-command")) {
        const inv = try cutlass_bridge.metadataInvocation(cfg);
        try inv.writeShell(&out);
    } else if (std.mem.eql(u8, cmd, "verify-command")) {
        const input = args.next() orelse {
            usage();
            return error.InvalidArgs;
        };
        if (args.next() != null) return error.InvalidArgs;
        var pipeline: mlir.TextBuffer(4096) = .{};
        try cutlass_bridge.writeDefaultCutlassPipeline(&pipeline);
        const inv = try cutlass_bridge.verifyInvocation(cfg, input, pipeline.slice());
        try inv.writeShell(&out);
    } else if (std.mem.eql(u8, cmd, "pipeline")) {
        try cutlass_bridge.writeDefaultCutlassPipeline(&out);
    } else if (std.mem.eql(u8, cmd, "lir-pipeline")) {
        try cutlass_bridge.writeLirCutlassPipeline(&out);
    } else if (std.mem.eql(u8, cmd, "rules")) {
        try cutlass_bridge.writeDiscoveryRules(&out);
    } else if (std.mem.eql(u8, cmd, "usage")) {
        usage();
        return;
    } else {
        usage();
        return error.InvalidArgs;
    }

    std.debug.print("{s}\n", .{out.slice()});
}

fn usage() void {
    std.debug.print(
        "usage:\n" ++
            "  not-cute-cutlass-bridge discover-command\n" ++
            "  not-cute-cutlass-bridge metadata-command\n" ++
            "  not-cute-cutlass-bridge verify-command <file.mlir>\n" ++
            "  not-cute-cutlass-bridge pipeline\n" ++
            "  not-cute-cutlass-bridge lir-pipeline\n" ++
            "  not-cute-cutlass-bridge rules\n",
        .{},
    );
}
