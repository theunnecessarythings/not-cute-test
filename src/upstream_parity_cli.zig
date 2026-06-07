const std = @import("std");
const upstream = @import("upstream_parity.zig");
const mlir = @import("mlir_text.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const mode = args.next() orelse "report";
    if (std.mem.eql(u8, mode, "json")) {
        var out: mlir.TextBuffer(60000) = .{};
        try upstream.writeInventoryJson(&out);
        std.debug.print("{s}", .{out.slice()});
        return;
    }
    if (std.mem.eql(u8, mode, "mlir")) {
        var out: mlir.TextBuffer(200000) = .{};
        for (upstream.examples) |ex| {
            try out.append("// ===== ");
            try out.append(ex.kind.name());
            try out.append(" =====\n");
            try upstream.writeMlirForExample(&out, ex.kind);
        }
        std.debug.print("{s}", .{out.slice()});
        return;
    }
    var out: mlir.TextBuffer(120000) = .{};
    try upstream.writeMarkdownReport(&out);
    std.debug.print("{s}", .{out.slice()});
}
