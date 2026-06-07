const std = @import("std");
const tiled_emit = @import("tiled_emit.zig");
const mlir = @import("mlir_text.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const cmd = args.next() orelse "status";

    if (std.mem.eql(u8, cmd, "status")) {
        var buf: mlir.TextBuffer(2048) = .{};
        try tiled_emit.writeStatus(&buf);
        std.debug.print("{s}", .{buf.slice()});
    } else if (std.mem.eql(u8, cmd, "list")) {
        for (tiled_emit.full_tiled_fixtures) |fixture| {
            std.debug.print("{s}\t{s}\n", .{ fixture.name, @tagName(fixture.kind) });
        }
    } else if (std.mem.eql(u8, cmd, "emit")) {
        const name = args.next() orelse return error.MissingFixtureName;
        var buf: mlir.TextBuffer(16000) = .{};
        try tiled_emit.emitByName(name, &buf);
        std.debug.print("{s}", .{buf.slice()});
    } else if (std.mem.eql(u8, cmd, "emit-all")) {
        var buf: mlir.TextBuffer(32000) = .{};
        try tiled_emit.writeAllGenerated(&buf);
        std.debug.print("{s}", .{buf.slice()});
    } else {
        return error.UnknownCommand;
    }
}
