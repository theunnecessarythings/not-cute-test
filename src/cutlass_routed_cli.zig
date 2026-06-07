const std = @import("std");
const cutlass_routed = @import("cutlass_routed.zig");
const mlir = @import("mlir_text.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const cmd = args.next() orelse "status";

    if (std.mem.eql(u8, cmd, "status")) {
        var buf: mlir.TextBuffer(2048) = .{};
        try cutlass_routed.writeStatus(&buf);
        std.debug.print("{s}", .{buf.slice()});
    } else if (std.mem.eql(u8, cmd, "list")) {
        for (cutlass_routed.routed_fixtures) |fixture| {
            std.debug.print("{s}\t{s}\n", .{ fixture.name, @tagName(fixture.kind) });
        }
    } else if (std.mem.eql(u8, cmd, "emit")) {
        const name = args.next() orelse return error.MissingFixtureName;
        var buf: mlir.TextBuffer(12000) = .{};
        try cutlass_routed.emitByName(name, &buf);
        std.debug.print("{s}", .{buf.slice()});
    } else if (std.mem.eql(u8, cmd, "emit-all")) {
        var buf: mlir.TextBuffer(24000) = .{};
        try cutlass_routed.writeAllGenerated(&buf);
        std.debug.print("{s}", .{buf.slice()});
    } else {
        return error.UnknownCommand;
    }
}
