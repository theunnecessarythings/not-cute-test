const std = @import("std");
const cutlass_emit = @import("cutlass_emit.zig");
const mlir = @import("mlir_text.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const cmd = args.next() orelse "status";

    if (std.mem.eql(u8, cmd, "status")) {
        var buf: mlir.TextBuffer(2048) = .{};
        try cutlass_emit.writeStatus(&buf);
        std.debug.print("{s}", .{buf.slice()});
    } else if (std.mem.eql(u8, cmd, "list")) {
        for (cutlass_emit.fixtures) |fixture| {
            std.debug.print("{s}\t{s}\n", .{ fixture.name, @tagName(fixture.kind) });
        }
    } else if (std.mem.eql(u8, cmd, "emit")) {
        const name = args.next() orelse return error.MissingFixtureName;
        const fixture = cutlass_emit.fixtureByName(name) orelse return error.UnknownFixtureName;
        std.debug.print("{s}", .{fixture.mlir_text});
    } else if (std.mem.eql(u8, cmd, "emit-all")) {
        var buf: mlir.TextBuffer(12000) = .{};
        try cutlass_emit.writeAllFixtures(&buf);
        std.debug.print("{s}", .{buf.slice()});
    } else {
        return error.UnknownCommand;
    }
}
