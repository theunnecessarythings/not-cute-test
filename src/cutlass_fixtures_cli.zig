const std = @import("std");
const cutlass_fixtures = @import("cutlass_fixtures.zig");
const mlir = @import("mlir_text.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const cmd = args.next() orelse "status";

    if (std.mem.eql(u8, cmd, "status")) {
        var buf: mlir.TextBuffer(2048) = .{};
        try cutlass_fixtures.writeStatus(&buf);
        std.debug.print("{s}", .{buf.slice()});
    } else if (std.mem.eql(u8, cmd, "list")) {
        for (cutlass_fixtures.fixtures) |fixture| {
            std.debug.print("{s}\t{s}\tfail={}\n", .{ fixture.name, @tagName(fixture.kind), fixture.expect_parse_failure });
        }
    } else if (std.mem.eql(u8, cmd, "emit")) {
        const name = args.next() orelse return error.MissingFixtureName;
        const fixture = cutlass_fixtures.fixtureByName(name) orelse return error.UnknownFixtureName;
        std.debug.print("{s}", .{fixture.mlir_text});
    } else {
        return error.UnknownCommand;
    }
}
