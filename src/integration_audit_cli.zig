const std = @import("std");
const integration_audit = @import("integration_audit.zig");
const mlir = @import("mlir_text.zig");

pub fn main(_: std.process.Init) !void {
    var out: mlir.TextBuffer(2048) = .{};
    try integration_audit.writeStatus(&out);
    std.debug.print("{s}", .{out.slice()});
}
