const std = @import("std");
const kernel_builders = @import("kernel_builders.zig");
const mlir = @import("mlir_text.zig");

pub fn main() !void {
    var out: mlir.TextBuffer(120000) = .{};
    try kernel_builders.writeAllKernels(&out);
    std.debug.print("{s}", .{out.slice()});
}
