const std = @import("std");
const not_cute = @import("not_cute");

pub fn main() !void {
    var out: not_cute.mlir_text.TextBuffer(60000) = .{};
    try not_cute.upstream_parity.writeMlirForExample(&out, .hello_world);
    std.debug.print("{s}", .{out.slice()});
}
