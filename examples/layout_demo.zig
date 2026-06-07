const std = @import("std");
const cute = @import("not_cute");

pub fn main() !void {
    var out: cute.mlir_text.TextBuffer(32768) = .{};
    try cute.examples_api.renderExample(.layout_demo, &out);
    std.debug.print("{s}", .{out.slice()});
}
