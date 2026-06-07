const std = @import("std");
const cute = @import("not_cute");

pub fn main() !void {
    const matrix = cute.layout.makeLayout(.{ 4, 8 }, .{ 8, 1 });
    const coord = cute.layout.Tree.fromComptime(.{ 2, 3 });
    const row_selector = cute.layout_algebra.Selector.fromComptime(.{ 2, cute.layout_algebra.keep });
    const row = try cute.layout_algebra.sliceAndOffset(&matrix, &row_selector);
    const row_shape = try row.layout.shape.flattenLeaves();
    const row_stride = try row.layout.stride.flattenLeaves();

    std.debug.print(
        \\matrix rank={} size={} cosize={}
        \\coord (2, 3) -> linear index {}
        \\row 2 -> offset {}, shape {}, stride {}
        \\
    , .{
        matrix.rank(),
        try matrix.size(),
        try matrix.cosize(),
        try matrix.crd2idx(coord),
        row.offset,
        row_shape.at(0),
        row_stride.at(0),
    });
}
