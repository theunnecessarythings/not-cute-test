const std = @import("std");
const layout = @import("layout.zig");
const core = @import("core.zig");

pub const Scalar = layout.Scalar;
pub const Unsigned = layout.Unsigned;

pub const Error = layout.Error || core.Error || error{
    InvalidRank,
    InvalidExtent,
    InvalidStride,
    InvalidCoordinate,
    InvalidBroadcast,
    InvalidDynamicValue,
    Overflow,
};

pub const max_rank = 8;

pub const Extent = union(enum) {
    static: Unsigned,
    dynamic: []const u8,

    pub fn validate(self: Extent) Error!void {
        switch (self) {
            .static => |v| if (v == 0) return Error.InvalidExtent,
            .dynamic => |name| if (name.len == 0) return Error.InvalidDynamicValue,
        }
    }

    pub fn known(self: Extent) ?Unsigned {
        return switch (self) {
            .static => |v| v,
            .dynamic => null,
        };
    }

    pub fn isOne(self: Extent) bool {
        return if (self.known()) |v| v == 1 else false;
    }

    pub fn write(self: Extent, out: anytype) !void {
        switch (self) {
            .static => |v| try out.appendUnsigned(v),
            .dynamic => |name| try out.append(name),
        }
    }
};

pub const Shape = struct {
    extents: [max_rank]Extent = undefined,
    len: usize = 0,

    pub fn init(extents: []const Extent) Error!Shape {
        if (extents.len == 0 or extents.len > max_rank) return Error.InvalidRank;
        var s: Shape = .{};
        for (extents) |e| {
            try e.validate();
            s.extents[s.len] = e;
            s.len += 1;
        }
        return s;
    }

    pub fn staticInts(values: []const Unsigned) Error!Shape {
        var tmp: [max_rank]Extent = undefined;
        if (values.len > max_rank) return Error.InvalidRank;
        for (values, 0..) |v, i| tmp[i] = .{ .static = v };
        return init(tmp[0..values.len]);
    }

    pub fn rank(self: Shape) usize {
        return self.len;
    }

    pub fn at(self: Shape, i: usize) Error!Extent {
        if (i >= self.len) return Error.InvalidRank;
        return self.extents[i];
    }

    pub fn staticSize(self: Shape) Error!Unsigned {
        var p: Unsigned = 1;
        for (self.extents[0..self.len]) |e| {
            const v = e.known() orelse return Error.InvalidDynamicValue;
            p = std.math.mul(Unsigned, p, v) catch return Error.Overflow;
        }
        return p;
    }

    pub fn rowMajorStrides(self: Shape) Error!Stride {
        var out: Stride = .{ .len = self.len };
        var running: Scalar = 1;
        var i = self.len;
        while (i > 0) {
            i -= 1;
            out.values[i] = running;
            const v = self.extents[i].known() orelse return Error.InvalidDynamicValue;
            running = std.math.mul(
                Scalar,
                running,
                @intCast(v),
            ) catch return Error.Overflow;
        }
        return out;
    }

    pub fn colMajorStrides(self: Shape) Error!Stride {
        var out: Stride = .{ .len = self.len };
        var running: Scalar = 1;
        for (0..self.len) |i| {
            out.values[i] = running;
            const v = self.extents[i].known() orelse return Error.InvalidDynamicValue;
            running = std.math.mul(
                Scalar,
                running,
                @intCast(v),
            ) catch return Error.Overflow;
        }
        return out;
    }

    pub fn broadcastWith(self: Shape, rhs: Shape) Error!Shape {
        const out_rank = @max(self.len, rhs.len);
        var out: [max_rank]Extent = undefined;
        for (0..out_rank) |k| {
            const li = if (k + self.len >= out_rank) k + self.len - out_rank else max_rank;
            const ri = if (k + rhs.len >= out_rank) k + rhs.len - out_rank else max_rank;
            const a: Extent = if (li < self.len) self.extents[li] else .{ .static = 1 };
            const b: Extent = if (ri < rhs.len) rhs.extents[ri] else .{ .static = 1 };
            out[k] = try broadcastExtent(a, b);
        }
        return Shape.init(out[0..out_rank]);
    }

    pub fn write(self: Shape, out: anytype) !void {
        try out.append("(");
        for (self.extents[0..self.len], 0..) |e, i| {
            if (i != 0) try out.append(",");
            try e.write(out);
        }
        try out.append(")");
    }
};

pub const Stride = struct {
    values: [max_rank]Scalar = undefined,
    len: usize = 0,

    pub fn init(values: []const Scalar) Error!Stride {
        if (values.len == 0 or values.len > max_rank) return Error.InvalidRank;
        var s: Stride = .{};
        for (values) |v| {
            if (v < 0) return Error.InvalidStride;
            s.values[s.len] = v;
            s.len += 1;
        }
        return s;
    }

    pub fn write(self: Stride, out: anytype) !void {
        try out.append("(");
        for (self.values[0..self.len], 0..) |v, i| {
            if (i != 0) try out.append(",");
            try out.appendSigned(v);
        }
        try out.append(")");
    }
};

pub const Coord = struct {
    values: [max_rank]Unsigned = undefined,
    len: usize = 0,

    pub fn init(values: []const Unsigned) Error!Coord {
        if (values.len == 0 or values.len > max_rank) return Error.InvalidRank;
        var c: Coord = .{};
        for (values) |v| {
            c.values[c.len] = v;
            c.len += 1;
        }
        return c;
    }

    pub fn validateFor(self: Coord, shape: Shape) Error!void {
        if (self.len != shape.len) return Error.InvalidRank;
        for (0..self.len) |i| {
            const extent = shape.extents[i].known() orelse continue;
            if (self.values[i] >= extent) return Error.InvalidCoordinate;
        }
    }
};

pub const LayoutView = struct {
    shape: Shape,
    stride: Stride,

    pub fn init(shape: Shape, stride: Stride) Error!LayoutView {
        if (shape.len != stride.len) return Error.InvalidRank;
        return .{ .shape = shape, .stride = stride };
    }

    pub fn rowMajor(shape: Shape) Error!LayoutView {
        return init(shape, try shape.rowMajorStrides());
    }
    pub fn colMajor(shape: Shape) Error!LayoutView {
        return init(shape, try shape.colMajorStrides());
    }

    pub fn linearize(self: LayoutView, coord: Coord) Error!Unsigned {
        try coord.validateFor(self.shape);
        var offset: Scalar = 0;
        for (0..coord.len) |i| {
            const term = std.math.mul(
                Scalar,
                @intCast(coord.values[i]),
                self.stride.values[i],
            ) catch return Error.Overflow;
            offset = std.math.add(Scalar, offset, term) catch return Error.Overflow;
        }
        if (offset < 0) return Error.InvalidCoordinate;
        return @intCast(offset);
    }

    pub fn delinearizeRowMajor(self: LayoutView, index: Unsigned) Error!Coord {
        const size = try self.shape.staticSize();
        if (index >= size) return Error.InvalidCoordinate;
        var c: Coord = .{ .len = self.shape.len };
        var rem = index;
        var i: usize = 0;
        while (i < self.shape.len) : (i += 1) {
            const extent = self.shape.extents[i].known() orelse return Error.InvalidDynamicValue;
            const stride: Unsigned = @intCast(self.stride.values[i]);
            if (stride == 0) return Error.InvalidStride;
            c.values[i] = rem / stride;
            rem %= stride;
            if (c.values[i] >= extent) c.values[i] = extent - 1;
        }
        return c;
    }

    pub fn subView(self: LayoutView, fixed_prefix: []const Unsigned) Error!SubView {
        if (fixed_prefix.len > self.shape.len) return Error.InvalidRank;
        var offset: Scalar = 0;
        for (fixed_prefix, 0..) |v, i| {
            const extent = self.shape.extents[i].known() orelse return Error.InvalidDynamicValue;
            if (v >= extent) return Error.InvalidCoordinate;
            const term = std.math.mul(
                Scalar,
                @intCast(v),
                self.stride.values[i],
            ) catch return Error.Overflow;
            offset = std.math.add(Scalar, offset, term) catch return Error.Overflow;
        }
        var new_shape: [max_rank]Extent = undefined;
        var new_stride: [max_rank]Scalar = undefined;
        var len: usize = 0;
        for (fixed_prefix.len..self.shape.len) |i| {
            new_shape[len] = self.shape.extents[i];
            new_stride[len] = self.stride.values[i];
            len += 1;
        }
        return .{
            .offset = @intCast(offset),
            .view = try LayoutView.init(
                try Shape.init(new_shape[0..len]),
                try Stride.init(new_stride[0..len]),
            ),
        };
    }
};

pub const SubView = struct { offset: Unsigned, view: LayoutView };

fn broadcastExtent(a: Extent, b: Extent) Error!Extent {
    if (a.isOne()) return b;
    if (b.isOne()) return a;
    if (a.known()) |av| {
        if (b.known()) |bv| {
            if (av == bv) return a;
            return Error.InvalidBroadcast;
        }
    }
    if (a.known() == null and b.known() == null) {
        // Without a symbolic equality engine, only identical variable names are congruent.
        if (std.mem.eql(u8, a.dynamic, b.dynamic)) return a;
    }
    return Error.InvalidBroadcast;
}

pub fn ceilDivUnsigned(a: Unsigned, b: Unsigned) Error!Unsigned {
    if (b == 0) return Error.InvalidExtent;
    return (a + b - 1) / b;
}

pub fn roundUpUnsigned(a: Unsigned, b: Unsigned) Error!Unsigned {
    return (try ceilDivUnsigned(a, b)) * b;
}

pub fn tileShape(shape: Shape, tile: Shape) Error!Shape {
    if (shape.len != tile.len) return Error.InvalidRank;
    var out: [max_rank]Extent = undefined;
    for (0..shape.len) |i| {
        const a = shape.extents[i].known() orelse return Error.InvalidDynamicValue;
        const b = tile.extents[i].known() orelse return Error.InvalidDynamicValue;
        out[i] = .{ .static = try ceilDivUnsigned(a, b) };
    }
    return Shape.init(out[0..shape.len]);
}

test "semantics: static layout linearization and subview offsets" {
    const shape = try Shape.staticInts(&.{ 3, 4, 5 });
    const view = try LayoutView.rowMajor(shape);
    const coord = try Coord.init(&.{ 1, 2, 3 });
    try std.testing.expectEqual(@as(Unsigned, 33), try view.linearize(coord));
    const sub = try view.subView(&.{ 1, 2 });
    try std.testing.expectEqual(@as(Unsigned, 30), sub.offset);
    try std.testing.expectEqual(@as(usize, 1), sub.view.shape.rank());
}

test "semantics: broadcast and tile shape handle edge cases" {
    const a = try Shape.staticInts(&.{ 2, 1, 4 });
    const b = try Shape.staticInts(&.{ 1, 3, 4 });
    const c = try a.broadcastWith(b);
    try std.testing.expectEqual(@as(Unsigned, 24), try c.staticSize());
    const tiled = try tileShape(
        try Shape.staticInts(&.{ 17, 33 }),
        try Shape.staticInts(&.{ 8, 16 }),
    );
    try std.testing.expectEqual(@as(Unsigned, 9), try tiled.staticSize());
}

pub const ProductScan = struct {
    values: [max_rank]Unsigned = undefined,
    len: usize = 0,

    pub fn at(self: ProductScan, i: usize) Error!Unsigned {
        if (i >= self.len) return Error.InvalidRank;
        return self.values[i];
    }
};

pub fn prefixProducts(shape: Shape) Error!ProductScan {
    var out: ProductScan = .{ .len = shape.len };
    var running: Unsigned = 1;
    for (0..shape.len) |i| {
        out.values[i] = running;
        const extent = shape.extents[i].known() orelse return Error.InvalidDynamicValue;
        running = std.math.mul(Unsigned, running, extent) catch return Error.Overflow;
    }
    return out;
}

pub fn suffixProducts(shape: Shape) Error!ProductScan {
    var out: ProductScan = .{ .len = shape.len };
    var running: Unsigned = 1;
    var i = shape.len;
    while (i > 0) {
        i -= 1;
        out.values[i] = running;
        const extent = shape.extents[i].known() orelse return Error.InvalidDynamicValue;
        running = std.math.mul(Unsigned, running, extent) catch return Error.Overflow;
    }
    return out;
}

pub fn incrementCoord(coord: *Coord, shape: Shape) Error!bool {
    if (coord.len != shape.len) return Error.InvalidRank;
    var i = coord.len;
    while (i > 0) {
        i -= 1;
        const extent = shape.extents[i].known() orelse return Error.InvalidDynamicValue;
        coord.values[i] += 1;
        if (coord.values[i] < extent) return true;
        coord.values[i] = 0;
    }
    return false;
}

pub fn forEachStaticCoord(shape: Shape, visitor: anytype) Error!usize {
    var zeros: [max_rank]Unsigned = [_]Unsigned{0} ** max_rank;
    var coord = try Coord.init(zeros[0..shape.len]);
    var count: usize = 0;
    while (true) {
        try visitor(coord);
        count += 1;
        if (!try incrementCoord(&coord, shape)) break;
    }
    return count;
}

pub fn isCompactRowMajor(view: LayoutView) Error!bool {
    const expected = try view.shape.rowMajorStrides();
    return std.mem.eql(
        Scalar,
        view.stride.values[0..view.stride.len],
        expected.values[0..expected.len],
    );
}

pub fn isCompactColMajor(view: LayoutView) Error!bool {
    const expected = try view.shape.colMajorStrides();
    return std.mem.eql(
        Scalar,
        view.stride.values[0..view.stride.len],
        expected.values[0..expected.len],
    );
}

pub fn cosize(view: LayoutView) Error!Unsigned {
    var max_offset: Scalar = 0;
    for (0..view.shape.len) |i| {
        const extent = view.shape.extents[i].known() orelse return Error.InvalidDynamicValue;
        if (extent == 0) return Error.InvalidExtent;
        const term = std.math.mul(
            Scalar,
            @intCast(extent - 1),
            view.stride.values[i],
        ) catch return Error.Overflow;
        max_offset = std.math.add(Scalar, max_offset, term) catch return Error.Overflow;
    }
    if (max_offset < 0) return Error.InvalidStride;
    return @as(Unsigned, @intCast(max_offset)) + 1;
}

pub const TiledIndex = struct {
    tile_coord: Coord,
    residue_coord: Coord,
};

pub fn splitCoordByTile(coord: Coord, tile: Shape) Error!TiledIndex {
    if (coord.len != tile.len) return Error.InvalidRank;
    var tile_values: [max_rank]Unsigned = undefined;
    var residue_values: [max_rank]Unsigned = undefined;
    for (0..coord.len) |i| {
        const t = tile.extents[i].known() orelse return Error.InvalidDynamicValue;
        if (t == 0) return Error.InvalidExtent;
        tile_values[i] = coord.values[i] / t;
        residue_values[i] = coord.values[i] % t;
    }
    return .{
        .tile_coord = try Coord.init(tile_values[0..coord.len]),
        .residue_coord = try Coord.init(residue_values[0..coord.len]),
    };
}

pub const LocalTile = struct {
    tile_shape: Shape,
    tile_count_shape: Shape,
    tile_coord: Coord,
    base_offset: Unsigned,
};

pub fn localTile(
    view: LayoutView,
    tile_shape: Shape,
    tile_coord: Coord,
) Error!LocalTile {
    if (view.shape.len != tile_shape.len or tile_coord.len != tile_shape.len)
        return Error.InvalidRank;
    const counts = try tileShape(view.shape, tile_shape);
    try tile_coord.validateFor(counts);
    var base: Scalar = 0;
    for (0..tile_shape.len) |i| {
        const tile_extent = tile_shape.extents[i].known() orelse return Error.InvalidDynamicValue;
        const coord = std.math.mul(
            Unsigned,
            tile_coord.values[i],
            tile_extent,
        ) catch return Error.Overflow;
        const term = std.math.mul(
            Scalar,
            @intCast(coord),
            view.stride.values[i],
        ) catch return Error.Overflow;
        base = std.math.add(Scalar, base, term) catch return Error.Overflow;
    }
    if (base < 0) return Error.InvalidCoordinate;
    return .{
        .tile_shape = tile_shape,
        .tile_count_shape = counts,
        .tile_coord = tile_coord,
        .base_offset = @intCast(base),
    };
}

pub fn logicalDivideShape(shape: Shape, tile: Shape) Error!Shape {
    const outer = try tileShape(shape, tile);
    var ext: [max_rank]Extent = undefined;
    if (shape.len * 2 > max_rank) return Error.InvalidRank;
    for (0..shape.len) |i| ext[i] = tile.extents[i];
    for (0..shape.len) |i| ext[shape.len + i] = outer.extents[i];
    return Shape.init(ext[0 .. shape.len * 2]);
}

pub fn flatDivideShape(shape: Shape, tile: Shape) Error!Shape {
    const divided = try logicalDivideShape(shape, tile);
    var static_values: [max_rank]Unsigned = undefined;
    for (0..divided.len) |i| static_values[i] = divided.extents[i].known() orelse return Error.InvalidDynamicValue;
    return Shape.staticInts(static_values[0..divided.len]);
}

test "semantics: product scans and coordinate iteration match row-major expectations" {
    const shape = try Shape.staticInts(&.{ 2, 3, 4 });
    const prefix = try prefixProducts(shape);
    const suffix = try suffixProducts(shape);
    try std.testing.expectEqual(@as(Unsigned, 6), try prefix.at(2));
    try std.testing.expectEqual(@as(Unsigned, 4), try suffix.at(1));
    var coord = try Coord.init(&.{ 0, 0, 0 });
    var visits: usize = 1;
    while (try incrementCoord(&coord, shape)) visits += 1;
    try std.testing.expectEqual(@as(usize, 24), visits);
}

test "semantics: compactness, cosize, local tile and divide shapes" {
    const shape = try Shape.staticInts(&.{ 4, 8 });
    const row = try LayoutView.rowMajor(shape);
    const col = try LayoutView.colMajor(shape);
    try std.testing.expect(try isCompactRowMajor(row));
    try std.testing.expect(try isCompactColMajor(col));
    try std.testing.expectEqual(@as(Unsigned, 32), try cosize(row));
    const tile = try Shape.staticInts(&.{ 2, 4 });
    const local = try localTile(row, tile, try Coord.init(&.{ 1, 1 }));
    try std.testing.expectEqual(@as(Unsigned, 20), local.base_offset);
    const divided = try logicalDivideShape(shape, tile);
    try std.testing.expectEqual(@as(usize, 4), divided.rank());
    const split = try splitCoordByTile(try Coord.init(&.{ 3, 7 }), tile);
    try std.testing.expectEqual(@as(Unsigned, 1), split.tile_coord.values[0]);
    try std.testing.expectEqual(@as(Unsigned, 3), split.residue_coord.values[1]);
}
