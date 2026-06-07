const std = @import("std");
const layout = @import("layout.zig");
const tuple = @import("tuple.zig");
const core = @import("core_static.zig");

pub const Scalar = layout.Scalar;
pub const Unsigned = layout.Unsigned;
pub const Tree = layout.Tree;
pub const Layout = layout.Layout;
pub const Flat = layout.Flat;
pub const Error = core.Error || error{ InvalidSelector, NegativeIndex };
pub const max_nodes = layout.max_nodes;
pub const max_children = layout.max_children;
pub const max_leaves = layout.max_leaves;

pub const Keep = enum { keep };
pub const keep: Keep = .keep;

pub const SelectorLeaf = union(enum) {
    keep,
    fixed: Scalar,
};

pub const SelectorNode = union(enum) {
    leaf: SelectorLeaf,
    tuple: layout.Span,
};

/// Static coordinate/dicer profile used by layout slicing and dicing.
/// `keep` corresponds to CuteDSL's `None`/underscore convention in a slicing
/// coordinate: slice keeps modes tagged `keep`, dice drops modes tagged `keep`.
/// Integer leaves are fixed coordinates.
pub const Selector = struct {
    const Self = @This();

    nodes: layout.BoundedList(SelectorNode, max_nodes) = .{},
    children: layout.BoundedList(u16, max_children) = .{},
    root: u16 = 0,

    pub fn initKeep() Error!Self {
        var out: Self = .{};
        try out.nodes.append(.{ .leaf = .keep });
        return out;
    }

    pub fn initFixed(value: Scalar) Error!Self {
        var out: Self = .{};
        try out.nodes.append(.{ .leaf = .{ .fixed = value } });
        return out;
    }

    pub fn initTuple(parts: []const Self) Error!Self {
        if (parts.len > max_children) return Error.OutOfCapacity;
        var out: Self = .{};
        var child_ids: [max_children]u16 = undefined;
        for (parts, 0..) |*part, i| child_ids[i] = try out.copySubtreeFrom(
            part,
            part.root,
        );
        const child_start = out.children.len;
        for (child_ids[0..parts.len]) |child_id| try out.children.append(child_id);
        const root_index = out.nodes.len;
        try out.nodes.append(.{ .tuple = .{ .start = child_start, .len = parts.len } });
        out.root = @intCast(root_index);
        return out;
    }

    pub fn fromComptime(comptime spec: anytype) Self {
        return comptime blk: {
            var out: Self = .{};
            out.root = out.addComptime(spec) catch |err| switch (err) {
                Error.OutOfCapacity => @compileError("selector literal exceeds bounded capacity"),
                else => @compileError("invalid selector literal"),
            };
            break :blk out;
        };
    }

    fn addComptime(self: *Self, comptime spec: anytype) Error!u16 {
        const T = @TypeOf(spec);
        switch (@typeInfo(T)) {
            .comptime_int, .int => {
                const index = self.nodes.len;
                try self.nodes.append(.{ .leaf = .{ .fixed = @as(Scalar, spec) } });
                return @intCast(index);
            },
            .@"enum" => {
                if (@TypeOf(spec) != Keep or spec != .keep) @compileError("selector enum leaves must be layout_algebra.keep");
                const index = self.nodes.len;
                try self.nodes.append(.{ .leaf = .keep });
                return @intCast(index);
            },
            .@"struct" => |info| {
                if (!info.is_tuple) @compileError("selector literals must be ints, layout_algebra.keep, or tuple literals");
                var child_ids: [info.fields.len]u16 = undefined;
                inline for (info.fields, 0..) |field, i| child_ids[i] = try self.addComptime(@field(spec, field.name));
                const child_start = self.children.len;
                for (child_ids) |child_id| try self.children.append(child_id);
                const index = self.nodes.len;
                try self.nodes.append(.{
                    .tuple = .{ .start = child_start, .len = info.fields.len },
                });
                return @intCast(index);
            },
            .array => |info| {
                var child_ids: [info.len]u16 = undefined;
                inline for (0..info.len) |i| child_ids[i] = try self.addComptime(spec[i]);
                const child_start = self.children.len;
                for (child_ids) |child_id| try self.children.append(child_id);
                const index = self.nodes.len;
                try self.nodes.append(.{
                    .tuple = .{ .start = child_start, .len = info.len },
                });
                return @intCast(index);
            },
            else => @compileError("selector leaves must be ints or layout_algebra.keep"),
        }
    }

    fn copySubtreeFrom(self: *Self, src: *const Self, src_id: u16) Error!u16 {
        switch (src.nodes.at(src_id)) {
            .leaf => |value| {
                const index = self.nodes.len;
                try self.nodes.append(.{ .leaf = value });
                return @intCast(index);
            },
            .tuple => |span| {
                var child_ids: [max_children]u16 = undefined;
                if (span.len > max_children) return Error.OutOfCapacity;
                for (0..span.len) |i| child_ids[i] = try self.copySubtreeFrom(
                    src,
                    src.children.at(span.start + i),
                );
                const child_start = self.children.len;
                for (child_ids[0..span.len]) |child_id| try self.children.append(child_id);
                const index = self.nodes.len;
                try self.nodes.append(.{
                    .tuple = .{ .start = child_start, .len = span.len },
                });
                return @intCast(index);
            },
        }
    }
};

pub const SliceResult = struct {
    layout: Layout,
    offset: Scalar,
};

pub fn groupModes(input: *const Tree, begin_raw: isize, end_raw: ?isize) Error!Tree {
    const r = input.rank();
    const range = try normalizeRange(begin_raw, end_raw, r);
    var parts: [max_children]Tree = undefined;
    var count: usize = 0;

    for (0..range.begin) |i| try appendTreePart(&parts, &count, try input.topMode(i));

    var grouped_parts: [max_children]Tree = undefined;
    var grouped_count: usize = 0;
    for (range.begin..range.end) |i| try appendTreePart(
        &grouped_parts,
        &grouped_count,
        try input.topMode(i),
    );
    try appendTreePart(
        &parts,
        &count,
        try Tree.initTuple(grouped_parts[0..grouped_count]),
    );

    for (range.end..r) |i| try appendTreePart(&parts, &count, try input.topMode(i));
    return Tree.initTuple(parts[0..count]);
}

pub fn groupModesLayout(
    input: *const Layout,
    begin_raw: isize,
    end_raw: ?isize,
) Error!Layout {
    return Layout.init(
        try groupModes(&input.shape, begin_raw, end_raw),
        try groupModes(&input.stride, begin_raw, end_raw),
    );
}

pub fn selectModes(input: *const Tree, modes: []const usize) Error!Tree {
    var parts: [max_children]Tree = undefined;
    var count: usize = 0;
    for (modes) |mode| try appendTreePart(&parts, &count, try input.topMode(mode));
    return Tree.initTuple(parts[0..count]);
}

pub fn selectModesLayout(input: *const Layout, modes: []const usize) Error!Layout {
    return Layout.init(
        try selectModes(&input.shape, modes),
        try selectModes(&input.stride, modes),
    );
}

pub fn sliceTree(src: *const Tree, selector: *const Selector) Error!Tree {
    var parts: [max_children]Tree = undefined;
    var count: usize = 0;
    try collectTreeBySelector(
        src,
        src.root,
        selector,
        selector.root,
        .slice,
        &parts,
        &count,
    );
    return Tree.initTuple(parts[0..count]);
}

pub fn diceTree(src: *const Tree, selector: *const Selector) Error!Tree {
    var parts: [max_children]Tree = undefined;
    var count: usize = 0;
    try collectTreeBySelector(
        src,
        src.root,
        selector,
        selector.root,
        .dice,
        &parts,
        &count,
    );
    return Tree.initTuple(parts[0..count]);
}

pub fn sliceAndOffset(src: *const Layout, selector: *const Selector) Error!SliceResult {
    var shape_parts: [max_children]Tree = undefined;
    var stride_parts: [max_children]Tree = undefined;
    var count: usize = 0;
    var offset: Scalar = 0;
    try collectLayoutSlice(
        src,
        src.shape.root,
        src.stride.root,
        selector,
        selector.root,
        &shape_parts,
        &stride_parts,
        &count,
        &offset,
    );
    const shape = try treeFromSelectedParts(shape_parts[0..count], true);
    const stride = try treeFromSelectedParts(stride_parts[0..count], false);
    return .{ .layout = try Layout.init(shape, stride), .offset = offset };
}

pub fn sliceLayout(src: *const Layout, selector: *const Selector) Error!Layout {
    return (try sliceAndOffset(src, selector)).layout;
}

pub fn diceLayout(src: *const Layout, selector: *const Selector) Error!Layout {
    var shape_parts: [max_children]Tree = undefined;
    var stride_parts: [max_children]Tree = undefined;
    var count: usize = 0;
    try collectLayoutDice(
        src,
        src.shape.root,
        src.stride.root,
        selector,
        selector.root,
        &shape_parts,
        &stride_parts,
        &count,
    );
    return Layout.init(
        try treeFromSelectedParts(shape_parts[0..count], true),
        try treeFromSelectedParts(stride_parts[0..count], false),
    );
}

pub fn shapeOf(input: *const Layout, mode: ?usize) Error!Tree {
    if (mode) |m| return input.shape.topMode(m);
    return input.shape;
}

pub fn strideOf(input: *const Layout, mode: ?usize) Error!Tree {
    if (mode) |m| return input.stride.topMode(m);
    return input.stride;
}

pub fn sizeOf(input: *const Layout, mode: ?usize) Error!Unsigned {
    const s = try shapeOf(input, mode);
    return s.product();
}

pub fn cosizeOf(input: *const Layout, mode: ?usize) Error!Unsigned {
    if (mode) |m| {
        const sub = try input.topMode(m);
        return sub.cosize();
    }
    return input.cosize();
}

pub fn idx2crdShape(idx: Scalar, shape: *const Tree) Error!Tree {
    if (idx < 0) return Error.CoordinateOutOfBounds;
    const shapes = try shape.flattenLeaves();
    var coords: Flat = .{};
    var remaining = idx;
    for (shapes.slice()) |extent| {
        if (extent <= 0) return Error.InvalidShape;
        try coords.append(@mod(remaining, extent));
        remaining = @divFloor(remaining, extent);
    }
    if (remaining != 0) return Error.CoordinateOutOfBounds;
    return Tree.fromProfileAndLeaves(shape, coords.slice());
}

pub fn incrementCoord(coord: *const Tree, shape: *const Tree) Error!Tree {
    if (!coord.sameProfile(shape)) return Error.ProfileMismatch;
    const cf = try coord.flattenLeaves();
    const sf = try shape.flattenLeaves();
    var out = cf;
    var carry = true;
    for (0..out.len) |i| {
        const extent = sf.at(i);
        if (extent <= 0) return Error.InvalidShape;
        if (out.at(i) < 0 or out.at(i) >= extent) return Error.CoordinateOutOfBounds;
        if (!carry) continue;
        const next = out.at(i) + 1;
        if (next == extent) {
            out.set(i, 0);
            carry = true;
        } else {
            out.set(i, next);
            carry = false;
        }
    }
    return Tree.fromProfileAndLeaves(shape, out.slice());
}

pub fn prependLayout(
    input: *const Layout,
    elem: Layout,
    up_to_rank: ?usize,
) Error!Layout {
    const target = up_to_rank orelse (input.rank() + 1);
    if (target < input.rank()) return Error.InvalidSelection;
    var parts: [max_children]Layout = undefined;
    var count: usize = 0;
    while (count < target - input.rank()) : (count += 1) parts[count] = elem;
    try appendLayoutTopModes(&parts, &count, input);
    return Layout.concatenate(parts[0..count]);
}

pub fn appendLayout(
    input: *const Layout,
    elem: Layout,
    up_to_rank: ?usize,
) Error!Layout {
    const target = up_to_rank orelse (input.rank() + 1);
    if (target < input.rank()) return Error.InvalidSelection;
    var parts: [max_children]Layout = undefined;
    var count: usize = 0;
    try appendLayoutTopModes(&parts, &count, input);
    while (count < target) : (count += 1) parts[count] = elem;
    return Layout.concatenate(parts[0..count]);
}

pub fn appendOnesLayout(input: *const Layout, up_to_rank: ?usize) Error!Layout {
    return appendLayout(input, layout.makeCompactLayout(1), up_to_rank);
}

pub fn prependOnesLayout(input: *const Layout, up_to_rank: ?usize) Error!Layout {
    return prependLayout(input, layout.makeCompactLayout(1), up_to_rank);
}

pub fn makeLayoutLike(input: *const Layout) Error!Layout {
    return Layout.makeCompact(input.shape);
}

pub fn sizeInBytes(type_bits: Unsigned, maybe_layout: ?*const Layout) Error!Unsigned {
    const l = maybe_layout orelse return 0;
    const bits = std.math.mul(
        Unsigned,
        try l.cosize(),
        type_bits,
    ) catch return Error.Overflow;
    return (bits + 7) / 8;
}

pub fn zippedDivideShape(target: *const Tree, tiler: *const Tree) Error!Tree {
    if (target.rank() < tiler.rank()) return Error.RankMismatch;
    const rest = try core.ceilDiv(target, tiler);
    return Tree.initTuple(&.{ tiler.*, rest });
}

pub fn logicalDivideShape(target: *const Tree, tiler: *const Tree) Error!Tree {
    return zippedDivideShape(target, tiler);
}

pub fn tiledDivideShape(target: *const Tree, tiler: *const Tree) Error!Tree {
    return zippedDivideShape(target, tiler);
}

pub fn flatDivideShape(target: *const Tree, tiler: *const Tree) Error!Tree {
    const z = try zippedDivideShape(target, tiler);
    return core.flatten(&z);
}

pub fn tileToShape(tile: *const Tree, target_profile: *const Tree) Error!Tree {
    if (tile.leafCount() > target_profile.leafCount()) return Error.RankMismatch;
    const flat = try tile.flattenLeaves();
    var padded: Flat = .{};
    for (flat.slice()) |v| try padded.append(v);
    while (padded.len < target_profile.leafCount()) try padded.append(1);
    return Tree.fromProfileAndLeaves(target_profile, padded.slice());
}

const Range = struct { begin: usize, end: usize };

fn normalizeRange(begin_raw: isize, end_raw: ?isize, r: usize) Error!Range {
    const r_i: isize = @intCast(r);
    var begin_i = if (begin_raw < 0) @max(begin_raw + r_i, 0) else begin_raw;
    var end_i = if (end_raw) |e| if (e < 0) e + r_i else e else r_i;
    if (end_i > r_i) end_i = r_i;
    if (begin_i > r_i) begin_i = r_i;
    if (begin_i < 0 or end_i < 0) return Error.InvalidSelection;
    if (begin_i >= end_i) return Error.InvalidSelection;
    return .{ .begin = @intCast(begin_i), .end = @intCast(end_i) };
}

fn appendTreePart(parts: *[max_children]Tree, count: *usize, part: Tree) Error!void {
    if (count.* >= parts.len) return Error.OutOfCapacity;
    parts[count.*] = part;
    count.* += 1;
}

fn appendLayoutPart(
    parts: *[max_children]Layout,
    count: *usize,
    part: Layout,
) Error!void {
    if (count.* >= parts.len) return Error.OutOfCapacity;
    parts[count.*] = part;
    count.* += 1;
}

fn appendLayoutTopModes(
    parts: *[max_children]Layout,
    count: *usize,
    input: *const Layout,
) Error!void {
    switch (input.shape.nodes.at(input.shape.root)) {
        .leaf => try appendLayoutPart(parts, count, input.*),
        .tuple => |span| for (0..span.len) |i| try appendLayoutPart(
            parts,
            count,
            try input.topMode(i),
        ),
    }
}

const CollectMode = enum { slice, dice };

fn collectTreeBySelector(
    src: *const Tree,
    src_id: u16,
    selector: *const Selector,
    selector_id: u16,
    mode: CollectMode,
    parts: *[max_children]Tree,
    count: *usize,
) Error!void {
    switch (selector.nodes.at(selector_id)) {
        .leaf => |leaf| {
            const keep_it = switch (mode) {
                .slice => switch (leaf) {
                    .keep => true,
                    .fixed => false,
                },
                .dice => switch (leaf) {
                    .keep => false,
                    .fixed => true,
                },
            };
            if (keep_it) try appendTreePart(parts, count, try src.subtree(src_id));
        },
        .tuple => |sel_span| {
            const src_span = switch (src.nodes.at(src_id)) {
                .tuple => |span| span,
                .leaf => return Error.ProfileMismatch,
            };
            if (src_span.len != sel_span.len) return Error.ProfileMismatch;
            for (0..sel_span.len) |i| try collectTreeBySelector(
                src,
                src.children.at(src_span.start + i),
                selector,
                selector.children.at(sel_span.start + i),
                mode,
                parts,
                count,
            );
        },
    }
}

fn collectLayoutSlice(
    src: *const Layout,
    shape_id: u16,
    stride_id: u16,
    selector: *const Selector,
    selector_id: u16,
    shape_parts: *[max_children]Tree,
    stride_parts: *[max_children]Tree,
    count: *usize,
    offset: *Scalar,
) Error!void {
    switch (selector.nodes.at(selector_id)) {
        .leaf => |leaf| switch (leaf) {
            .keep => {
                try appendTreePart(shape_parts, count, try src.shape.subtree(shape_id));
                stride_parts[count.* - 1] = try src.stride.subtree(stride_id);
            },
            .fixed => |fixed| {
                const extent = switch (src.shape.nodes.at(shape_id)) {
                    .leaf => |v| v,
                    .tuple => return Error.UnsupportedDynamicOperation,
                };
                const stride_value = switch (src.stride.nodes.at(stride_id)) {
                    .leaf => |v| v,
                    .tuple => return Error.UnsupportedDynamicOperation,
                };
                if (fixed < 0 or fixed >= extent) return Error.CoordinateOutOfBounds;
                const term = std.math.mul(
                    Scalar,
                    fixed,
                    stride_value,
                ) catch return Error.Overflow;
                offset.* = std.math.add(
                    Scalar,
                    offset.*,
                    term,
                ) catch return Error.Overflow;
            },
        },
        .tuple => |sel_span| {
            const shape_span = switch (src.shape.nodes.at(shape_id)) {
                .tuple => |span| span,
                .leaf => return Error.ProfileMismatch,
            };
            const stride_span = switch (src.stride.nodes.at(stride_id)) {
                .tuple => |span| span,
                .leaf => return Error.ProfileMismatch,
            };
            if (shape_span.len != sel_span.len or stride_span.len != sel_span.len)
                return Error.ProfileMismatch;
            for (0..sel_span.len) |i| try collectLayoutSlice(
                src,
                src.shape.children.at(shape_span.start + i),
                src.stride.children.at(stride_span.start + i),
                selector,
                selector.children.at(sel_span.start + i),
                shape_parts,
                stride_parts,
                count,
                offset,
            );
        },
    }
}

fn collectLayoutDice(
    src: *const Layout,
    shape_id: u16,
    stride_id: u16,
    selector: *const Selector,
    selector_id: u16,
    shape_parts: *[max_children]Tree,
    stride_parts: *[max_children]Tree,
    count: *usize,
) Error!void {
    switch (selector.nodes.at(selector_id)) {
        .leaf => |leaf| switch (leaf) {
            .keep => {},
            .fixed => {
                try appendTreePart(shape_parts, count, try src.shape.subtree(shape_id));
                stride_parts[count.* - 1] = try src.stride.subtree(stride_id);
            },
        },
        .tuple => |sel_span| {
            const shape_span = switch (src.shape.nodes.at(shape_id)) {
                .tuple => |span| span,
                .leaf => return Error.ProfileMismatch,
            };
            const stride_span = switch (src.stride.nodes.at(stride_id)) {
                .tuple => |span| span,
                .leaf => return Error.ProfileMismatch,
            };
            if (shape_span.len != sel_span.len or stride_span.len != sel_span.len)
                return Error.ProfileMismatch;
            for (0..sel_span.len) |i| try collectLayoutDice(
                src,
                src.shape.children.at(shape_span.start + i),
                src.stride.children.at(stride_span.start + i),
                selector,
                selector.children.at(sel_span.start + i),
                shape_parts,
                stride_parts,
                count,
            );
        },
    }
}

fn treeFromSelectedParts(parts: []const Tree, comptime is_shape: bool) Error!Tree {
    if (parts.len == 0) return Tree.initLeaf(if (is_shape) 1 else 0);
    if (parts.len == 1) return parts[0];
    return Tree.initTuple(parts);
}

test "layout_algebra: group and select modes for tree and layout" {
    const t = Tree.fromComptime(.{ 2, 3, 4, 5 });
    const g = try groupModes(&t, 1, 3);
    try std.testing.expectEqual(@as(usize, 3), g.rank());
    const middle = try g.topMode(1);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 3, 4 },
        (try middle.flattenLeaves()).slice(),
    );

    const l = layout.makeLayout(.{ 2, 3, 4 }, .{ 1, 2, 6 });
    const s = try selectModesLayout(&l, &.{ 2, 0 });
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 4, 2 },
        (try s.shape.flattenLeaves()).slice(),
    );
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 6, 1 },
        (try s.stride.flattenLeaves()).slice(),
    );
}

test "layout_algebra: slice and dice static tuples" {
    const t = Tree.fromComptime(.{ 2, .{ 3, 4 }, 5 });
    const selector = Selector.fromComptime(.{ keep, .{ 1, keep }, 0 });
    const sliced = try sliceTree(&t, &selector);
    try std.testing.expectEqual(@as(usize, 2), sliced.rank());
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 2, 4 },
        (try sliced.flattenLeaves()).slice(),
    );

    const diced = try diceTree(&t, &selector);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 3, 5 },
        (try diced.flattenLeaves()).slice(),
    );
}

test "layout_algebra: slice layout returns residual layout and offset" {
    const l = layout.makeLayout(.{ 4, 5 }, .{ 5, 1 });
    const selector = Selector.fromComptime(.{ 2, keep });
    const result = try sliceAndOffset(&l, &selector);
    try std.testing.expectEqual(@as(Scalar, 10), result.offset);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{5},
        (try result.layout.shape.flattenLeaves()).slice(),
    );
    try std.testing.expectEqualSlices(
        Scalar,
        &.{1},
        (try result.layout.stride.flattenLeaves()).slice(),
    );
    try std.testing.expectEqual(@as(Scalar, 3), try result.layout.crd2idxFlat(&.{3}));
}

test "layout_algebra: dice layout keeps fixed-tagged modes" {
    const l = layout.makeLayout(.{ 4, 5, 6 }, .{ 30, 6, 1 });
    const selector = Selector.fromComptime(.{ 1, keep, 1 });
    const d = try diceLayout(&l, &selector);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 4, 6 },
        (try d.shape.flattenLeaves()).slice(),
    );
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 30, 1 },
        (try d.stride.flattenLeaves()).slice(),
    );
}

test "layout_algebra: idx2crd shape and increment coord" {
    const shape = Tree.fromComptime(.{ 5, 4 });
    const coord = try idx2crdShape(11, &shape);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 1, 2 },
        (try coord.flattenLeaves()).slice(),
    );

    const c0 = Tree.fromComptime(.{ 2, 0, 0 });
    const s0 = Tree.fromComptime(.{ 3, 3, 3 });
    const c1 = try incrementCoord(&c0, &s0);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 0, 1, 0 },
        (try c1.flattenLeaves()).slice(),
    );
}

test "layout_algebra: append/prepend layouts and size in bytes" {
    const l = layout.makeCompactLayout(.{ 8, 8 });
    const appended = try appendOnesLayout(&l, 4);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 8, 8, 1, 1 },
        (try appended.shape.flattenLeaves()).slice(),
    );
    const prepended = try prependOnesLayout(&l, 3);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 1, 8, 8 },
        (try prepended.shape.flattenLeaves()).slice(),
    );
    try std.testing.expectEqual(@as(Unsigned, 128), try sizeInBytes(16, &l));
}

test "layout_algebra: static tiler shape transforms" {
    const target = Tree.fromComptime(.{ 128, 64, 7 });
    const tiler = Tree.fromComptime(.{ 8, 8 });
    const z = try zippedDivideShape(&target, &tiler);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 8, 8, 16, 8, 7 },
        (try z.flattenLeaves()).slice(),
    );
    const flat = try flatDivideShape(&target, &tiler);
    try std.testing.expectEqual(@as(usize, 5), flat.rank());
}
