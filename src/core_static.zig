const std = @import("std");
const layout = @import("layout.zig");
const tuple = @import("tuple.zig");

pub const Scalar = layout.Scalar;
pub const Unsigned = layout.Unsigned;
pub const Error = tuple.Error || error{
    DivisionByZero,
    UnsupportedDynamicOperation,
};

pub const Tree = layout.Tree;
pub const Layout = layout.Layout;
pub const AnyLayout = layout.AnyLayout;
pub const ComposedLayout = layout.ComposedLayout;
pub const Flat = layout.Flat;

pub const RoundMode = enum { compact_left, compact_right };

/// Source-compatible static `static(x)`: Zig already distinguishes comptime data,
/// so this is an identity function for constant-friendly integer values.
pub fn static(value: Scalar) Scalar {
    return value;
}

pub fn isStaticTree(_: *const Tree) bool {
    return true;
}

pub fn depth(input: *const Tree) usize {
    return tuple.depth(input);
}

pub fn rank(input: *const Tree) usize {
    return tuple.rank(input);
}

pub fn isCongruent(lhs: *const Tree, rhs: *const Tree) bool {
    return tuple.isCongruent(lhs, rhs);
}

pub fn isWeaklyCongruent(lhs: *const Tree, rhs: *const Tree) bool {
    return tuple.isWeaklyCongruent(lhs, rhs);
}

pub fn get(input: *const Tree, path: []const usize) Error!Tree {
    var current: Tree = input.*;
    for (path) |mode| current = try current.topMode(mode);
    return current;
}

pub fn select(input: *const Tree, modes: []const usize) Error!Tree {
    if (modes.len == 0) return Error.EmptyTuple;
    var parts: [layout.max_children]Tree = undefined;
    if (modes.len > parts.len) return Error.OutOfCapacity;
    for (modes, 0..) |mode, i| parts[i] = try input.topMode(mode);
    return Tree.initTuple(parts[0..modes.len]);
}

pub fn prepend(input: *const Tree, value: Scalar, target_rank: usize) Error!Tree {
    if (target_rank < input.rank()) return Error.InvalidSelection;
    var parts: [layout.max_children]Tree = undefined;
    var count: usize = 0;
    const missing = target_rank - input.rank();
    if (missing + input.rank() > parts.len) return Error.OutOfCapacity;
    for (0..missing) |_| {
        parts[count] = try Tree.initLeaf(value);
        count += 1;
    }
    switch (input.nodes.at(input.root)) {
        .leaf => {
            parts[count] = input.*;
            count += 1;
        },
        .tuple => |span| for (0..span.len) |i| {
            parts[count] = try input.topMode(i);
            count += 1;
        },
    }
    return Tree.initTuple(parts[0..count]);
}

pub fn append(input: *const Tree, value: Scalar, target_rank: usize) Error!Tree {
    if (target_rank < input.rank()) return Error.InvalidSelection;
    var parts: [layout.max_children]Tree = undefined;
    var count: usize = 0;
    if (target_rank > parts.len) return Error.OutOfCapacity;
    switch (input.nodes.at(input.root)) {
        .leaf => {
            parts[count] = input.*;
            count += 1;
        },
        .tuple => |span| for (0..span.len) |i| {
            parts[count] = try input.topMode(i);
            count += 1;
        },
    }
    while (count < target_rank) : (count += 1) parts[count] = try Tree.initLeaf(value);
    return Tree.initTuple(parts[0..count]);
}

pub fn flatten(input: *const Tree) Error!Tree {
    const leaves = try input.flattenLeaves();
    return treeFromFlat(leaves.slice());
}

/// Removes zero-stride modes, preserving the corresponding shape leaves.
pub fn filterZeros(input: *const Layout) Error!Layout {
    const shapes = try input.shape.flattenLeaves();
    const strides = try input.stride.flattenLeaves();
    var fs: Flat = .{};
    var fd: Flat = .{};
    for (strides.slice(), 0..) |stride, i| {
        if (stride != 0) {
            try fs.append(shapes.at(i));
            try fd.append(stride);
        }
    }
    if (fs.len == 0) {
        try fs.append(1);
        try fd.append(0);
    }
    return Layout.init(try treeFromFlat(fs.slice()), try treeFromFlat(fd.slice()));
}

/// Static approximation of CuTe's filter for layouts: filter zero strides then
/// coalesce contiguous modes. This is exact for static affine Layout values.
pub fn filter(input: *const Layout) Error!Layout {
    const no_zeros = try filterZeros(input);
    return no_zeros.coalesce();
}

pub fn size(input: *const Tree) Error!Unsigned {
    return input.product();
}

pub fn shapeDiv(lhs: *const Tree, rhs: *const Tree) Error!Tree {
    if (!lhs.sameProfile(rhs)) return Error.ProfileMismatch;
    return tuple.zipLeaves(lhs, rhs, divExactOrCeil);
}

fn divExactOrCeil(a: Scalar, b: Scalar) Scalar {
    if (b == 0) @panic("division by zero in shapeDiv");
    return @divTrunc(a, b);
}

pub fn ceilDiv(input: *const Tree, tiler: *const Tree) Error!Tree {
    const padded_tiler = if (input.rank() >= tiler.rank()) try append(tiler, 1, input.rank()) else return Error.RankMismatch;
    if (!input.sameProfile(&padded_tiler)) return Error.ProfileMismatch;
    return tuple.zipLeaves(input, &padded_tiler, ceilDivScalar);
}

fn ceilDivScalar(a: Scalar, b: Scalar) Scalar {
    if (b <= 0) @panic("ceilDiv requires positive divisor");
    return @divTrunc(a + b - 1, b);
}

pub fn roundUp(a: *const Tree, b: *const Tree) Error!Tree {
    const padded_b = if (a.rank() >= b.rank()) try append(b, 1, a.rank()) else return Error.RankMismatch;
    if (!a.sameProfile(&padded_b)) return Error.ProfileMismatch;
    return tuple.zipLeaves(a, &padded_b, roundUpScalar);
}

fn roundUpScalar(a: Scalar, b: Scalar) Scalar {
    if (b <= 0) @panic("roundUp requires positive divisor");
    return @divTrunc(a + b - 1, b) * b;
}

pub fn makeLayout(comptime shape_spec: anytype, comptime stride_spec: anytype) Layout {
    return layout.makeLayout(shape_spec, stride_spec);
}

pub fn makeIdentityLayout(comptime shape_spec: anytype) Layout {
    return layout.makeCompactLayout(shape_spec);
}

pub fn makeCompactLayout(comptime shape_spec: anytype) Layout {
    return layout.makeCompactLayout(shape_spec);
}

pub fn makeCompactRightLayout(comptime shape_spec: anytype) Layout {
    return comptime blk: {
        const shape = Tree.fromComptime(shape_spec);
        break :blk Layout.makeCompactRight(shape) catch |err| switch (err) {
            Error.InvalidShape => @compileError("compact-right layout shape has a non-positive extent"),
            Error.Overflow => @compileError("compact-right layout stride computation overflowed"),
            Error.OutOfCapacity => @compileError("compact-right layout exceeds not-cute bounded tree capacity"),
            else => @compileError("invalid compact-right layout shape"),
        };
    };
}

/// Static ordered-layout constructor. `order` lists fastest-to-slowest logical
/// leaf indices, matching the compact-left stride convention when order is
/// .{0,1,2,...}.
pub fn makeOrderedLayout(shape: Tree, order: []const usize) Error!Layout {
    const flat_shape = try shape.flattenLeaves();
    if (order.len != flat_shape.len) return Error.RankMismatch;
    var seen = [_]bool{false} ** layout.max_leaves;
    var strides: Flat = .{};
    for (0..flat_shape.len) |_| try strides.append(0);
    var stride_value: Scalar = 1;
    for (order) |idx| {
        if (idx >= flat_shape.len or seen[idx]) return Error.InvalidSelection;
        seen[idx] = true;
        strides.set(idx, stride_value);
        stride_value = std.math.mul(Scalar, stride_value, flat_shape.at(idx)) catch return Error.Overflow;
    }
    return Layout.init(shape, try Tree.fromProfileAndLeaves(&shape, strides.slice()));
}

pub fn coalesce(input: *const Layout) Error!Layout {
    return input.coalesce();
}

pub fn crd2idx(input: *const Layout, coord: Tree) Error!Scalar {
    return input.crd2idx(coord);
}

pub fn idx2crd(input: *const Layout, idx: Scalar) Error!Tree {
    const flat = try input.idx2crdFlat(idx);
    return Tree.fromProfileAndLeaves(&input.shape, flat.slice());
}

pub fn composition(a: Layout, b: Layout) Error!AnyLayout {
    return layout.composition(a, b);
}

/// A conservative static complement for compact affine layouts. It returns a
/// compact layout over the residue domain between size(input) and max(cosize,
/// target_cosize). This is intentionally only exposed for cases where the
/// exact target domain is known.
pub fn complement(input: *const Layout, target_cosize: Unsigned) Error!Layout {
    const used = try input.size();
    if (target_cosize < used) return Error.InvalidShape;
    const rest = target_cosize - used;
    const rest_scalar: Scalar = @intCast(if (rest == 0) 1 else rest);
    return Layout.makeCompact(try Tree.initLeaf(rest_scalar));
}

pub fn rightInverse(input: *const Layout) Error!Layout {
    if (!try input.isCompact()) return Error.NotCompact;
    return input.*;
}

pub fn leftInverse(input: *const Layout) Error!Layout {
    if (!try input.isCompact()) return Error.NotCompact;
    return input.*;
}

/// Static logical product over two layouts as a hierarchical concatenation.
pub fn logicalProduct(lhs: Layout, rhs: Layout) Error!Layout {
    return Layout.concatenate(&.{ lhs, rhs });
}

/// Static zipped product: shape/stride modes grouped as ((lhs),(rhs)).
pub fn zippedProduct(lhs: Layout, rhs: Layout) Error!Layout {
    return Layout.concatenate(&.{ lhs, rhs });
}

/// Static flat product: flatten both operands before concatenation.
pub fn flatProduct(lhs: *const Layout, rhs: *const Layout) Error!Layout {
    const lf = try lhs.flatten();
    const rf = try rhs.flatten();
    return Layout.concatenate(&.{ lf, rf });
}

fn treeFromFlat(leaves: []const Scalar) Error!Tree {
    if (leaves.len == 0) return Error.InvalidShape;
    if (leaves.len == 1) return Tree.initLeaf(leaves[0]);
    var parts: [layout.max_leaves]Tree = undefined;
    if (leaves.len > parts.len) return Error.OutOfCapacity;
    for (leaves, 0..) |leaf, i| parts[i] = try Tree.initLeaf(leaf);
    return Tree.initTuple(parts[0..leaves.len]);
}

test "core_static: get select append prepend flatten" {
    const t = Tree.fromComptime(.{ 2, .{ 3, 4 }, 5 });
    const got = try get(&t, &.{ 1, 0 });
    try std.testing.expectEqualSlices(Scalar, &.{3}, (try got.flattenLeaves()).slice());

    const sel = try select(&t, &.{ 2, 0 });
    try std.testing.expectEqualSlices(Scalar, &.{ 5, 2 }, (try sel.flattenLeaves()).slice());

    const flat = try flatten(&t);
    try std.testing.expectEqual(@as(usize, 4), flat.rank());
}

test "core_static: shape math" {
    const a = Tree.fromComptime(.{ 10, 6 });
    const b = Tree.fromComptime(.{ 3, 4 });
    const c = try ceilDiv(&a, &b);
    try std.testing.expectEqualSlices(Scalar, &.{ 4, 2 }, (try c.flattenLeaves()).slice());
    const r = try roundUp(&a, &b);
    try std.testing.expectEqualSlices(Scalar, &.{ 12, 8 }, (try r.flattenLeaves()).slice());
}

test "core_static: ordered layout and filter zeros" {
    const shape = Tree.fromComptime(.{ 2, 3, 4 });
    const ordered = try makeOrderedLayout(shape, &.{ 2, 1, 0 });
    try std.testing.expectEqual(@as(Scalar, 1 * 12 + 2 * 4 + 3), try ordered.crd2idxFlat(&.{ 1, 2, 3 }));

    const with_zero = layout.makeLayout(.{ 2, 3, 4 }, .{ 12, 0, 1 });
    const filtered = try filterZeros(&with_zero);
    try std.testing.expectEqualSlices(Scalar, &.{ 2, 4 }, (try filtered.shape.flattenLeaves()).slice());
}
