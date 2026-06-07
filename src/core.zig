const std = @import("std");

pub const layout = @import("layout.zig");
pub const tuple = @import("tuple.zig");
pub const layout_algebra = layout;
pub const layout_core = layout;
pub const basis = @import("basis.zig");
pub const typing = @import("typing.zig");

pub const Scalar = layout.Scalar;
pub const Unsigned = layout.Unsigned;
pub const Tree = layout.Tree;
pub const Flat = layout.Flat;
pub const Layout = layout.Layout;
pub const AnyLayout = layout.AnyLayout;
pub const ComposedLayout = layout.ComposedLayout;
pub const Ratio = basis.Ratio;
pub const ScaledBasis = basis.ScaledBasis;
pub const Swizzle = basis.Swizzle;
pub const Selector = layout_algebra.Selector;
pub const SelectorItem = layout_algebra.SelectorItem;
pub const Error = basis.Error || error{
    EmptyInput,
    NonStaticInput,
    NotImplementedForDynamicValue,
};

/// Static integer value wrapper used where Python CuteDSL uses IntValue.
pub const IntValue = struct {
    value: Scalar,

    pub fn init(value: Scalar) IntValue {
        return .{ .value = value };
    }

    pub fn asScalar(self: IntValue) Scalar {
        return self.value;
    }
};

pub fn E(mode: usize) Error!ScaledBasis {
    return ScaledBasis.E(mode);
}

pub fn getDivisibility(value: Scalar) Unsigned {
    const abs_value: Unsigned = if (value < 0) @intCast(-value) else @intCast(value);
    return if (abs_value == 0) 1 else abs_value;
}

pub fn basisValue(value: ScaledBasis) basis.Scale {
    return value.basisValue();
}

pub fn basisGet(value: ScaledBasis, tree: *const Tree) Error!Tree {
    return basis.basisGet(value, tree);
}

pub fn isTuple(tree: *const Tree) bool {
    return switch (tree.nodes.at(tree.root)) {
        .tuple => true,
        .leaf => false,
    };
}

pub fn isStatic(_: anytype) bool {
    return true;
}

pub fn isValidLeaf(value: Scalar) bool {
    return value > 0;
}

pub fn hasUnderscore(_: *const Tree) bool {
    return false;
}

pub fn hasScaledBasis(_: *const Tree) bool {
    return false;
}

pub fn getLeaves(input: *const Tree) Error!Flat {
    return input.flattenLeaves();
}

pub fn groupModes(input: *const Tree, begin: isize, end: ?isize) Error!Tree {
    return layout_algebra.groupModes(input, begin, end);
}

pub fn groupModesLayout(
    input: *const Layout,
    begin: isize,
    end: ?isize,
) Error!Layout {
    return layout_algebra.groupModesLayout(input, begin, end);
}

pub fn slice_(input: *const Tree, selector: *const Selector) Error!Tree {
    return layout_algebra.sliceTree(input, selector);
}

pub fn sliceLayout(input: *const Layout, selector: *const Selector) Error!Layout {
    return layout_algebra.sliceLayout(input, selector);
}

pub fn dice(input: *const Tree, selector: *const Selector) Error!Tree {
    return layout_algebra.diceTree(input, selector);
}

pub fn diceLayout(input: *const Layout, selector: *const Selector) Error!Layout {
    return layout_algebra.diceLayout(input, selector);
}

pub fn prependOnes(input: *const Layout, up_to_rank: ?usize) Error!Layout {
    return layout_algebra.prependOnesLayout(input, up_to_rank);
}

pub fn appendOnes(input: *const Layout, up_to_rank: ?usize) Error!Layout {
    return layout_algebra.appendOnesLayout(input, up_to_rank);
}

pub fn repeatAsTuple(value: Tree, count: usize) Error!Tree {
    if (count > layout.max_children) return Error.OutOfCapacity;
    var parts: [layout.max_children]Tree = undefined;
    for (0..count) |i| parts[i] = value;
    return Tree.initTuple(parts[0..count]);
}

pub fn repeat(value: Tree, count: usize) Error!Tree {
    return repeatAsTuple(value, count);
}

pub fn repeatLike(value: Tree, profile: *const Tree) Error!Tree {
    const leaves = try profile.flattenLeaves();
    var repeated: Flat = .{};
    const value_flat = try value.flattenLeaves();
    if (value_flat.len == 0) return Error.EmptyInput;
    for (0..leaves.len) |i| try repeated.append(value_flat.at(i % value_flat.len));
    return Tree.fromProfileAndLeaves(profile, repeated.slice());
}

pub fn product(input: *const Tree) Error!Unsigned {
    return tuple.product(input);
}

pub fn innerProduct(lhs: *const Tree, rhs: *const Tree) Error!Scalar {
    if (!lhs.sameProfile(rhs)) return Error.ProfileMismatch;
    const lf = try lhs.flattenLeaves();
    const rf = try rhs.flattenLeaves();
    var total: Scalar = 0;
    for (lf.slice(), 0..) |v, i| {
        const prod = std.math.mul(Scalar, v, rf.at(i)) catch return Error.Overflow;
        total = std.math.add(Scalar, total, prod) catch return Error.Overflow;
    }
    return total;
}

pub fn prefixProduct(input: *const Tree) Error!Tree {
    const flat = try input.flattenLeaves();
    var out: Flat = .{};
    var running: Scalar = 1;
    for (flat.slice()) |v| {
        try out.append(running);
        if (v <= 0) return Error.InvalidShape;
        running = std.math.mul(Scalar, running, v) catch return Error.Overflow;
    }
    return Tree.fromProfileAndLeaves(input, out.slice());
}

fn maxScalar(a: Scalar, b: Scalar) Scalar {
    return @max(a, b);
}
fn minScalar(a: Scalar, b: Scalar) Scalar {
    return @min(a, b);
}

pub fn elemLess(lhs: *const Tree, rhs: *const Tree) Error!bool {
    return tuple.elemLess(lhs, rhs);
}

pub fn elemMax(lhs: *const Tree, rhs: *const Tree) Error!Tree {
    return tuple.zipLeaves(lhs, rhs, maxScalar);
}

pub fn elemMin(lhs: *const Tree, rhs: *const Tree) Error!Tree {
    return tuple.zipLeaves(lhs, rhs, minScalar);
}

pub fn makeLayoutLike(input: *const Layout) Error!Layout {
    return layout_algebra.makeLayoutLike(input);
}

pub fn compactColMajor(shape_value: Tree) Error!Layout {
    return Layout.makeCompact(shape_value);
}

pub fn compactRowMajor(shape_value: Tree) Error!Layout {
    return Layout.makeCompactRight(shape_value);
}

pub fn makeComposedLayout(a: Layout, offset: Scalar, b: Layout) ComposedLayout {
    return .{ .a = a, .offset = offset, .b = b };
}

pub fn cosize(input: *const Layout) Error!Unsigned {
    return input.cosize();
}

pub fn sizeInBytes(type_bits: Unsigned, maybe_layout: ?*const Layout) Error!Unsigned {
    return layout_algebra.sizeInBytes(type_bits, maybe_layout);
}

pub fn incrementCoord(coord: *const Tree, shape_value: *const Tree) Error!Tree {
    var c = coord.*;
    return layout_algebra.incrementCoord(&c, shape_value);
}

pub fn recastLayout(input: *const Layout, new_shape: Tree) Error!Layout {
    if ((try input.size()) != (try new_shape.product())) return Error.InvalidShape;
    return Layout.makeCompact(new_shape);
}

pub fn sliceAndOffset(
    input: *const Layout,
    selector: *const Selector,
) Error!layout_algebra.SliceResult {
    return layout_algebra.sliceAndOffset(input, selector);
}

pub fn shape(input: *const Layout, mode: ?usize) Error!Tree {
    return layout_algebra.shapeOf(input, mode);
}

pub fn stride(input: *const Layout, mode: ?usize) Error!Tree {
    return layout_algebra.strideOf(input, mode);
}

pub fn logicalProduct(block: *const Layout, tiler: *const Tree) Error!Layout {
    return layout_core.logicalProduct(block, tiler);
}

pub fn zippedProduct(block: *const Layout, tiler: *const Tree) Error!Layout {
    return layout_core.zippedProduct(block, tiler);
}

pub fn tiledProduct(block: *const Layout, tiler: *const Tree) Error!Layout {
    return layout_core.tiledProduct(block, tiler);
}

pub fn flatProduct(block: *const Layout, tiler: *const Tree) Error!Layout {
    return layout_core.flatProduct(block, tiler);
}

pub fn rakedProduct(block: *const Layout, tiler: *const Tree) Error!Layout {
    return layout_core.rakedProduct(block, tiler);
}

pub fn blockedProduct(block: *const Layout, tiler: *const Tree) Error!Layout {
    return layout_core.blockedProduct(block, tiler);
}

pub fn logicalDivide(target: *const Layout, tiler: *const Tree) Error!Layout {
    return layout_core.logicalDivide(target, tiler);
}

pub fn zippedDivide(target: *const Layout, tiler: *const Tree) Error!Layout {
    return layout_core.zippedDivide(target, tiler);
}

pub fn tiledDivide(target: *const Layout, tiler: *const Tree) Error!Layout {
    return layout_core.tiledDivide(target, tiler);
}

pub fn flatDivide(target: *const Layout, tiler: *const Tree) Error!Layout {
    return layout_core.flatDivide(target, tiler);
}

pub fn maxCommonLayout(a: *const Layout, b: *const Layout) Error!Layout {
    return layout_core.maxCommonLayout(a, b);
}

pub fn maxCommonVector(a: *const Layout, b: *const Layout) Error!Scalar {
    return layout_core.maxCommonVector(a, b);
}

pub fn tileToShape(
    atom: *const Layout,
    target_shape: *const Tree,
    order: []const usize,
) Error!Layout {
    return layout_core.tileToShape(atom, target_shape, order);
}

pub fn localPartition(
    target: *const Layout,
    tiler: *const Tree,
    worker_coord: *const Tree,
) Error!Layout {
    const divided = try layout_core.flatDivide(target, tiler);
    const selector = try selectorForWorker(worker_coord, divided.leafCount());
    return layout_algebra.sliceLayout(&divided, &selector);
}

pub fn localTile(
    target: *const Layout,
    tiler: *const Tree,
    tile_coord: *const Tree,
) Error!Layout {
    const divided = try layout_core.flatDivide(target, tiler);
    const selector = try selectorForWorker(tile_coord, divided.leafCount());
    return layout_algebra.sliceLayout(&divided, &selector);
}

fn selectorForWorker(coord: *const Tree, rank_value: usize) Error!Selector {
    const coord_flat = try coord.flattenLeaves();
    var parts: [layout.max_leaves]Selector = undefined;
    if (rank_value > parts.len) return Error.OutOfCapacity;
    var c: usize = 0;
    while (c < coord_flat.len and c < rank_value) : (c += 1) parts[c] = try Selector.initFixed(coord_flat.at(c));
    while (c < rank_value) : (c += 1) parts[c] = try Selector.initKeep();
    return Selector.initTuple(parts[0..rank_value]);
}

pub fn makeLayoutImageMask(
    lay: *const Layout,
    coord: []const Scalar,
    mode: usize,
) Error!u16 {
    return layout_core.makeLayoutImageMask(lay, coord, mode);
}

pub fn leadingDim(shape_value: *const Tree, stride_value: *const Tree) Error!?usize {
    return layout_core.leadingDim(shape_value, stride_value);
}

pub fn makeLayoutTv(
    thr_layout: *const Layout,
    val_layout: *const Layout,
) Error!layout_core.LayoutTv {
    return layout_core.makeLayoutTv(thr_layout, val_layout);
}

pub fn getNonswizzlePortion(input: *const Layout) Error!Layout {
    return input.*;
}

pub fn getSwizzlePortion(_: *const Layout) ?Swizzle {
    return null;
}

pub fn nullspace(input: *const Layout) Error!Layout {
    return layout_core.nullspace(input);
}

pub const FastDivmodDivisor = struct {
    divisor: u32,

    pub fn init(divisor: u32) Error!FastDivmodDivisor {
        if (divisor == 0) return Error.DivisionByZero;
        return .{ .divisor = divisor };
    }

    pub fn divmod(self: FastDivmodDivisor, value: u32) struct { q: u32, r: u32 } {
        return .{ .q = value / self.divisor, .r = value % self.divisor };
    }
};

pub fn fastDivmodCreateDivisor(divisor: u32) Error!FastDivmodDivisor {
    return FastDivmodDivisor.init(divisor);
}

test "core API: static tree predicates and arithmetic helpers" {
    const t = Tree.fromComptime(.{ 2, .{ 3, 4 } });
    try std.testing.expect(isTuple(&t));
    try std.testing.expectEqual(@as(usize, 2), rank(&t));
    try std.testing.expectEqual(@as(usize, 2), depth(&t));
    try std.testing.expectEqual(@as(Unsigned, 24), try product(&t));

    const flat = try flatten(&t);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 2, 3, 4 },
        (try flat.flattenLeaves()).slice(),
    );

    const p = try prefixProduct(&flat);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 1, 2, 6 },
        (try p.flattenLeaves()).slice(),
    );
}

test "core API: layout construction and divide/product wrappers" {
    const l = layout.makeLayout(.{ 8, 4 }, .{ 4, 1 });
    const tiler = Tree.fromComptime(.{ 2, 2 });
    const z = try zippedDivide(&l, &tiler);
    const back = try zippedProduct(&z, &tiler);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 8, 4 },
        (try back.shape.flattenLeaves()).slice(),
    );

    const cm = try compactColMajor(Tree.fromComptime(.{ 2, 3 }));
    try std.testing.expectEqual(@as(Scalar, 1 + 2 * 2), try cm.crd2idxFlat(&.{ 1, 2 }));
    const rm = try compactRowMajor(Tree.fromComptime(.{ 2, 3 }));
    try std.testing.expectEqual(@as(Scalar, 1 * 3 + 2), try rm.crd2idxFlat(&.{ 1, 2 }));
}

test "core API: local partition/tile and masks" {
    const l = layout.makeLayout(.{ 4, 4 }, .{ 4, 1 });
    const tiler = Tree.fromComptime(.{ 2, 2 });
    const worker = Tree.fromComptime(.{ 1, 0 });
    const p = try localPartition(&l, &tiler, &worker);
    try std.testing.expect(p.leafCount() >= 1);
    const mask = try makeLayoutImageMask(
        &layout.makeLayout(.{ 2, 2 }, .{ 2, 1 }),
        &.{ 0, 0 },
        1,
    );
    try std.testing.expect(mask != 0);
}

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
    for (strides.slice(), 0..) |strd, i| {
        if (strd != 0) {
            try fs.append(shapes.at(i));
            try fd.append(strd);
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
    const padded_tiler = if (input.rank() >= tiler.rank()) try append(
        tiler,
        1,
        input.rank(),
    ) else return Error.RankMismatch;
    if (!input.sameProfile(&padded_tiler)) return Error.ProfileMismatch;
    return tuple.zipLeaves(input, &padded_tiler, ceilDivScalar);
}

fn ceilDivScalar(a: Scalar, b: Scalar) Scalar {
    if (b <= 0) @panic("ceilDiv requires positive divisor");
    return @divTrunc(a + b - 1, b);
}

pub fn roundUp(a: *const Tree, b: *const Tree) Error!Tree {
    const padded_b = if (a.rank() >= b.rank()) try append(
        b,
        1,
        a.rank(),
    ) else return Error.RankMismatch;
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
        const shp = Tree.fromComptime(shape_spec);
        break :blk Layout.makeCompactRight(shp) catch |err| switch (err) {
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
pub fn makeOrderedLayout(shape_val: Tree, order: []const usize) Error!Layout {
    const flat_shape = try shape_val.flattenLeaves();
    if (order.len != flat_shape.len) return Error.RankMismatch;
    var seen = [_]bool{false} ** layout.max_leaves;
    var strides: Flat = .{};
    for (0..flat_shape.len) |_| try strides.append(0);
    var stride_value: Scalar = 1;
    for (order) |idx| {
        if (idx >= flat_shape.len or seen[idx]) return Error.InvalidSelection;
        seen[idx] = true;
        strides.set(idx, stride_value);
        stride_value = std.math.mul(
            Scalar,
            stride_value,
            flat_shape.at(idx),
        ) catch return Error.Overflow;
    }
    return Layout.init(shape_val, try Tree.fromProfileAndLeaves(&shape_val, strides.slice()));
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
/// Static zipped product: shape/stride modes grouped as ((lhs),(rhs)).
/// Static flat product: flatten both operands before concatenation.
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
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 5, 2 },
        (try sel.flattenLeaves()).slice(),
    );

    const flat = try flatten(&t);
    try std.testing.expectEqual(@as(usize, 4), flat.rank());
}

test "core_static: shape math" {
    const a = Tree.fromComptime(.{ 10, 6 });
    const b = Tree.fromComptime(.{ 3, 4 });
    const c = try ceilDiv(&a, &b);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 4, 2 },
        (try c.flattenLeaves()).slice(),
    );
    const r = try roundUp(&a, &b);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 12, 8 },
        (try r.flattenLeaves()).slice(),
    );
}

test "core_static: ordered layout and filter zeros" {
    const shp = Tree.fromComptime(.{ 2, 3, 4 });
    const ordered = try makeOrderedLayout(shp, &.{ 2, 1, 0 });
    try std.testing.expectEqual(
        @as(Scalar, 1 * 12 + 2 * 4 + 3),
        try ordered.crd2idxFlat(&.{ 1, 2, 3 }),
    );

    const with_zero = layout.makeLayout(.{ 2, 3, 4 }, .{ 12, 0, 1 });
    const filtered = try filterZeros(&with_zero);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 2, 4 },
        (try filtered.shape.flattenLeaves()).slice(),
    );
}
