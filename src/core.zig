const std = @import("std");

pub const layout = @import("layout.zig");
pub const tuple = @import("tuple.zig");
pub const core_static = @import("core_static.zig");
pub const layout_algebra = @import("layout_algebra.zig");
pub const layout_core = @import("layout_core.zig");
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
pub const Error = layout.Error || tuple.Error || core_static.Error || layout_algebra.Error || layout_core.Error || basis.Error || error{
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

pub fn isStaticTree(tree: *const Tree) bool {
    return core_static.isStaticTree(tree);
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

pub fn static(value: Scalar) Scalar {
    return core_static.static(value);
}

pub fn getLeaves(input: *const Tree) Error!Flat {
    return input.flattenLeaves();
}

pub fn depth(input: *const Tree) usize {
    return core_static.depth(input);
}

pub fn rank(input: *const Tree) usize {
    return core_static.rank(input);
}

pub fn isCongruent(lhs: *const Tree, rhs: *const Tree) bool {
    return core_static.isCongruent(lhs, rhs);
}

pub fn isWeaklyCongruent(lhs: *const Tree, rhs: *const Tree) bool {
    return core_static.isWeaklyCongruent(lhs, rhs);
}

pub fn get(input: *const Tree, path: []const usize) Error!Tree {
    return core_static.get(input, path);
}

pub fn select(input: *const Tree, modes: []const usize) Error!Tree {
    return core_static.select(input, modes);
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

pub fn prepend(input: *const Tree, value: Scalar, target_rank: usize) Error!Tree {
    return core_static.prepend(input, value, target_rank);
}

pub fn append(input: *const Tree, value: Scalar, target_rank: usize) Error!Tree {
    return core_static.append(input, value, target_rank);
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

pub fn flatten(input: *const Tree) Error!Tree {
    return core_static.flatten(input);
}

pub fn filterZeros(input: *const Layout) Error!Layout {
    return core_static.filterZeros(input);
}

pub fn filter(input: *const Layout) Error!Layout {
    return core_static.filter(input);
}

pub fn size(input: *const Tree) Error!Unsigned {
    return core_static.size(input);
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

pub fn shapeDiv(lhs: *const Tree, rhs: *const Tree) Error!Tree {
    return core_static.shapeDiv(lhs, rhs);
}

pub fn ceilDiv(lhs: *const Tree, rhs: *const Tree) Error!Tree {
    return core_static.ceilDiv(lhs, rhs);
}

pub fn roundUp(lhs: *const Tree, rhs: *const Tree) Error!Tree {
    return core_static.roundUp(lhs, rhs);
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

pub fn makeLayout(comptime shape_spec: anytype, comptime stride_spec: anytype) Layout {
    return core_static.makeLayout(shape_spec, stride_spec);
}

pub fn makeIdentityLayout(comptime shape_spec: anytype) Layout {
    return core_static.makeIdentityLayout(shape_spec);
}

pub fn makeOrderedLayout(shape_value: Tree, order: []const usize) Error!Layout {
    return core_static.makeOrderedLayout(shape_value, order);
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

pub fn coalesce(input: *const Layout) Error!Layout {
    return core_static.coalesce(input);
}

pub fn crd2idx(input: *const Layout, coord: Tree) Error!Scalar {
    return core_static.crd2idx(input, coord);
}

pub fn idx2crd(input: *const Layout, idx: Scalar) Error!Tree {
    return core_static.idx2crd(input, idx);
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

pub fn composition(lhs: Layout, rhs: Layout) Error!AnyLayout {
    return layout_core.composition(lhs, rhs);
}

pub fn complement(input: *const Layout, cotarget: Unsigned) Error!Layout {
    return layout_core.complement(input, cotarget);
}

pub fn rightInverse(input: *const Layout) Error!Layout {
    return layout_core.rightInverse(input);
}

pub fn leftInverse(input: *const Layout) Error!Layout {
    return layout_core.leftInverse(input);
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
