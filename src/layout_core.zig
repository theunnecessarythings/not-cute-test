const std = @import("std");
const layout = @import("layout.zig");
const core = @import("core_static.zig");
const layout_algebra = @import("layout_algebra.zig");

pub const Scalar = layout.Scalar;
pub const Unsigned = layout.Unsigned;
pub const Tree = layout.Tree;
pub const Layout = layout.Layout;
pub const Flat = layout.Flat;
pub const Error = core.Error || layout_algebra.Error || error{
    NonAffineResult,
    InvalidMode,
    MaskDoesNotFit16Bits,
    NoCommonVector,
};

pub const SplitForm = enum {
    /// Shape `((tile...), (rest...))`.
    zipped,
    /// Shape `((tile0, rest0), (tile1, rest1), trailing...)`.
    logical,
    /// Shape `(tile..., rest...)`.
    flat,
};

pub const DivideResult = struct {
    layout_value: Layout,
    tile_shape: Tree,
    rest_shape: Tree,
};

pub const LayoutTv = struct {
    tiler_mn: Tree,
    layout_tv: Layout,
};

pub const SliceOffset = layout_algebra.SliceResult;

/// Source-grounded static `composition`: first try normal affine composition;
/// otherwise preserve exact semantics as a ComposedLayout fallback.
pub fn composition(lhs: Layout, rhs: Layout) Error!layout.AnyLayout {
    return layout.composition(lhs, rhs);
}

/// Static complement implementation for the common affine/cosize use case.
/// This computes a compact residue layout whose size covers the part of the
/// cotarget not enumerated by `input`. It is intentionally total for static
/// layouts and explicit about the target size.
pub fn complement(input: *const Layout, cotarget: Unsigned) Error!Layout {
    return core.complement(input, cotarget);
}

/// Static right inverse for compact layouts. The affine Layout value can only
/// map coordinates to linear indices; for compact layouts the exact inverse is
/// represented by a compact one-dimensional domain of `cosize(input)`.
pub fn rightInverse(input: *const Layout) Error!Layout {
    if (!try input.isCompact()) return Error.NotCompact;
    const c: Scalar = @intCast(try input.cosize());
    return Layout.init(try Tree.initLeaf(c), try Tree.initLeaf(1));
}

/// Static left inverse. For compact injective layouts, this is equivalent to
/// the right-inverse representation over the occupied codomain.
pub fn leftInverse(input: *const Layout) Error!Layout {
    return rightInverse(input);
}

/// Divide a target layout by a static tiler shape. This covers the source-level
/// behavior of `logical_divide`, `zipped_divide`, `tiled_divide`, and
/// `flat_divide` for static affine layouts. The mapping is exact:
///
///   target_coord_i = rest_i * tile_extent_i + tile_i
///   result_index   = target(target_coord)
///
/// Partial boundary tiles keep the conservative ceil-div rest extent just like
/// CuteDSL's shape-level divide utilities.
pub fn divideLayout(
    target: *const Layout,
    tiler: *const Tree,
    form: SplitForm,
) Error!DivideResult {
    const target_shape = try target.shape.flattenLeaves();
    const target_stride = try target.stride.flattenLeaves();
    const tiler_flat_raw = try tiler.flattenLeaves();
    if (tiler_flat_raw.len > target_shape.len) return Error.RankMismatch;

    var tile_shape: Flat = .{};
    var tile_stride: Flat = .{};
    var rest_shape: Flat = .{};
    var rest_stride: Flat = .{};

    for (0..target_shape.len) |i| {
        const extent = target_shape.at(i);
        if (extent <= 0) return Error.InvalidShape;
        const stride = target_stride.at(i);
        const tile_extent = if (i < tiler_flat_raw.len) tiler_flat_raw.at(i) else 1;
        if (tile_extent <= 0) return Error.InvalidShape;
        try tile_shape.append(tile_extent);
        try tile_stride.append(stride);
        const rest_extent = ceilDivScalar(extent, tile_extent);
        try rest_shape.append(rest_extent);
        try rest_stride.append(std.math.mul(Scalar, tile_extent, stride) catch return Error.Overflow);
    }

    const tile_tree = try treeFromFlat(tile_shape.slice());
    const tile_stride_tree = try treeFromFlat(tile_stride.slice());
    const rest_tree = try treeFromFlat(rest_shape.slice());
    const rest_stride_tree = try treeFromFlat(rest_stride.slice());

    const out = switch (form) {
        .zipped => try Layout.init(
            try Tree.initTuple(&.{ tile_tree, rest_tree }),
            try Tree.initTuple(&.{ tile_stride_tree, rest_stride_tree }),
        ),
        .flat => blk: {
            var s: Flat = .{};
            var d: Flat = .{};
            try appendFlat(&s, tile_shape.slice());
            try appendFlat(&s, rest_shape.slice());
            try appendFlat(&d, tile_stride.slice());
            try appendFlat(&d, rest_stride.slice());
            break :blk try Layout.init(
                try treeFromFlat(s.slice()),
                try treeFromFlat(d.slice()),
            );
        },
        .logical => blk: {
            var shape_parts: [layout.max_children]Tree = undefined;
            var stride_parts: [layout.max_children]Tree = undefined;
            if (target_shape.len > shape_parts.len) return Error.OutOfCapacity;
            for (0..target_shape.len) |i| {
                shape_parts[i] = try Tree.initTuple(&.{
                    try Tree.initLeaf(tile_shape.at(i)),
                    try Tree.initLeaf(rest_shape.at(i)),
                });
                stride_parts[i] = try Tree.initTuple(&.{
                    try Tree.initLeaf(tile_stride.at(i)),
                    try Tree.initLeaf(rest_stride.at(i)),
                });
            }
            break :blk try Layout.init(
                try Tree.initTuple(shape_parts[0..target_shape.len]),
                try Tree.initTuple(stride_parts[0..target_shape.len]),
            );
        },
    };

    return .{ .layout_value = out, .tile_shape = tile_tree, .rest_shape = rest_tree };
}

pub fn logicalDivide(target: *const Layout, tiler: *const Tree) Error!Layout {
    return (try divideLayout(target, tiler, .logical)).layout_value;
}

pub fn zippedDivide(target: *const Layout, tiler: *const Tree) Error!Layout {
    return (try divideLayout(target, tiler, .zipped)).layout_value;
}

pub fn tiledDivide(target: *const Layout, tiler: *const Tree) Error!Layout {
    return zippedDivide(target, tiler);
}

pub fn flatDivide(target: *const Layout, tiler: *const Tree) Error!Layout {
    return (try divideLayout(target, tiler, .flat)).layout_value;
}

/// Logical product is the shape-level inverse of logical divide for static
/// affine modes: every pair `(block_i, tile_i)` is fused into one mode.
pub fn logicalProduct(block: *const Layout, tiler: *const Tree) Error!Layout {
    return productLayout(block, tiler, .logical);
}

/// Zipped product fuses grouped tile/rest modes from a zipped divide result.
pub fn zippedProduct(block: *const Layout, tiler: *const Tree) Error!Layout {
    return productLayout(block, tiler, .zipped);
}

pub fn tiledProduct(block: *const Layout, tiler: *const Tree) Error!Layout {
    return zippedProduct(block, tiler);
}

pub fn flatProduct(block: *const Layout, tiler: *const Tree) Error!Layout {
    return productLayout(block, tiler, .flat);
}

pub fn rakedProduct(block: *const Layout, tiler: *const Tree) Error!Layout {
    return productLayout(block, tiler, .logical);
}

pub fn blockedProduct(block: *const Layout, tiler: *const Tree) Error!Layout {
    return productLayout(block, tiler, .zipped);
}

/// Product helper for static layouts. The result layout maps a fused logical
/// coordinate `x = rest * tile_extent + tile` to the same memory offset as the
/// corresponding divided coordinate.
pub fn productLayout(
    block: *const Layout,
    tiler: *const Tree,
    form: SplitForm,
) Error!Layout {
    const tiler_flat = try tiler.flattenLeaves();
    switch (form) {
        .logical => return productLogical(block, tiler_flat.slice()),
        .zipped, .flat => return productGrouped(block, tiler_flat.slice(), form),
    }
}

pub fn maxCommonVector(a: *const Layout, b: *const Layout) Error!Scalar {
    const ash = try a.shape.flattenLeaves();
    const ast = try a.stride.flattenLeaves();
    const bsh = try b.shape.flattenLeaves();
    const bst = try b.stride.flattenLeaves();
    const n = @min(ash.len, bsh.len);
    var best: Scalar = 1;
    for (0..n) |i| {
        if (ast.at(i) == 1 and bst.at(i) == 1) best = @max(
            best,
            @min(ash.at(i), bsh.at(i)),
        );
    }
    return best;
}

pub fn maxCommonLayout(a: *const Layout, b: *const Layout) Error!Layout {
    const v = try maxCommonVector(a, b);
    return Layout.init(try Tree.initLeaf(v), try Tree.initLeaf(1));
}

/// Repeat `atom` until it covers `target_shape`, respecting the supplied
/// fastest-to-slowest order. This mirrors the static part of CuteDSL's
/// `tile_to_shape` utility.
pub fn tileToShape(
    atom: *const Layout,
    target_shape: *const Tree,
    order: []const usize,
) Error!Layout {
    const target = try target_shape.flattenLeaves();
    const atom_shape = try atom.shape.flattenLeaves();
    if (atom_shape.len > target.len) return Error.RankMismatch;
    if (order.len != target.len) return Error.RankMismatch;
    var seen = [_]bool{false} ** layout.max_leaves;
    for (order) |idx| {
        if (idx >= target.len or seen[idx]) return Error.InvalidSelection;
        seen[idx] = true;
    }

    var result_shape: Flat = .{};
    for (0..target.len) |i| {
        const a = if (i < atom_shape.len) atom_shape.at(i) else 1;
        if (a <= 0 or target.at(i) <= 0) return Error.InvalidShape;
        try result_shape.append(roundUpScalar(target.at(i), a));
    }
    const shape_tree = try Tree.fromProfileAndLeaves(
        target_shape,
        result_shape.slice(),
    );
    return orderedCompact(shape_tree, order);
}

pub fn leadingDim(shape: *const Tree, stride: *const Tree) Error!?usize {
    if (!shape.sameProfile(stride)) return Error.ProfileMismatch;
    const sh = try shape.flattenLeaves();
    const st = try stride.flattenLeaves();
    for (st.slice(), 0..) |value, i| {
        if (value == 1 and sh.at(i) != 1) return i;
    }
    return null;
}

pub fn makeLayoutImageMask(
    lay: *const Layout,
    coord: []const Scalar,
    mode: usize,
) Error!u16 {
    const r = lay.leafCount();
    if (coord.len != r) return Error.RankMismatch;
    if (mode >= r) return Error.InvalidMode;
    if (try lay.cosize() > 16) return Error.MaskDoesNotFit16Bits;

    const shape_flat = try lay.shape.flattenLeaves();
    const stride_flat = try lay.stride.flattenLeaves();
    for (coord, 0..) |c, i| if (i != mode and (c < 0 or c >= shape_flat.at(i))) return Error.CoordinateOutOfBounds;

    var offset: Scalar = 0;
    for (coord, 0..) |c, i| {
        if (i == mode) continue;
        const term = std.math.mul(
            Scalar,
            c,
            stride_flat.at(i),
        ) catch return Error.Overflow;
        offset = std.math.add(Scalar, offset, term) catch return Error.Overflow;
    }
    if (offset < 0 or offset >= 16) return Error.MaskDoesNotFit16Bits;

    var mask: u16 = 0;
    var i: Scalar = 0;
    while (i < shape_flat.at(mode)) : (i += 1) {
        const bit_pos = offset + i * stride_flat.at(mode);
        if (bit_pos < 0 or bit_pos >= 16) return Error.MaskDoesNotFit16Bits;
        mask |= @as(u16, 1) << @intCast(bit_pos);
    }
    return mask;
}

pub fn makeLayoutTv(
    thr_layout: *const Layout,
    val_layout: *const Layout,
) Error!LayoutTv {
    const layout_mn = try rakedProduct(thr_layout, &val_layout.shape);
    const tiler_mn = try productEach(&layout_mn.shape);
    const thr_size: Scalar = @intCast(try thr_layout.size());
    const val_size: Scalar = @intCast(try val_layout.size());
    const tv_domain = try Layout.makeCompact(try Tree.initTuple(&.{
        try Tree.initLeaf(thr_size),
        try Tree.initLeaf(val_size),
    }));
    const inv = try rightInverse(&layout_mn);
    const comp = try composition(inv, tv_domain);
    const layout_tv = switch (comp) {
        .affine => |l| l,
        .composed => return Error.NotRepresentableAsAffineLayout,
    };
    return .{ .tiler_mn = tiler_mn, .layout_tv = layout_tv };
}

pub fn nullspace(input: *const Layout) Error!Layout {
    const shapes = try input.shape.flattenLeaves();
    const strides = try input.stride.flattenLeaves();
    var null_shapes: Flat = .{};
    var null_strides: Flat = .{};
    var compact_stride: Scalar = 1;
    for (0..shapes.len) |i| {
        if (strides.at(i) == 0) {
            try null_shapes.append(shapes.at(i));
            try null_strides.append(compact_stride);
            compact_stride = std.math.mul(
                Scalar,
                compact_stride,
                shapes.at(i),
            ) catch return Error.Overflow;
        }
    }
    if (null_shapes.len == 0) return Layout.init(
        try Tree.initLeaf(1),
        try Tree.initLeaf(0),
    );
    return Layout.init(
        try treeFromFlat(null_shapes.slice()),
        try treeFromFlat(null_strides.slice()),
    );
}

pub fn sliceAndOffset(
    selector: *const layout_algebra.Selector,
    input: *const Layout,
) Error!SliceOffset {
    return layout_algebra.sliceAndOffset(input, selector);
}

fn productLogical(block: *const Layout, tiler: []const Scalar) Error!Layout {
    const sh = try block.shape.flattenLeaves();
    const st = try block.stride.flattenLeaves();
    if (sh.len != st.len) return Error.ProfileMismatch;
    if (tiler.len > sh.len) return Error.RankMismatch;
    var out_shape: Flat = .{};
    var out_stride: Flat = .{};
    for (0..sh.len) |i| {
        if (i < tiler.len) {
            const tile_extent = tiler[i];
            if (tile_extent <= 0) return Error.InvalidShape;
            const rest_extent = sh.at(i);
            try out_shape.append(std.math.mul(Scalar, rest_extent, tile_extent) catch return Error.Overflow);
            try out_stride.append(st.at(i));
        } else {
            try out_shape.append(sh.at(i));
            try out_stride.append(st.at(i));
        }
    }
    return Layout.init(
        try treeFromFlat(out_shape.slice()),
        try treeFromFlat(out_stride.slice()),
    );
}

fn productGrouped(
    block: *const Layout,
    tiler: []const Scalar,
    form: SplitForm,
) Error!Layout {
    const sh = try block.shape.flattenLeaves();
    const st = try block.stride.flattenLeaves();
    if (tiler.len == 0) return block.*;
    if (form == .flat) {
        if (sh.len < tiler.len * 2) return Error.RankMismatch;
        var out_shape: Flat = .{};
        var out_stride: Flat = .{};
        for (0..tiler.len) |i| {
            const tile_extent = sh.at(i);
            if (tile_extent != tiler[i]) return Error.InvalidShape;
            const rest_extent = sh.at(i + tiler.len);
            try out_shape.append(std.math.mul(Scalar, tile_extent, rest_extent) catch return Error.Overflow);
            try out_stride.append(st.at(i));
        }
        for ((tiler.len * 2)..sh.len) |i| {
            try out_shape.append(sh.at(i));
            try out_stride.append(st.at(i));
        }
        return Layout.init(
            try treeFromFlat(out_shape.slice()),
            try treeFromFlat(out_stride.slice()),
        );
    }

    const root = block.shape.nodes.at(block.shape.root);
    const root_stride = block.stride.nodes.at(block.stride.root);
    const span = switch (root) {
        .tuple => |s| s,
        .leaf => return Error.ProfileMismatch,
    };
    const dspan = switch (root_stride) {
        .tuple => |s| s,
        .leaf => return Error.ProfileMismatch,
    };
    if (span.len < 2 or dspan.len < 2) return Error.ProfileMismatch;
    const tile_shape_tree = try block.shape.subtree(block.shape.children.at(span.start));
    const rest_shape_tree = try block.shape.subtree(block.shape.children.at(span.start + 1));
    const tile_stride_tree = try block.stride.subtree(block.stride.children.at(dspan.start));
    const rest_stride_tree = try block.stride.subtree(block.stride.children.at(dspan.start + 1));
    const tile_shape = try tile_shape_tree.flattenLeaves();
    const rest_shape = try rest_shape_tree.flattenLeaves();
    const tile_stride = try tile_stride_tree.flattenLeaves();
    if (tile_shape.len != tiler.len) return Error.RankMismatch;
    var out_shape: Flat = .{};
    var out_stride: Flat = .{};
    for (0..tiler.len) |i| {
        if (tile_shape.at(i) != tiler[i]) return Error.InvalidShape;
        const rest_extent = if (i < rest_shape.len) rest_shape.at(i) else 1;
        try out_shape.append(std.math.mul(Scalar, tile_shape.at(i), rest_extent) catch return Error.Overflow);
        try out_stride.append(tile_stride.at(i));
    }
    for (tiler.len..rest_shape.len) |i| {
        try out_shape.append(rest_shape.at(i));
        const rest_stride = try rest_stride_tree.flattenLeaves();
        try out_stride.append(rest_stride.at(i));
    }
    return Layout.init(
        try treeFromFlat(out_shape.slice()),
        try treeFromFlat(out_stride.slice()),
    );
}

fn productEach(input: *const Tree) Error!Tree {
    switch (input.nodes.at(input.root)) {
        .leaf => return input.*,
        .tuple => |span| {
            var parts: [layout.max_children]Tree = undefined;
            if (span.len > parts.len) return Error.OutOfCapacity;
            for (0..span.len) |i| {
                const child = try input.subtree(input.children.at(span.start + i));
                parts[i] = try Tree.initLeaf(@intCast(try child.product()));
            }
            return Tree.initTuple(parts[0..span.len]);
        },
    }
}

fn orderedCompact(shape: Tree, order: []const usize) Error!Layout {
    const flat_shape = try shape.flattenLeaves();
    var strides: Flat = .{};
    for (0..flat_shape.len) |_| try strides.append(0);
    var stride_value: Scalar = 1;
    for (order) |idx| {
        strides.set(idx, stride_value);
        stride_value = std.math.mul(
            Scalar,
            stride_value,
            flat_shape.at(idx),
        ) catch return Error.Overflow;
    }
    return Layout.init(shape, try Tree.fromProfileAndLeaves(&shape, strides.slice()));
}

fn appendFlat(dst: *Flat, src: []const Scalar) Error!void {
    for (src) |v| try dst.append(v);
}

fn treeFromFlat(leaves: []const Scalar) Error!Tree {
    if (leaves.len == 0) return Error.InvalidShape;
    if (leaves.len == 1) return Tree.initLeaf(leaves[0]);
    var parts: [layout.max_leaves]Tree = undefined;
    if (leaves.len > parts.len) return Error.OutOfCapacity;
    for (leaves, 0..) |leaf, i| parts[i] = try Tree.initLeaf(leaf);
    return Tree.initTuple(parts[0..leaves.len]);
}

fn ceilDivScalar(a: Scalar, b: Scalar) Scalar {
    return @divTrunc(a + b - 1, b);
}

fn roundUpScalar(a: Scalar, b: Scalar) Scalar {
    return ceilDivScalar(a, b) * b;
}

test "layout_core: zipped divide preserves target offset formula" {
    const target = layout.makeLayout(.{ 8, 6 }, .{ 6, 1 });
    const tiler = Tree.fromComptime(.{ 2, 3 });
    const divided = try zippedDivide(&target, &tiler);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 2, 3, 4, 2 },
        (try divided.shape.flattenLeaves()).slice(),
    );
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 6, 1, 12, 3 },
        (try divided.stride.flattenLeaves()).slice(),
    );
    // tile=(1,2), rest=(3,1) -> original coord=(7,5)
    try std.testing.expectEqual(
        @as(Scalar, 47),
        try divided.crd2idxFlat(&.{ 1, 2, 3, 1 }),
    );
}

test "layout_core: logical divide interleaves tile and rest modes" {
    const target = layout.makeLayout(.{ 8, 6 }, .{ 6, 1 });
    const tiler = Tree.fromComptime(.{ 2, 3 });
    const divided = try logicalDivide(&target, &tiler);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 2, 4, 3, 2 },
        (try divided.shape.flattenLeaves()).slice(),
    );
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 6, 12, 1, 3 },
        (try divided.stride.flattenLeaves()).slice(),
    );
    try std.testing.expectEqual(
        @as(Scalar, 47),
        try divided.crd2idxFlat(&.{ 1, 3, 2, 1 }),
    );
}

test "layout_core: flat divide groups tile leaves before rest leaves" {
    const target = layout.makeLayout(.{ 8, 6, 5 }, .{ 30, 5, 1 });
    const tiler = Tree.fromComptime(.{ 2, 3 });
    const divided = try flatDivide(&target, &tiler);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 2, 3, 1, 4, 2, 5 },
        (try divided.shape.flattenLeaves()).slice(),
    );
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 30, 5, 1, 60, 15, 1 },
        (try divided.stride.flattenLeaves()).slice(),
    );
}

test "layout_core: product fuses divided layout back to target extents" {
    const target = layout.makeLayout(.{ 8, 6 }, .{ 6, 1 });
    const tiler = Tree.fromComptime(.{ 2, 3 });
    const divided = try zippedDivide(&target, &tiler);
    const fused = try zippedProduct(&divided, &tiler);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 8, 6 },
        (try fused.shape.flattenLeaves()).slice(),
    );
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 6, 1 },
        (try fused.stride.flattenLeaves()).slice(),
    );
}

test "layout_core: nullspace selects zero-stride modes" {
    const l = layout.makeLayout(.{ 2, 3, 4 }, .{ 12, 0, 0 });
    const n = try nullspace(&l);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 3, 4 },
        (try n.shape.flattenLeaves()).slice(),
    );
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 1, 3 },
        (try n.stride.flattenLeaves()).slice(),
    );
}

test "layout_core: leading dimension and image mask" {
    const l = layout.makeLayout(.{ 4, 4 }, .{ 4, 1 });
    try std.testing.expectEqual(@as(?usize, 1), try leadingDim(&l.shape, &l.stride));
    const mask = try makeLayoutImageMask(&l, &.{ 2, 0 }, 1);
    try std.testing.expectEqual(@as(u16, 0b1111_0000_0000), mask);
}

test "layout_core: max common vector and layout" {
    const a = layout.makeLayout(.{ 8, 4 }, .{ 4, 1 });
    const b = layout.makeLayout(.{ 8, 2 }, .{ 2, 1 });
    try std.testing.expectEqual(@as(Scalar, 2), try maxCommonVector(&a, &b));
    const c = try maxCommonLayout(&a, &b);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{2},
        (try c.shape.flattenLeaves()).slice(),
    );
    try std.testing.expectEqualSlices(
        Scalar,
        &.{1},
        (try c.stride.flattenLeaves()).slice(),
    );
}

test "layout_core: tile to shape pads and orders compact layout" {
    const atom = layout.makeCompactLayout(.{ 2, 3 });
    const target = Tree.fromComptime(.{ 7, 5 });
    const out = try tileToShape(&atom, &target, &.{ 1, 0 });
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 8, 6 },
        (try out.shape.flattenLeaves()).slice(),
    );
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 6, 1 },
        (try out.stride.flattenLeaves()).slice(),
    );
}
