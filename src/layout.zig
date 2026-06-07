const std = @import("std");

/// Core CuTe-style layout algebra for not-cute.
///
/// A Layout is a finite hierarchical shape tree paired with a congruent stride
/// tree. Coordinates are hierarchical trees with the same profile as shape.
/// The coordinate-to-index function is the usual generalized strided mapping:
///   crd2idx(c, s, d) = sum_i flatten(c)[i] * flatten(d)[i]
/// with bounds checked against flatten(s).
///
/// The implementation is deliberately allocation-free. All trees and flat
/// vectors are stored in bounded value types so the same code can run at
/// comptime and runtime without dependencies.
pub const Scalar = i128;
pub const Unsigned = u128;

pub const max_nodes = 256;
pub const max_children = 256;
pub const max_leaves = 128;

pub const Error = error{
    OutOfCapacity,
    InvalidShape,
    ProfileMismatch,
    RankMismatch,
    CoordinateOutOfBounds,
    Overflow,
    NotCompact,
    NotInjective,
    NotRepresentableAsAffineLayout,
    EmptyTuple,
    InvalidSelection,
};

pub fn BoundedList(comptime T: type, comptime capacity: usize) type {
    return struct {
        const Self = @This();

        items: [capacity]T = undefined,
        len: usize = 0,

        pub fn append(self: *Self, item: T) Error!void {
            if (self.len >= capacity) return Error.OutOfCapacity;
            self.items[self.len] = item;
            self.len += 1;
        }

        pub fn appendSlice(self: *Self, values: []const T) Error!void {
            if (values.len > capacity - self.len) return Error.OutOfCapacity;
            @memcpy(self.items[self.len..][0..values.len], values);
            self.len += values.len;
        }

        pub fn clear(self: *Self) void {
            self.len = 0;
        }

        pub fn slice(self: *const Self) []const T {
            return self.items[0..self.len];
        }

        pub fn mutableSlice(self: *Self) []T {
            return self.items[0..self.len];
        }

        pub fn at(self: *const Self, index: usize) T {
            std.debug.assert(index < self.len);
            return self.items[index];
        }

        pub fn set(self: *Self, index: usize, item: T) void {
            std.debug.assert(index < self.len);
            self.items[index] = item;
        }
    };
}

pub const Flat = BoundedList(Scalar, max_leaves);

pub const Span = struct {
    start: usize,
    len: usize,
};

pub const Node = union(enum) {
    leaf: Scalar,
    tuple: Span,
};

pub const Tree = struct {
    const Self = @This();

    nodes: BoundedList(Node, max_nodes) = .{},
    children: BoundedList(u16, max_children) = .{},
    root: u16 = 0,

    pub fn initLeaf(value: Scalar) Error!Self {
        var out: Self = .{};
        try out.nodes.append(.{ .leaf = value });
        out.root = 0;
        return out;
    }

    pub fn initTuple(parts: []const Self) Error!Self {
        if (parts.len > max_children) return Error.OutOfCapacity;
        var out: Self = .{};
        var child_ids: [max_children]u16 = undefined;
        for (parts, 0..) |*part, i| {
            child_ids[i] = try out.copySubtreeFrom(part, part.root);
        }
        const child_start = out.children.len;
        for (child_ids[0..parts.len]) |child_id| try out.children.append(child_id);
        const root_index = out.nodes.len;
        try out.nodes.append(.{ .tuple = .{ .start = child_start, .len = parts.len } });
        out.root = @intCast(root_index);
        return out;
    }

    pub fn initEmptyTuple() Error!Self {
        return initTuple(&.{});
    }

    pub fn fromComptime(comptime spec: anytype) Self {
        return comptime blk: {
            var out: Self = .{};
            out.root = out.addComptime(spec) catch |err| switch (err) {
                Error.OutOfCapacity => @compileError("shape literal exceeds not-cute bounded tree capacity"),
                Error.EmptyTuple => @compileError("empty tuple shape literals are not valid"),
                else => @compileError("invalid compile-time tree literal"),
            };
            break :blk out;
        };
    }

    fn addComptime(self: *Self, comptime spec: anytype) Error!u16 {
        const T = @TypeOf(spec);
        switch (@typeInfo(T)) {
            .comptime_int, .int => {
                const index = self.nodes.len;
                try self.nodes.append(.{ .leaf = @as(Scalar, spec) });
                return @intCast(index);
            },
            .@"struct" => |info| {
                if (!info.is_tuple) @compileError("tree literals must be ints or tuple literals such as .{2, .{3, 4,},}");
                var child_ids: [info.fields.len]u16 = undefined;
                inline for (info.fields, 0..) |field, i| {
                    child_ids[i] = try self.addComptime(@field(spec, field.name));
                }
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
                inline for (0..info.len) |i| {
                    child_ids[i] = try self.addComptime(spec[i]);
                }
                const child_start = self.children.len;
                for (child_ids) |child_id| try self.children.append(child_id);
                const index = self.nodes.len;
                try self.nodes.append(.{
                    .tuple = .{ .start = child_start, .len = info.len },
                });
                return @intCast(index);
            },
            else => @compileError("shape literals must contain only integer leaves"),
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
                if (span.len > max_children) return Error.OutOfCapacity;
                var child_ids: [max_children]u16 = undefined;
                for (0..span.len) |offset| {
                    const src_child = src.children.at(span.start + offset);
                    child_ids[offset] = try self.copySubtreeFrom(src, src_child);
                }
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

    pub fn equals(self: *const Self, other: *const Self) bool {
        return self.equalSubtree(self.root, other, other.root);
    }

    fn equalSubtree(
        self: *const Self,
        lhs_id: u16,
        other: *const Self,
        rhs_id: u16,
    ) bool {
        const lhs = self.nodes.at(lhs_id);
        const rhs = other.nodes.at(rhs_id);
        switch (lhs) {
            .leaf => |lv| switch (rhs) {
                .leaf => |rv| return lv == rv,
                .tuple => return false,
            },
            .tuple => |ls| switch (rhs) {
                .leaf => return false,
                .tuple => |rs| {
                    if (ls.len != rs.len) return false;
                    for (0..ls.len) |i| {
                        if (!self.equalSubtree(self.children.at(ls.start + i), other, other.children.at(rs.start + i)))
                            return false;
                    }
                    return true;
                },
            },
        }
    }

    pub fn sameProfile(self: *const Self, other: *const Self) bool {
        return self.sameProfileSubtree(self.root, other, other.root);
    }

    fn sameProfileSubtree(
        self: *const Self,
        lhs_id: u16,
        other: *const Self,
        rhs_id: u16,
    ) bool {
        const lhs = self.nodes.at(lhs_id);
        const rhs = other.nodes.at(rhs_id);
        switch (lhs) {
            .leaf => switch (rhs) {
                .leaf => return true,
                .tuple => return false,
            },
            .tuple => |ls| switch (rhs) {
                .leaf => return false,
                .tuple => |rs| {
                    if (ls.len != rs.len) return false;
                    for (0..ls.len) |i| {
                        if (!self.sameProfileSubtree(self.children.at(ls.start + i), other, other.children.at(rs.start + i)))
                            return false;
                    }
                    return true;
                },
            },
        }
    }

    pub fn assertPositive(self: *const Self) Error!void {
        try self.assertPositiveSubtree(self.root);
    }

    fn assertPositiveSubtree(self: *const Self, id: u16) Error!void {
        switch (self.nodes.at(id)) {
            .leaf => |value| if (value <= 0) return Error.InvalidShape,
            .tuple => |span| for (0..span.len) |i| try self.assertPositiveSubtree(self.children.at(span.start + i)),
        }
    }

    pub fn flattenLeaves(self: *const Self) Error!Flat {
        var out: Flat = .{};
        try self.flattenInto(self.root, &out);
        return out;
    }

    fn flattenInto(self: *const Self, id: u16, out: *Flat) Error!void {
        switch (self.nodes.at(id)) {
            .leaf => |value| try out.append(value),
            .tuple => |span| for (0..span.len) |i| try self.flattenInto(
                self.children.at(span.start + i),
                out,
            ),
        }
    }

    pub fn leafCount(self: *const Self) usize {
        return self.leafCountSubtree(self.root);
    }

    fn leafCountSubtree(self: *const Self, id: u16) usize {
        return switch (self.nodes.at(id)) {
            .leaf => 1,
            .tuple => |span| blk: {
                var total: usize = 0;
                for (0..span.len) |i| total += self.leafCountSubtree(self.children.at(span.start + i));
                break :blk total;
            },
        };
    }

    pub fn rank(self: *const Self) usize {
        return switch (self.nodes.at(self.root)) {
            .leaf => 1,
            .tuple => |span| span.len,
        };
    }

    pub fn depth(self: *const Self) usize {
        return self.depthSubtree(self.root);
    }

    fn depthSubtree(self: *const Self, id: u16) usize {
        return switch (self.nodes.at(id)) {
            .leaf => 0,
            .tuple => |span| blk: {
                var max_child: usize = 0;
                for (0..span.len) |i| max_child = @max(
                    max_child,
                    self.depthSubtree(self.children.at(span.start + i)),
                );
                break :blk max_child + 1;
            },
        };
    }

    pub fn product(self: *const Self) Error!Unsigned {
        const flat = try self.flattenLeaves();
        var total: Unsigned = 1;
        for (flat.slice()) |value| {
            if (value <= 0) return Error.InvalidShape;
            const u: Unsigned = @intCast(value);
            total = std.math.mul(Unsigned, total, u) catch return Error.Overflow;
        }
        return total;
    }

    pub fn fromProfileAndLeaves(
        profile: *const Self,
        leaves: []const Scalar,
    ) Error!Self {
        var out: Self = .{};
        var cursor: usize = 0;
        out.root = try out.copyProfileReplacingLeaves(
            profile,
            profile.root,
            leaves,
            &cursor,
        );
        if (cursor != leaves.len) return Error.RankMismatch;
        return out;
    }

    fn copyProfileReplacingLeaves(
        self: *Self,
        profile: *const Self,
        profile_id: u16,
        leaves: []const Scalar,
        cursor: *usize,
    ) Error!u16 {
        switch (profile.nodes.at(profile_id)) {
            .leaf => {
                if (cursor.* >= leaves.len) return Error.RankMismatch;
                const index = self.nodes.len;
                try self.nodes.append(.{ .leaf = leaves[cursor.*] });
                cursor.* += 1;
                return @intCast(index);
            },
            .tuple => |span| {
                if (span.len > max_children) return Error.OutOfCapacity;
                var child_ids: [max_children]u16 = undefined;
                for (0..span.len) |i| {
                    child_ids[i] = try self.copyProfileReplacingLeaves(
                        profile,
                        profile.children.at(span.start + i),
                        leaves,
                        cursor,
                    );
                }
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

    pub fn subtree(self: *const Self, id: u16) Error!Self {
        var out: Self = .{};
        out.root = try out.copySubtreeFrom(self, id);
        return out;
    }

    pub fn topMode(self: *const Self, mode: usize) Error!Self {
        switch (self.nodes.at(self.root)) {
            .leaf => {
                if (mode != 0) return Error.InvalidSelection;
                return self.*;
            },
            .tuple => |span| {
                if (mode >= span.len) return Error.InvalidSelection;
                return self.subtree(self.children.at(span.start + mode));
            },
        }
    }
};

pub const Layout = struct {
    const Self = @This();

    shape: Tree,
    stride: Tree,

    pub fn init(shape: Tree, stride: Tree) Error!Self {
        if (!shape.sameProfile(&stride)) return Error.ProfileMismatch;
        try shape.assertPositive();
        return .{ .shape = shape, .stride = stride };
    }

    pub fn make(comptime shape_spec: anytype, comptime stride_spec: anytype) Self {
        return comptime blk: {
            const s = Tree.fromComptime(shape_spec);
            const d = Tree.fromComptime(stride_spec);
            break :blk init(s, d) catch |err| switch (err) {
                Error.ProfileMismatch => @compileError("layout literal shape/stride profiles do not match"),
                Error.InvalidShape => @compileError("layout literal shape has a non-positive extent"),
                else => @compileError("invalid layout literal"),
            };
        };
    }

    pub fn compact(comptime shape_spec: anytype) Self {
        return comptime blk: {
            const shape_tree = Tree.fromComptime(shape_spec);
            break :blk makeCompact(shape_tree) catch |err| switch (err) {
                Error.InvalidShape => @compileError("compact layout shape has a non-positive extent"),
                Error.Overflow => @compileError("compact layout stride computation overflowed"),
                Error.OutOfCapacity => @compileError("compact layout exceeds not-cute bounded tree capacity"),
                else => @compileError("invalid compact layout shape"),
            };
        };
    }

    pub fn makeCompact(shape: Tree) Error!Self {
        return makeCompactWithOrder(shape, .left);
    }

    pub fn makeCompactRight(shape: Tree) Error!Self {
        return makeCompactWithOrder(shape, .right);
    }

    const CompactOrder = enum { left, right };

    fn makeCompactWithOrder(shape: Tree, order: CompactOrder) Error!Self {
        try shape.assertPositive();
        const flat = try shape.flattenLeaves();
        var strides: Flat = .{};
        for (0..flat.len) |_| try strides.append(0);

        var stride_value: Scalar = 1;
        switch (order) {
            .left => for (0..flat.len) |i| {
                strides.set(i, stride_value);
                stride_value = std.math.mul(
                    Scalar,
                    stride_value,
                    flat.at(i),
                ) catch return Error.Overflow;
            },
            .right => {
                var i = flat.len;
                while (i > 0) {
                    i -= 1;
                    strides.set(i, stride_value);
                    stride_value = std.math.mul(
                        Scalar,
                        stride_value,
                        flat.at(i),
                    ) catch return Error.Overflow;
                }
            },
        }

        const stride_tree = try Tree.fromProfileAndLeaves(&shape, strides.slice());
        return init(shape, stride_tree);
    }

    pub fn rank(self: *const Self) usize {
        return self.shape.rank();
    }

    pub fn leafCount(self: *const Self) usize {
        return self.shape.leafCount();
    }

    pub fn depth(self: *const Self) usize {
        return self.shape.depth();
    }

    pub fn size(self: *const Self) Error!Unsigned {
        return self.shape.product();
    }

    pub fn cosize(self: *const Self) Error!Unsigned {
        const shapes = try self.shape.flattenLeaves();
        const strides = try self.stride.flattenLeaves();
        var max_index: Unsigned = 0;
        for (shapes.slice(), 0..) |extent, i| {
            if (extent <= 0) return Error.InvalidShape;
            const span: Unsigned = @intCast(extent - 1);
            const abs_stride = absScalar(strides.at(i));
            const term = std.math.mul(
                Unsigned,
                span,
                abs_stride,
            ) catch return Error.Overflow;
            max_index = std.math.add(
                Unsigned,
                max_index,
                term,
            ) catch return Error.Overflow;
        }
        return std.math.add(Unsigned, max_index, 1) catch return Error.Overflow;
    }

    pub fn crd2idx(self: *const Self, coord: Tree) Error!Scalar {
        if (!coord.sameProfile(&self.shape)) return Error.ProfileMismatch;
        return self.crd2idxSubtree(
            coord.root,
            &coord,
            self.shape.root,
            self.stride.root,
        );
    }

    fn crd2idxSubtree(
        self: *const Self,
        coord_id: u16,
        coord: *const Tree,
        shape_id: u16,
        stride_id: u16,
    ) Error!Scalar {
        const cnode = coord.nodes.at(coord_id);
        const snode = self.shape.nodes.at(shape_id);
        const dnode = self.stride.nodes.at(stride_id);
        switch (snode) {
            .leaf => |extent| {
                const c = switch (cnode) {
                    .leaf => |value| value,
                    .tuple => return Error.ProfileMismatch,
                };
                const d = switch (dnode) {
                    .leaf => |value| value,
                    .tuple => return Error.ProfileMismatch,
                };
                if (c < 0 or c >= extent) return Error.CoordinateOutOfBounds;
                return std.math.mul(Scalar, c, d) catch Error.Overflow;
            },
            .tuple => |sspan| {
                const cspan = switch (cnode) {
                    .tuple => |span| span,
                    .leaf => return Error.ProfileMismatch,
                };
                const dspan = switch (dnode) {
                    .tuple => |span| span,
                    .leaf => return Error.ProfileMismatch,
                };
                if (cspan.len != sspan.len or dspan.len != sspan.len)
                    return Error.ProfileMismatch;
                var total: Scalar = 0;
                for (0..sspan.len) |i| {
                    const term = try self.crd2idxSubtree(
                        coord.children.at(cspan.start + i),
                        coord,
                        self.shape.children.at(sspan.start + i),
                        self.stride.children.at(dspan.start + i),
                    );
                    total = std.math.add(
                        Scalar,
                        total,
                        term,
                    ) catch return Error.Overflow;
                }
                return total;
            },
        }
    }

    pub fn crd2idxFlat(self: *const Self, coords: []const Scalar) Error!Scalar {
        const shapes = try self.shape.flattenLeaves();
        const strides = try self.stride.flattenLeaves();
        if (coords.len != shapes.len) return Error.RankMismatch;
        var total: Scalar = 0;
        for (coords, 0..) |coord, i| {
            const extent = shapes.at(i);
            if (coord < 0 or coord >= extent) return Error.CoordinateOutOfBounds;
            const term = std.math.mul(
                Scalar,
                coord,
                strides.at(i),
            ) catch return Error.Overflow;
            total = std.math.add(Scalar, total, term) catch return Error.Overflow;
        }
        return total;
    }

    pub fn idx2crdFlat(self: *const Self, idx: Scalar) Error!Flat {
        if (idx < 0) return Error.CoordinateOutOfBounds;
        if (!try self.isCompact()) return Error.NotCompact;
        const cosize_value = try self.cosize();
        if (@as(Unsigned, @intCast(idx)) >= cosize_value)
            return Error.CoordinateOutOfBounds;

        const shapes = try self.shape.flattenLeaves();
        const strides = try self.stride.flattenLeaves();
        var out: Flat = .{};
        for (0..shapes.len) |_| try out.append(0);
        for (0..shapes.len) |i| {
            const stride_abs = absScalar(strides.at(i));
            const stride_signed: Scalar = @intCast(stride_abs);
            const coord = @mod(@divFloor(idx, stride_signed), shapes.at(i));
            out.set(i, coord);
        }
        return out;
    }

    pub fn isCompact(self: *const Self) Error!bool {
        const shapes = try self.shape.flattenLeaves();
        const strides = try self.stride.flattenLeaves();
        if (shapes.len != strides.len) return Error.ProfileMismatch;
        if (shapes.len == 0) return true;

        var used = [_]bool{false} ** max_leaves;
        var expected: Unsigned = 1;
        var consumed: usize = 0;
        while (consumed < shapes.len) : (consumed += 1) {
            var found: ?usize = null;
            for (0..shapes.len) |i| {
                if (used[i]) continue;
                if (shapes.at(i) == 1) {
                    found = i;
                    break;
                }
                if (absScalar(strides.at(i)) == expected) {
                    found = i;
                    break;
                }
            }
            const j = found orelse return false;
            used[j] = true;
            if (shapes.at(j) > 1) {
                const extent: Unsigned = @intCast(shapes.at(j));
                expected = std.math.mul(
                    Unsigned,
                    expected,
                    extent,
                ) catch return Error.Overflow;
            }
        }
        return expected == try self.size();
    }

    pub fn flatten(self: *const Self) Error!Self {
        const shapes = try self.shape.flattenLeaves();
        const strides = try self.stride.flattenLeaves();
        const s = try tryTreeFromFlat(shapes.slice());
        const d = try tryTreeFromFlat(strides.slice());
        return init(s, d);
    }

    pub fn topMode(self: *const Self, mode: usize) Error!Self {
        const s = try self.shape.topMode(mode);
        const d = try self.stride.topMode(mode);
        return init(s, d);
    }

    pub fn concatenate(layouts: []const Self) Error!Self {
        if (layouts.len == 0) return Error.EmptyTuple;
        var shape_parts: [max_leaves]Tree = undefined;
        var stride_parts: [max_leaves]Tree = undefined;
        if (layouts.len > max_leaves) return Error.OutOfCapacity;
        for (layouts, 0..) |layout, i| {
            shape_parts[i] = layout.shape;
            stride_parts[i] = layout.stride;
        }
        const s = try Tree.initTuple(shape_parts[0..layouts.len]);
        const d = try Tree.initTuple(stride_parts[0..layouts.len]);
        return init(s, d);
    }

    pub fn coalesce(self: *const Self) Error!Self {
        const shapes = try self.shape.flattenLeaves();
        const strides = try self.stride.flattenLeaves();
        if (shapes.len != strides.len) return Error.ProfileMismatch;
        if (shapes.len == 0) return Error.InvalidShape;

        var new_shapes: Flat = .{};
        var new_strides: Flat = .{};

        var i = shapes.len;
        while (i > 0) {
            i -= 1;
            const old_shape = shapes.at(i);
            const old_stride = strides.at(i);
            if (old_shape == 1) continue;
            if (new_shapes.len == 0) {
                try prepend(&new_shapes, old_shape);
                try prepend(&new_strides, old_stride);
                continue;
            }
            const front_shape = new_shapes.at(0);
            const front_stride = new_strides.at(0);
            const contiguous = (std.math.mul(Scalar, old_shape, old_stride) catch return Error.Overflow) == front_stride;
            if (contiguous) {
                new_shapes.set(0, std.math.mul(Scalar, old_shape, front_shape) catch return Error.Overflow);
                new_strides.set(0, old_stride);
            } else {
                try prepend(&new_shapes, old_shape);
                try prepend(&new_strides, old_stride);
            }
        }

        if (new_shapes.len == 0) {
            try new_shapes.append(1);
            try new_strides.append(0);
        }

        const s = try tryTreeFromFlat(new_shapes.slice());
        const d = try tryTreeFromFlat(new_strides.slice());
        return init(s, d);
    }

    /// Composition as a normal affine layout when the result is representable.
    /// Falls back to Error.NotRepresentableAsAffineLayout instead of silently
    /// producing a wrong shape/stride pair.
    pub fn composeAffine(a: *const Self, b: *const Self) Error!Self {
        if (!try a.isCompact()) return Error.NotRepresentableAsAffineLayout;
        _ = try b.size();

        const b_shape = b.shape;
        const b_strides = try b.stride.flattenLeaves();
        var result_strides: Flat = .{};

        const zero_coords: [max_leaves]Scalar = [_]Scalar{0} ** max_leaves;
        const b_rank = b.leafCount();
        const base_index = try b.crd2idxFlat(zero_coords[0..b_rank]);
        const base = try a.indexFromLinear(base_index);
        if (base != 0) return Error.NotRepresentableAsAffineLayout;

        const b_shapes = try b.shape.flattenLeaves();
        for (0..b_strides.len) |i| {
            if (b_shapes.at(i) == 1) {
                try result_strides.append(0);
                continue;
            }
            var basis: [max_leaves]Scalar = [_]Scalar{0} ** max_leaves;
            basis[i] = 1;
            const lin = try b.crd2idxFlat(basis[0..b_rank]);
            const mapped = try a.indexFromLinear(lin);
            try result_strides.append(mapped);
        }
        const d = try Tree.fromProfileAndLeaves(&b_shape, result_strides.slice());
        return init(b_shape, d);
    }

    pub fn indexFromLinear(self: *const Self, linear: Scalar) Error!Scalar {
        const coord = try self.idx2crdFlat(linear);
        return self.crd2idxFlat(coord.slice());
    }
};

pub const ComposedLayout = struct {
    /// Represents A(offset + B(coord)). This intentionally mirrors CuTe's
    /// ComposedLayout for cases where affine composition is not legal.
    a: Layout,
    offset: Scalar = 0,
    b: Layout,

    pub fn shape(self: *const ComposedLayout) Tree {
        return self.b.shape;
    }

    pub fn size(self: *const ComposedLayout) Error!Unsigned {
        return self.b.size();
    }

    pub fn crd2idx(self: *const ComposedLayout, coord: Tree) Error!Scalar {
        const inner = try self.b.crd2idx(coord);
        return self.a.indexFromLinear(std.math.add(Scalar, self.offset, inner) catch return Error.Overflow);
    }

    pub fn crd2idxFlat(
        self: *const ComposedLayout,
        coords: []const Scalar,
    ) Error!Scalar {
        const inner = try self.b.crd2idxFlat(coords);
        return self.a.indexFromLinear(std.math.add(Scalar, self.offset, inner) catch return Error.Overflow);
    }
};

pub const AnyLayout = union(enum) {
    affine: Layout,
    composed: ComposedLayout,

    pub fn crd2idxFlat(self: *const AnyLayout, coords: []const Scalar) Error!Scalar {
        return switch (self.*) {
            .affine => |*layout| layout.crd2idxFlat(coords),
            .composed => |*layout| layout.crd2idxFlat(coords),
        };
    }
};

pub fn composition(a: Layout, b: Layout) Error!AnyLayout {
    if (a.composeAffine(&b)) |layout| {
        return .{ .affine = layout };
    } else |err| switch (err) {
        Error.NotRepresentableAsAffineLayout, Error.NotCompact => return .{
            .composed = .{ .a = a, .offset = 0, .b = b },
        },
        else => return err,
    }
}

pub fn makeLayout(comptime shape_spec: anytype, comptime stride_spec: anytype) Layout {
    return Layout.make(shape_spec, stride_spec);
}

pub fn makeCompactLayout(comptime shape_spec: anytype) Layout {
    return Layout.compact(shape_spec);
}

fn absScalar(value: Scalar) Unsigned {
    if (value < 0) return @intCast(-value);
    return @intCast(value);
}

fn prepend(list: *Flat, value: Scalar) Error!void {
    if (list.len >= max_leaves) return Error.OutOfCapacity;
    var i = list.len;
    while (i > 0) {
        list.items[i] = list.items[i - 1];
        i -= 1;
    }
    list.items[0] = value;
    list.len += 1;
}

fn tryFlatProfile(len: usize) Error!Tree {
    var leaves: Flat = .{};
    for (0..len) |_| try leaves.append(0);
    return tryTreeFromFlat(leaves.slice());
}

fn tryTreeFromFlat(leaves: []const Scalar) Error!Tree {
    if (leaves.len == 0) return Error.InvalidShape;
    if (leaves.len == 1) return Tree.initLeaf(leaves[0]);
    var parts: [max_leaves]Tree = undefined;
    if (leaves.len > max_leaves) return Error.OutOfCapacity;
    for (leaves, 0..) |leaf, i| parts[i] = try Tree.initLeaf(leaf);
    return Tree.initTuple(parts[0..leaves.len]);
}

test "layout: compact construction and coordinate mapping" {
    const l = makeCompactLayout(.{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 3), l.leafCount());
    try std.testing.expectEqual(@as(Unsigned, 24), try l.size());
    try std.testing.expectEqual(
        @as(Scalar, 1 + 2 * 2 + 3 * 6),
        try l.crd2idxFlat(&.{ 1, 2, 3 }),
    );
}

test "layout: bounded appendSlice is atomic on overflow" {
    var list: BoundedList(u8, 3) = .{};
    try list.append(1);
    try std.testing.expectError(Error.OutOfCapacity, list.appendSlice(&.{ 2, 3, 4 }));
    try std.testing.expectEqualSlices(u8, &.{1}, list.slice());
}

test "layout: hierarchical profile is preserved" {
    const l = makeLayout(.{ 2, .{ 3, 4 } }, .{ 1, .{ 2, 6 } });
    const coord = Tree.fromComptime(.{ 1, .{ 2, 3 } });
    try std.testing.expectEqual(@as(Scalar, 23), try l.crd2idx(coord));
    try std.testing.expectEqual(@as(usize, 2), l.rank());
    try std.testing.expectEqual(@as(usize, 2), l.depth());
}

test "layout: coalesce contiguous modes" {
    const l = makeLayout(.{ 2, 3, 4 }, .{ 1, 2, 6 });
    const c = try l.coalesce();
    try std.testing.expectEqual(@as(usize, 1), c.leafCount());
    try std.testing.expectEqual(@as(Scalar, 23), try c.crd2idxFlat(&.{23}));
}

test "layout: coalesce removes unit modes" {
    const l = makeLayout(.{ 1, 2, 1, 3 }, .{ 0, 1, 0, 2 });
    const c = try l.coalesce();
    try std.testing.expectEqual(@as(usize, 1), c.leafCount());
    try std.testing.expectEqual(@as(Unsigned, 6), try c.size());
}

test "layout: compact inverse coordinate" {
    const l = makeCompactLayout(.{ 2, 3, 4 });
    const coord = try l.idx2crdFlat(23);
    try std.testing.expectEqualSlices(Scalar, &.{ 1, 2, 3 }, coord.slice());
}

test "layout: nested compact preserves profile and leaf order" {
    const l = makeCompactLayout(.{ 2, .{ 3, 4 } });
    const expected_stride = Tree.fromComptime(.{ 1, .{ 2, 6 } });
    try std.testing.expect(l.stride.equals(&expected_stride));

    const coord = Tree.fromComptime(.{ 1, .{ 2, 3 } });
    try std.testing.expectEqual(@as(Scalar, 23), try l.crd2idx(coord));
}

test "layout: compact-right construction" {
    const shape = Tree.fromComptime(.{ 2, 3, 4 });
    const l = try Layout.makeCompactRight(shape);
    try std.testing.expectEqual(
        @as(Scalar, 12 + 2 * 4 + 3),
        try l.crd2idxFlat(&.{ 1, 2, 3 }),
    );
    try std.testing.expect(try l.isCompact());
}

test "layout: concatenate and top-mode selection keep nested children valid" {
    const a = makeLayout(.{ 2, .{ 3, 4 } }, .{ 1, .{ 2, 6 } });
    const b = makeLayout(.{ 5, 6 }, .{ 24, 120 });
    const cat = try Layout.concatenate(&.{ a, b });
    try std.testing.expectEqual(@as(usize, 2), cat.rank());
    try std.testing.expectEqual(@as(usize, 5), cat.leafCount());

    const first = try cat.topMode(0);
    try std.testing.expect(first.shape.equals(&a.shape));
    try std.testing.expect(first.stride.equals(&a.stride));
}

test "layout: composition handles unit-extent modes" {
    const a = makeCompactLayout(.{ 2, 3 });
    const b = makeLayout(.{ 1, 3 }, .{ 0, 1 });
    const c = try a.composeAffine(&b);
    try std.testing.expectEqual(@as(Scalar, 2), try c.crd2idxFlat(&.{ 0, 2 }));
}
