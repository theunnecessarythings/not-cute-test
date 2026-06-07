const core = @import("core.zig");
const std = @import("std");
const tuple = @import("tuple.zig");

pub const Error = error{
    CoordinateOutOfBounds,
    EmptyTuple,
    NonCongruentArguments,
    InvalidMode,
    InvalidSelection,
    InvalidShape,
    MaskDoesNotFit16Bits,
    NoCommonVector,
    NonAffineResult,
    NotCompact,
    NotInjective,
    NotRepresentableAsAffineLayout,
    OutOfCapacity,
    Overflow,
    ProfileMismatch,
    RankMismatch,
    EmptyInput,
    NonStaticInput,
    NotImplementedForDynamicValue,
    InvalidSelector,
    NegativeIndex,
    MissingCutlassIrLibrary,
    Invalidcutlass_emitFixture,
    InvalidTensorType,
    InvalidSharedLibrary,
    InvalidRoutedFixture,
    InvalidFullTiledFixture,
    InvalidPythonExecutable,
    InvalidPackageModule,
    InvalidDiscoveryJson,
    InvalidCuteTypePayload,
    InvalidCuteMemorySpace,
    InvalidBridgeScript,
    InvalidBridgeMode,
    InvalidBridgeConfig,
    InvalidAtomType,
    BridgeFailed,
    InvalidNumericType,
    UnknownOperation,
    InvalidTiler,
    InvalidAtomLayout,
    InvalidCopyBits,
    MissingRuntimeField,
    UnsupportedField,
    UnsupportedRank,
    TypeWidthMismatch,
    InvalidOperandCount,
    InvalidModeIndex,
    InvalidThreadIndex,
    MissingTraitLayout,
    WrongAtomKind,
    InvalidOperand,
    UnsupportedOperation,
    BroadcastMismatch,
    InvalidElementType,
    InvalidReductionProfile,
    InvalidGatherScatterMode,
    InvalidVectorRank,
    InvalidVectorSlice,
    InvalidVectorOrder,
    NarrowPrecisionAlignment,
    IncompatibleTensorShapes,
    IncompatibleTensorShape,
    TooManyTensorElements,
    MissingTensorValue,
    UnsupportedDynamicTensor,
    InvalidTensorOperation,
    InvalidTensorConstruction,
    EmptyTensorSsa,
    UnsupportedMaskedMemoryOperation,
    InvalidTensorAccess,
    InvalidTensorEngine,
    InvalidMaskShape,
    InvalidTensorShape,
    TooManyArguments,
    OutOfMemory,
    InvalidCudaStream,
    InvalidCudaModule,
    InvalidCudaMemory,
    InvalidCudaFunction,
    InvalidCudaArgument,
    InvalidCString,
    FileReadFailed,
    CudaSymbolUnavailable,
    CudaDriverUnavailable,
    CudaCallFailed,
    UnterminatedString,
    UnbalancedRegion,
    TooManyResults,
    RegionUnderflow,
    NegativeTestUnexpectedSuccess,
    MissingTerminator,
    MissingExpectedDiagnostic,
    InvalidToolConfig,
    InvalidMlirType,
    InvalidMlirString,
    InvalidMlirOperation,
    InvalidMlirIdentifier,
    InvalidMlirAttribute,
    GoldenMismatch,
    EmptyCase,
    ToolNotConfigured,
    ToolFailed,
    TooManyPackedArguments,
    MisalignedPointer,
    InvalidToolPath,
    InvalidTensorDescriptor,
    InvalidSymbol,
    InvalidRuntimeSymbol,
    InvalidPackedArgument,
    InvalidLibraryPath,
    InvalidLaunchShape,
    InvalidDynamicMask,
    InvalidDevice,
    InvalidCompileOption,
    InvalidArtifactPath,
    InvalidExportName,
    InvalidDivisibility,
    InvalidNumericKind,
    BufferTooSmall,
    InvalidCompilerState,
    InvalidJitArgument,
    TooManySpecializations,
    DivisionByZero,
    PredicateArityUnsupported,
    UnsupportedDynamicOperation,
    InvalidSwizzle,
};

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

pub const Keep = enum { keep };
pub const keep: Keep = .keep;

pub const SelectorLeaf = union(enum) {
    keep,
    fixed: Scalar,
};

pub const SelectorNode = union(enum) {
    leaf: SelectorLeaf,
    tuple: Span,
};

/// Static coordinate/dicer profile used by layout slicing and dicing.
/// `keep` corresponds to CuteDSL's `None`/underscore convention in a slicing
/// coordinate: slice keeps modes tagged `keep`, dice drops modes tagged `keep`.
/// Integer leaves are fixed coordinates.
pub const Selector = struct {
    const Self = @This();

    nodes: BoundedList(SelectorNode, max_nodes) = .{},
    children: BoundedList(u16, max_children) = .{},
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
                if (@TypeOf(spec) != Keep or spec != .keep) @compileError("selector enum leaves must be keep");
                const index = self.nodes.len;
                try self.nodes.append(.{ .leaf = .keep });
                return @intCast(index);
            },
            .@"struct" => |info| {
                if (!info.is_tuple) @compileError("selector literals must be ints, keep, or tuple literals");
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
            else => @compileError("selector leaves must be ints or keep"),
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
    return appendLayout(input, makeCompactLayout(1), up_to_rank);
}

pub fn prependOnesLayout(input: *const Layout, up_to_rank: ?usize) Error!Layout {
    return prependLayout(input, makeCompactLayout(1), up_to_rank);
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

pub fn tileToShapeTree(tile: *const Tree, target_profile: *const Tree) Error!Tree {
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

    const l = makeLayout(.{ 2, 3, 4 }, .{ 1, 2, 6 });
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
    const l = makeLayout(.{ 4, 5 }, .{ 5, 1 });
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
    const l = makeLayout(.{ 4, 5, 6 }, .{ 30, 6, 1 });
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
    const l = makeCompactLayout(.{ 8, 8 });
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

pub const SliceOffset = SliceResult;

/// Source-grounded static `composition`: first try normal affine composition;
/// otherwise preserve exact semantics as a ComposedLayout fallback.
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
            var shape_parts: [max_children]Tree = undefined;
            var stride_parts: [max_children]Tree = undefined;
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
pub fn tileToShapeLayout(
    atom: *const Layout,
    target_shape: *const Tree,
    order: []const usize,
) Error!Layout {
    const target = try target_shape.flattenLeaves();
    const atom_shape = try atom.shape.flattenLeaves();
    if (atom_shape.len > target.len) return Error.RankMismatch;
    if (order.len != target.len) return Error.RankMismatch;
    var seen = [_]bool{false} ** max_leaves;
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

pub fn sliceAndOffsetFlip(
    selector: *const Selector,
    input: *const Layout,
) Error!SliceOffset {
    return sliceAndOffset(input, selector);
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
            var parts: [max_children]Tree = undefined;
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
    var parts: [max_leaves]Tree = undefined;
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
    const target = makeLayout(.{ 8, 6 }, .{ 6, 1 });
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
    const target = makeLayout(.{ 8, 6 }, .{ 6, 1 });
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
    const target = makeLayout(.{ 8, 6, 5 }, .{ 30, 5, 1 });
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
    const target = makeLayout(.{ 8, 6 }, .{ 6, 1 });
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
    const l = makeLayout(.{ 2, 3, 4 }, .{ 12, 0, 0 });
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
    const l = makeLayout(.{ 4, 4 }, .{ 4, 1 });
    try std.testing.expectEqual(@as(?usize, 1), try leadingDim(&l.shape, &l.stride));
    const mask = try makeLayoutImageMask(&l, &.{ 2, 0 }, 1);
    try std.testing.expectEqual(@as(u16, 0b1111_0000_0000), mask);
}

test "layout_core: max common vector and layout" {
    const a = makeLayout(.{ 8, 4 }, .{ 4, 1 });
    const b = makeLayout(.{ 8, 2 }, .{ 2, 1 });
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
    const atom = makeCompactLayout(.{ 2, 3 });
    const target = Tree.fromComptime(.{ 7, 5 });
    const out = try tileToShapeLayout(&atom, &target, &.{ 1, 0 });
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
