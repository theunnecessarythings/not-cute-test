const std = @import("std");
const layout = @import("layout.zig");

pub const Scalar = layout.Scalar;
pub const Unsigned = layout.Unsigned;
pub const Error = layout.Error || error{
    PredicateArityUnsupported,
    NonCongruentArguments,
    DivisionByZero,
};

pub const Tree = layout.Tree;
pub const Flat = layout.Flat;
pub const max_leaves = layout.max_leaves;
pub const Path = layout.BoundedList(usize, max_leaves);

/// Zig representation of CuteDSL tuple.py's wrap(x): a scalar is represented as
/// a leaf tree, while an existing Tree is already wrapped by the type system.
pub fn wrapScalar(value: Scalar) Error!Tree {
    return Tree.initLeaf(value);
}

/// Mirrors CuteDSL tuple.py unwrap(x): repeatedly remove a single-child tuple.
pub fn unwrapSingleton(input: Tree) Error!Tree {
    var current = input;
    while (true) {
        switch (current.nodes.at(current.root)) {
            .leaf => return current,
            .tuple => |span| {
                if (span.len != 1) return current;
                var next: Tree = .{};
                next.root = try copySubtree(
                    &next,
                    &current,
                    current.children.at(span.start),
                );
                current = next;
            },
        }
    }
}

/// Mirrors CuteDSL tuple.py flatten_to_tuple for static integer leaves.
pub fn flattenToTuple(input: *const Tree) Error!Flat {
    return input.flattenLeaves();
}

/// Mirrors CuteDSL tuple.py unflatten(sequence, profile).
pub fn unflatten(sequence: []const Scalar, profile: *const Tree) Error!Tree {
    return Tree.fromProfileAndLeaves(profile, sequence);
}

/// Mirrors CuteDSL tuple.py product for static shapes/int-tuples.
pub fn product(input: *const Tree) Error!Unsigned {
    return input.product();
}

/// Mirrors CuteDSL tuple.py product_each: each top-level mode is replaced by
/// the product of that mode. A scalar input returns a scalar copy.
pub fn productEach(input: *const Tree) Error!Tree {
    switch (input.nodes.at(input.root)) {
        .leaf => return input.*,
        .tuple => |span| {
            var parts: [layout.max_children]Tree = undefined;
            if (span.len > layout.max_children) return Error.OutOfCapacity;
            for (0..span.len) |i| {
                var mode_tree: Tree = .{};
                mode_tree.root = try copySubtree(
                    &mode_tree,
                    input,
                    input.children.at(span.start + i),
                );
                const p = try mode_tree.product();
                if (p > std.math.maxInt(Scalar)) return Error.Overflow;
                parts[i] = try Tree.initLeaf(@intCast(p));
            }
            return Tree.initTuple(parts[0..span.len]);
        },
    }
}

/// Mirrors CuteDSL tuple.py product_like for static integer trees.
pub fn productLike(input: *const Tree, target_profile: *const Tree) Error!Tree {
    var out: Tree = .{};
    out.root = try productLikeSubtree(
        &out,
        input,
        input.root,
        target_profile,
        target_profile.root,
    );
    return out;
}

fn productLikeSubtree(
    out: *Tree,
    input: *const Tree,
    input_id: u16,
    profile: *const Tree,
    profile_id: u16,
) Error!u16 {
    switch (profile.nodes.at(profile_id)) {
        .leaf => {
            var subtree: Tree = .{};
            subtree.root = try copySubtree(&subtree, input, input_id);
            const p = try subtree.product();
            if (p > std.math.maxInt(Scalar)) return Error.Overflow;
            const idx = out.nodes.len;
            try out.nodes.append(.{ .leaf = @intCast(p) });
            return @intCast(idx);
        },
        .tuple => |pspan| {
            const ispan = switch (input.nodes.at(input_id)) {
                .tuple => |span| span,
                .leaf => return Error.ProfileMismatch,
            };
            if (ispan.len != pspan.len) return Error.ProfileMismatch;
            var child_ids: [layout.max_children]u16 = undefined;
            if (pspan.len > layout.max_children) return Error.OutOfCapacity;
            for (0..pspan.len) |i| {
                child_ids[i] = try productLikeSubtree(
                    out,
                    input,
                    input.children.at(ispan.start + i),
                    profile,
                    profile.children.at(pspan.start + i),
                );
            }
            const child_start = out.children.len;
            for (child_ids[0..pspan.len]) |child_id| try out.children.append(child_id);
            const idx = out.nodes.len;
            try out.nodes.append(.{
                .tuple = .{ .start = child_start, .len = pspan.len },
            });
            return @intCast(idx);
        },
    }
}

pub const FindResult = struct {
    found: bool = false,
    /// Hierarchical path if hierarchical=true. Top-mode-only path if false.
    path: Path = .{},
    /// Zero-based flattened leaf index for the match.
    flat_leaf: usize = 0,
};

/// Source-grounded subset of tuple.py find(): static integer equality only.
pub fn findScalar(
    input: *const Tree,
    needle: Scalar,
    hierarchical: bool,
) Error!FindResult {
    var result: FindResult = .{};
    var path: Path = .{};
    var flat_index: usize = 0;
    try findScalarSubtree(
        input,
        input.root,
        needle,
        hierarchical,
        &path,
        &flat_index,
        &result,
    );
    return result;
}

fn findScalarSubtree(
    input: *const Tree,
    id: u16,
    needle: Scalar,
    hierarchical: bool,
    path: *Path,
    flat_index: *usize,
    result: *FindResult,
) Error!void {
    if (result.found) return;
    switch (input.nodes.at(id)) {
        .leaf => |value| {
            if (value == needle) {
                result.found = true;
                result.flat_leaf = flat_index.*;
                if (hierarchical) {
                    try result.path.appendSlice(path.slice());
                } else if (path.len > 0) {
                    try result.path.append(path.at(0));
                } else {
                    try result.path.append(0);
                }
                return;
            }
            flat_index.* += 1;
        },
        .tuple => |span| {
            for (0..span.len) |i| {
                try path.append(i);
                try findScalarSubtree(
                    input,
                    input.children.at(span.start + i),
                    needle,
                    hierarchical,
                    path,
                    flat_index,
                    result,
                );
                path.len -= 1;
                if (result.found) return;
            }
        },
    }
}

/// Mirrors transform_leaf for one static integer tree.
pub fn mapLeaves(input: *const Tree, comptime f: fn (Scalar) Scalar) Error!Tree {
    const flat = try input.flattenLeaves();
    var mapped: Flat = .{};
    for (flat.slice()) |value| try mapped.append(f(value));
    return Tree.fromProfileAndLeaves(input, mapped.slice());
}

/// Mirrors transform_leaf for two congruent static integer trees.
pub fn zipLeaves(
    lhs: *const Tree,
    rhs: *const Tree,
    comptime f: fn (Scalar, Scalar) Scalar,
) Error!Tree {
    if (!lhs.sameProfile(rhs)) return Error.ProfileMismatch;
    const lf = try lhs.flattenLeaves();
    const rf = try rhs.flattenLeaves();
    var mapped: Flat = .{};
    for (lf.slice(), 0..) |value, i| try mapped.append(f(value, rf.at(i)));
    return Tree.fromProfileAndLeaves(lhs, mapped.slice());
}

/// Mirrors tuple.py elem_less for static integer leaves.
pub fn elemLess(lhs: *const Tree, rhs: *const Tree) Error!bool {
    if (!lhs.sameProfile(rhs)) return Error.ProfileMismatch;
    const lf = try lhs.flattenLeaves();
    const rf = try rhs.flattenLeaves();
    for (lf.slice(), 0..) |value, i| {
        if (!(value < rf.at(i))) return false;
    }
    return true;
}

/// Mirrors tuple.py tuple_cat. A tuple root contributes its children; a leaf
/// contributes one element.
pub fn tupleCat(inputs: []const Tree) Error!Tree {
    if (inputs.len == 0) return Error.EmptyTuple;
    var parts: [layout.max_children]Tree = undefined;
    var count: usize = 0;
    for (inputs) |input| {
        switch (input.nodes.at(input.root)) {
            .leaf => {
                if (count >= parts.len) return Error.OutOfCapacity;
                parts[count] = input;
                count += 1;
            },
            .tuple => |span| {
                for (0..span.len) |i| {
                    if (count >= parts.len) return Error.OutOfCapacity;
                    parts[count] = try input.topMode(i);
                    count += 1;
                }
            },
        }
    }
    return Tree.initTuple(parts[0..count]);
}

/// Static equivalent of core.py repeat_as_tuple.
pub fn repeatAsTuple(value: Scalar, n: usize) Error!Tree {
    if (n < 1) return Error.InvalidShape;
    var parts: [layout.max_children]Tree = undefined;
    if (n > parts.len) return Error.OutOfCapacity;
    for (0..n) |i| parts[i] = try Tree.initLeaf(value);
    return Tree.initTuple(parts[0..n]);
}

/// Static equivalent of core.py repeat_like.
pub fn repeatLike(value: Scalar, target: *const Tree) Error!Tree {
    var leaves: Flat = .{};
    for (0..target.leafCount()) |_| try leaves.append(value);
    return Tree.fromProfileAndLeaves(target, leaves.slice());
}

/// A source-compatible alias for core.py rank on static trees.
pub fn rank(input: *const Tree) usize {
    return input.rank();
}

/// A source-compatible alias for core.py depth on static trees.
pub fn depth(input: *const Tree) usize {
    return input.depth();
}

/// Source-compatible structural congruence predicate.
pub fn isCongruent(lhs: *const Tree, rhs: *const Tree) bool {
    return lhs.sameProfile(rhs);
}

/// Source-compatible weak congruence: a leaf is weakly congruent to any target.
pub fn isWeaklyCongruent(lhs: *const Tree, rhs: *const Tree) bool {
    return weakCongruentSubtree(lhs, lhs.root, rhs, rhs.root);
}

fn weakCongruentSubtree(
    lhs: *const Tree,
    lhs_id: u16,
    rhs: *const Tree,
    rhs_id: u16,
) bool {
    return switch (lhs.nodes.at(lhs_id)) {
        .leaf => true,
        .tuple => |ls| switch (rhs.nodes.at(rhs_id)) {
            .leaf => false,
            .tuple => |rs| blk: {
                if (ls.len != rs.len) break :blk false;
                for (0..ls.len) |i| {
                    if (!weakCongruentSubtree(lhs, lhs.children.at(ls.start + i), rhs, rhs.children.at(rs.start + i))) break :blk false;
                }
                break :blk true;
            },
        },
    };
}

fn copySubtree(dst: *Tree, src: *const Tree, src_id: u16) Error!u16 {
    switch (src.nodes.at(src_id)) {
        .leaf => |value| {
            const index = dst.nodes.len;
            try dst.nodes.append(.{ .leaf = value });
            return @intCast(index);
        },
        .tuple => |span| {
            var child_ids: [layout.max_children]u16 = undefined;
            if (span.len > layout.max_children) return Error.OutOfCapacity;
            for (0..span.len) |i| child_ids[i] = try copySubtree(
                dst,
                src,
                src.children.at(span.start + i),
            );
            const child_start = dst.children.len;
            for (child_ids[0..span.len]) |child_id| try dst.children.append(child_id);
            const index = dst.nodes.len;
            try dst.nodes.append(.{
                .tuple = .{ .start = child_start, .len = span.len },
            });
            return @intCast(index);
        },
    }
}

test "tuple: wrap unwrap flatten and unflatten" {
    const wrapped = try wrapScalar(7);
    try std.testing.expectEqual(@as(usize, 1), wrapped.leafCount());

    const nested = Tree.fromComptime(.{.{7}});
    const unwrapped = try unwrapSingleton(nested);
    try std.testing.expect(unwrapped.equals(&wrapped));

    const profile = Tree.fromComptime(.{ 0, .{ 0, 0 } });
    const rebuilt = try unflatten(&.{ 2, 3, 4 }, &profile);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 2, 3, 4 },
        (try flattenToTuple(&rebuilt)).slice(),
    );
}

test "tuple: product_each and product_like" {
    const shape = Tree.fromComptime(.{ 2, .{ 3, 4 } });
    const each = try productEach(&shape);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 2, 12 },
        (try each.flattenLeaves()).slice(),
    );

    const target = Tree.fromComptime(.{ 0, 0 });
    const like = try productLike(&shape, &target);
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 2, 12 },
        (try like.flattenLeaves()).slice(),
    );
}

test "tuple: find elem_less tuple_cat repeat_like" {
    const t = Tree.fromComptime(.{ 3, .{ 4, 5 } });
    const found = try findScalar(&t, 5, true);
    try std.testing.expect(found.found);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, found.path.slice());
    try std.testing.expectEqual(@as(usize, 2), found.flat_leaf);

    const a = Tree.fromComptime(.{ 1, .{ 2, 3 } });
    const b = Tree.fromComptime(.{ 2, .{ 3, 4 } });
    try std.testing.expect(try elemLess(&a, &b));

    const cat = try tupleCat(&.{
        Tree.fromComptime(.{ 1, 2 }),
        Tree.fromComptime(.{3}),
    });
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 1, 2, 3 },
        (try cat.flattenLeaves()).slice(),
    );

    const repeated = try repeatLike(9, &t);
    try std.testing.expect(repeated.sameProfile(&t));
    try std.testing.expectEqualSlices(
        Scalar,
        &.{ 9, 9, 9 },
        (try repeated.flattenLeaves()).slice(),
    );
}
