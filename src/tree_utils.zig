const std = @import("std");
const layout = @import("layout.zig");

pub const Error = layout.Error || error{ InvalidTreeDef, TooManyLeaves, TypeMismatch };
pub const DSLTreeFlattenError = Error;
pub const NodeType = enum { leaf, node, tuple, array, struct_ };
pub const Leaf = struct { index: usize, type_name: []const u8 = "" };
pub const PyTreeDef = struct { node_type: NodeType = .leaf, children: u16 = 0, leaves: u16 = 1 };
pub const RegistryEntry = struct { type_name: []const u8, node_type: NodeType };

pub fn unzip2(comptime T: type, comptime U: type, input: []const struct { T, U }, out_a: []T, out_b: []U) Error!void {
    if (out_a.len < input.len or out_b.len < input.len) return Error.OutOfCapacity;
    for (input, 0..) |v, i| {
        out_a[i] = v[0];
        out_b[i] = v[1];
    }
}
pub fn unzip3(comptime A: type, comptime B: type, comptime C: type, input: []const struct { A, B, C }, out_a: []A, out_b: []B, out_c: []C) Error!void {
    if (out_a.len < input.len or out_b.len < input.len or out_c.len < input.len) return Error.OutOfCapacity;
    for (input, 0..) |v, i| {
        out_a[i] = v[0];
        out_b[i] = v[1];
        out_c[i] = v[2];
    }
}
pub fn get_fully_qualified_class_name(comptime T: type) []const u8 {
    return @typeName(T);
}
pub fn is_frozen_dataclass(_: anytype) bool {
    return false;
}
pub fn is_namedtuple_instance(_: anytype) bool {
    return false;
}
pub fn is_constexpr_field(_: []const u8) bool {
    return false;
}
pub fn extract_dataclass_members(_: anytype) []const u8 {
    return "";
}
pub fn default_dataclass_to_iterable(_: anytype) PyTreeDef {
    return .{};
}
pub fn set_dataclass_attributes(value: anytype, _: anytype) @TypeOf(value) {
    return value;
}
pub fn default_dataclass_from_iterable(comptime T: type, _: anytype) T {
    return std.mem.zeroes(T);
}
pub fn namedtuple_to_iterable(_: anytype) PyTreeDef {
    return .{};
}
pub fn namedtuple_from_iterable(comptime T: type, _: anytype) T {
    return std.mem.zeroes(T);
}
pub fn dynamic_expression_to_iterable(_: anytype) PyTreeDef {
    return .{};
}
pub fn dynamic_expression_from_iterable(comptime T: type, _: anytype) T {
    return std.mem.zeroes(T);
}
pub fn default_dict_to_iterable(_: anytype) PyTreeDef {
    return .{ .node_type = .struct_ };
}
pub fn default_dict_from_iterable(comptime T: type, _: anytype) T {
    return std.mem.zeroes(T);
}
pub fn register_pytree_node(_: []const u8, _: NodeType) void {}
pub fn register_default_node_types() void {}
pub fn get_registered_node_types_or_insert(type_name: []const u8, node_type: NodeType) RegistryEntry {
    return .{ .type_name = type_name, .node_type = node_type };
}
pub fn create_leaf_for_value(index: usize) Leaf {
    return .{ .index = index };
}

pub fn tree_flatten(tree: *const layout.Tree, leaves_out: *layout.Flat) Error!PyTreeDef {
    leaves_out.* = try tree.flattenLeaves();
    return .{ .node_type = if (tree.rank() == 1) .leaf else .tuple, .children = @intCast(tree.rank()), .leaves = @intCast(leaves_out.len) };
}

test "tree_utils: flatten layout tree into leaves and definition" {
    const t = layout.Tree.fromComptime(.{ 2, .{ 3, 4 } });
    var leaves: layout.Flat = .{};
    const def = try tree_flatten(&t, &leaves);
    try std.testing.expectEqual(@as(usize, 3), leaves.len);
    try std.testing.expectEqual(NodeType.tuple, def.node_type);
}
