const std = @import("std");
const layout = @import("layout.zig");
const layout_algebra = @import("layout_algebra.zig");
const typing = @import("typing.zig");
const mlir = @import("mlir_text.zig");
const runtime = @import("runtime.zig");

pub const Error = layout.Error || mlir.Error || runtime.Error || error{
    InvalidTensorEngine,
    InvalidTensorShape,
    InvalidTensorOperation,
    BroadcastMismatch,
    UnsupportedDynamicTensor,
    TooManyTensorElements,
};

pub const EngineKind = enum { pointer, mlir_value, identity, rmem, fragment, tensor_ssa };

pub const Engine = union(EngineKind) {
    pointer: runtime.Pointer,
    mlir_value: mlir.Value,
    identity: void,
    rmem: void,
    fragment: void,
    tensor_ssa: mlir.Value,
};

pub const Tensor = struct {
    engine: Engine,
    layout_value: layout.Layout,
    dtype: typing.Numeric,
    memspace: typing.AddressSpace = .generic,

    pub fn init(engine: Engine, layout_value: layout.Layout, dtype: typing.Numeric, memspace: typing.AddressSpace) Tensor {
        return .{ .engine = engine, .layout_value = layout_value, .dtype = dtype, .memspace = memspace };
    }

    pub fn shape(self: Tensor) layout.Tree {
        return self.layout_value.shape;
    }

    pub fn stride(self: Tensor) layout.Tree {
        return self.layout_value.stride;
    }

    pub fn rank(self: *const Tensor) usize {
        return self.layout_value.rank();
    }

    pub fn size(self: *const Tensor) Error!layout.Unsigned {
        return self.layout_value.size();
    }

    pub fn elementType(self: Tensor) typing.Numeric {
        return self.dtype;
    }

    pub fn leadingDim(self: Tensor) Error!?usize {
        const strides = try self.layout_value.stride.flattenLeaves();
        var found: ?usize = null;
        for (strides.slice(), 0..) |s, i| {
            if (s == 1) {
                if (found != null) return null;
                found = i;
            }
        }
        return found;
    }

    pub fn coordToIndex(self: Tensor, coord: layout.Tree) Error!layout.Scalar {
        return self.layout_value.crd2idx(coord);
    }

    pub fn domainOffset(self: Tensor, coord: layout.Tree) Error!Tensor {
        const offset = try self.coordToIndex(coord);
        var out = self;
        switch (out.engine) {
            .pointer => |p| out.engine = .{ .pointer = p.add(@intCast(offset)) },
            else => {},
        }
        return out;
    }

    pub fn recast(self: Tensor, new_dtype: typing.Numeric) Error!Tensor {
        const old_bytes = self.dtype.bytes();
        const new_bytes = new_dtype.bytes();
        if (old_bytes == 0 or new_bytes == 0) return Error.InvalidTensorOperation;
        var out = self;
        out.dtype = new_dtype;
        if (out.engine == .pointer) {
            var p = out.engine.pointer;
            p.dtype = new_dtype;
            p.assumed_align = @min(p.assumed_align, new_bytes);
            out.engine = .{ .pointer = p };
        }
        return out;
    }

    pub fn makeTypedTensor(self: Tensor) Error!typing.TypedTensor {
        return typing.TypedTensor.init(self.dtype, self.layout_value.shape, self.layout_value.stride, self.memspace, self.dtype.bytes());
    }

    pub fn emitMakeTensor(self: Tensor, builder: anytype, engine_value: mlir.Operand, engine_type: mlir.Type) Error!mlir.Value {
        var type_buf: mlir.TextBuffer(512) = .{};
        const tt = try self.makeTypedTensor();
        try tt.writeMlirType(&type_buf);
        return builder.genericOp(
            "cute.make_tensor",
            &.{engine_value},
            &.{},
            &.{engine_type},
            &.{mlir.Type.raw(type_buf.slice())},
        );
    }

    pub fn emitLoad(self: Tensor, builder: anytype, tensor_value: mlir.Operand, tensor_type: mlir.Type, coord: mlir.Operand, coord_type: mlir.Type) Error!mlir.Value {
        return builder.genericOp("cute.tensor_load", &.{ tensor_value, coord }, &.{}, &.{ tensor_type, coord_type }, &.{mlir.Type.raw(self.dtype.mlir_type)});
    }

    pub fn emitStore(self: Tensor, builder: anytype, tensor_value: mlir.Operand, tensor_type: mlir.Type, coord: mlir.Operand, coord_type: mlir.Type, value: mlir.Operand) Error!void {
        try builder.operationNoResult(.{
            .name = "cute.tensor_store",
            .operands = &.{ tensor_value, coord, value },
            .operand_types = &.{ tensor_type, coord_type, mlir.Type.raw(self.dtype.mlir_type) },
            .result_types = &.{},
        });
    }

    pub fn emitFill(self: Tensor, builder: anytype, tensor_value: mlir.Operand, tensor_type: mlir.Type, value: mlir.Operand) Error!void {
        try builder.operationNoResult(.{
            .name = "cute.tensor_fill",
            .operands = &.{ tensor_value, value },
            .operand_types = &.{ tensor_type, mlir.Type.raw(self.dtype.mlir_type) },
            .result_types = &.{},
        });
    }
};

pub const TensorSsa = struct {
    value: mlir.Value,
    shape_value: layout.Tree,
    dtype: typing.Numeric,

    pub fn init(value: mlir.Value, shape_value: layout.Tree, dtype: typing.Numeric) Error!TensorSsa {
        try shape_value.assertPositive();
        return .{ .value = value, .shape_value = shape_value, .dtype = dtype };
    }

    pub fn vectorType(self: TensorSsa, out: anytype) Error!void {
        try out.append("vector<");
        try out.appendUnsigned(@intCast(try self.shape_value.product()));
        try out.append("x");
        try out.append(self.dtype.mlir_type);
        try out.append(">");
    }

    pub fn toVector(self: TensorSsa) mlir.Value {
        return self.value;
    }

    pub fn broadcastTo(self: TensorSsa, builder: anytype, result_shape: layout.Tree) Error!TensorSsa {
        const result_elems = try result_shape.product();
        const src_elems = try self.shape_value.product();
        if (src_elems != 1 and src_elems != result_elems) return Error.BroadcastMismatch;
        var type_buf: mlir.TextBuffer(128) = .{};
        try type_buf.append("vector<");
        try type_buf.appendUnsigned(@intCast(result_elems));
        try type_buf.append("x");
        try type_buf.append(self.dtype.mlir_type);
        try type_buf.append(">");
        const result = try mlir.vector.broadcast(builder, .{ .value = self.value }, mlir.Type.raw(self.dtype.mlir_type), mlir.Type.raw(type_buf.slice()));
        return TensorSsa.init(result, result_shape, self.dtype);
    }

    pub fn applyBinary(self: TensorSsa, builder: anytype, op: BinaryOp, rhs: TensorSsa) Error!TensorSsa {
        const res_shape = try inferBroadcastShape(self.shape_value, rhs.shape_value);
        const lhs_b = if (self.shape_value.equals(&res_shape)) self else try self.broadcastTo(builder, res_shape);
        const rhs_b = if (rhs.shape_value.equals(&res_shape)) rhs else try rhs.broadcastTo(builder, res_shape);
        var vt: mlir.TextBuffer(128) = .{};
        try lhs_b.vectorType(&vt);
        const ty = mlir.Type.raw(vt.slice());
        const name = op.mlirName(lhs_b.dtype);
        const value = try builder.genericOp(name, &.{ .{ .value = lhs_b.value }, .{ .value = rhs_b.value } }, op.attrs(), &.{ ty, ty }, &.{ty});
        return TensorSsa.init(value, res_shape, op.resultDtype(lhs_b.dtype, rhs_b.dtype));
    }

    pub fn applyUnary(self: TensorSsa, builder: anytype, op: UnaryOp) Error!TensorSsa {
        var vt: mlir.TextBuffer(128) = .{};
        try self.vectorType(&vt);
        const ty = mlir.Type.raw(vt.slice());
        const value = try builder.genericOp(op.mlirName(self.dtype), &.{.{ .value = self.value }}, &.{}, &.{ty}, &.{ty});
        return TensorSsa.init(value, self.shape_value, self.dtype);
    }

    pub fn reshape(self: TensorSsa, builder: anytype, new_shape: layout.Tree) Error!TensorSsa {
        if (try self.shape_value.product() != try new_shape.product()) return Error.InvalidTensorShape;
        var old_type: mlir.TextBuffer(128) = .{};
        var new_type: mlir.TextBuffer(128) = .{};
        try self.vectorType(&old_type);
        try new_type.append("vector<");
        try new_type.appendUnsigned(@intCast(try new_shape.product()));
        try new_type.append("x");
        try new_type.append(self.dtype.mlir_type);
        try new_type.append(">");
        const v = try builder.genericOp("vector.shape_cast", &.{.{ .value = self.value }}, &.{}, &.{mlir.Type.raw(old_type.slice())}, &.{mlir.Type.raw(new_type.slice())});
        return TensorSsa.init(v, new_shape, self.dtype);
    }

    pub fn reduce(self: TensorSsa, builder: anytype, op: ReduceOp) Error!mlir.Value {
        var vt: mlir.TextBuffer(128) = .{};
        try self.vectorType(&vt);
        return builder.genericOp(op.mlirName(), &.{.{ .value = self.value }}, &.{}, &.{mlir.Type.raw(vt.slice())}, &.{mlir.Type.raw(self.dtype.mlir_type)});
    }
};

pub const BinaryOp = enum {
    add,
    sub,
    mul,
    div,
    rem,
    min,
    max,
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    bit_and,
    bit_or,
    bit_xor,

    pub fn mlirName(self: BinaryOp, dtype: typing.Numeric) []const u8 {
        return switch (self) {
            .add => if (dtype.isFloat()) "arith.addf" else "arith.addi",
            .sub => if (dtype.isFloat()) "arith.subf" else "arith.subi",
            .mul => if (dtype.isFloat()) "arith.mulf" else "arith.muli",
            .div => if (dtype.isFloat()) "arith.divf" else "arith.divsi",
            .rem => "arith.remsi",
            .min => if (dtype.isFloat()) "arith.minimumf" else "arith.minsi",
            .max => if (dtype.isFloat()) "arith.maximumf" else "arith.maxsi",
            .eq, .ne, .lt, .le, .gt, .ge => if (dtype.isFloat()) "arith.cmpf" else "arith.cmpi",
            .bit_and => "arith.andi",
            .bit_or => "arith.ori",
            .bit_xor => "arith.xori",
        };
    }

    pub fn attrs(self: BinaryOp) []const mlir.Attribute {
        return switch (self) {
            .eq => &.{.{ .key = "predicate", .value = "eq" }},
            .ne => &.{.{ .key = "predicate", .value = "ne" }},
            .lt => &.{.{ .key = "predicate", .value = "slt" }},
            .le => &.{.{ .key = "predicate", .value = "sle" }},
            .gt => &.{.{ .key = "predicate", .value = "sgt" }},
            .ge => &.{.{ .key = "predicate", .value = "sge" }},
            else => &.{},
        };
    }

    pub fn resultDtype(self: BinaryOp, lhs: typing.Numeric, _: typing.Numeric) typing.Numeric {
        return switch (self) {
            .eq, .ne, .lt, .le, .gt, .ge => typing.Boolean,
            else => lhs,
        };
    }
};

pub const UnaryOp = enum {
    neg,
    abs,

    pub fn mlirName(self: UnaryOp, dtype: typing.Numeric) []const u8 {
        return switch (self) {
            .neg => if (dtype.isFloat()) "arith.negf" else "cute.neg_int",
            .abs => if (dtype.isFloat()) "math.absf" else "cute.abs_int",
        };
    }
};

pub const ReduceOp = enum {
    add,
    mul,
    min,
    max,
    any,
    all,

    pub fn mlirName(self: ReduceOp) []const u8 {
        return switch (self) {
            .add => "vector.reduction.add",
            .mul => "vector.reduction.mul",
            .min => "vector.reduction.min",
            .max => "vector.reduction.max",
            .any => "vector.reduction.or",
            .all => "vector.reduction.and",
        };
    }
};

pub fn makeTensor(engine: Engine, layout_value: layout.Layout, dtype: typing.Numeric, memspace: typing.AddressSpace) Tensor {
    return Tensor.init(engine, layout_value, dtype, memspace);
}

pub fn makeIdentityTensor(shape: layout.Tree) Error!Tensor {
    const l = try layout.Layout.makeCompact(shape);
    return Tensor.init(.{ .identity = {} }, l, typing.Int32, .generic);
}

pub fn makeRmemTensor(dtype: typing.Numeric, shape: layout.Tree) Error!Tensor {
    const l = try layout.Layout.makeCompact(shape);
    return Tensor.init(.{ .rmem = {} }, l, dtype, .generic);
}

pub fn makeFragment(dtype: typing.Numeric, shape: layout.Tree) Error!Tensor {
    const l = try layout.Layout.makeCompact(shape);
    return Tensor.init(.{ .fragment = {} }, l, dtype, .generic);
}

pub fn makeFragmentLike(source: Tensor) Error!Tensor {
    return makeFragment(source.dtype, source.layout_value.shape);
}

pub fn makeRmemTensorLike(source: Tensor) Error!Tensor {
    return makeRmemTensor(source.dtype, source.layout_value.shape);
}

pub fn recastTensor(source: Tensor, dtype: typing.Numeric) Error!Tensor {
    return source.recast(dtype);
}

pub fn domainOffset(source: Tensor, coord: layout.Tree) Error!Tensor {
    return source.domainOffset(coord);
}

pub fn full(builder: anytype, shape: layout.Tree, dtype: typing.Numeric, value: mlir.Operand) Error!TensorSsa {
    const elems = try shape.product();
    if (elems > 64) return Error.TooManyTensorElements;
    var type_buf: mlir.TextBuffer(128) = .{};
    try type_buf.append("vector<");
    try type_buf.appendUnsigned(@intCast(elems));
    try type_buf.append("x");
    try type_buf.append(dtype.mlir_type);
    try type_buf.append(">");
    const v = try mlir.vector.broadcast(builder, value, mlir.Type.raw(dtype.mlir_type), mlir.Type.raw(type_buf.slice()));
    return TensorSsa.init(v, shape, dtype);
}

pub fn zerosLike(builder: anytype, source: TensorSsa) Error!TensorSsa {
    const zero = try builder.constantI(0, mlir.Type.raw(source.dtype.mlir_type));
    return full(builder, source.shape_value, source.dtype, .{ .value = zero });
}

pub fn onesLike(builder: anytype, source: TensorSsa) Error!TensorSsa {
    const one = try builder.constantI(1, mlir.Type.raw(source.dtype.mlir_type));
    return full(builder, source.shape_value, source.dtype, .{ .value = one });
}

pub fn where(builder: anytype, cond: TensorSsa, if_value: TensorSsa, else_value: TensorSsa) Error!TensorSsa {
    const shape = try inferBroadcastShape(if_value.shape_value, else_value.shape_value);
    const a = if (if_value.shape_value.equals(&shape)) if_value else try if_value.broadcastTo(builder, shape);
    const b = if (else_value.shape_value.equals(&shape)) else_value else try else_value.broadcastTo(builder, shape);
    const c = if (cond.shape_value.equals(&shape)) cond else try cond.broadcastTo(builder, shape);
    var vt: mlir.TextBuffer(128) = .{};
    try a.vectorType(&vt);
    const v = try builder.genericOp("arith.select", &.{ .{ .value = c.value }, .{ .value = a.value }, .{ .value = b.value } }, &.{}, &.{ mlir.Type.raw("vector<?xi1>"), mlir.Type.raw(vt.slice()), mlir.Type.raw(vt.slice()) }, &.{mlir.Type.raw(vt.slice())});
    return TensorSsa.init(v, shape, a.dtype);
}

pub fn gather(builder: anytype, source: TensorSsa, indices: TensorSsa) Error!TensorSsa {
    var src_ty: mlir.TextBuffer(128) = .{};
    var idx_ty: mlir.TextBuffer(128) = .{};
    try source.vectorType(&src_ty);
    try indices.vectorType(&idx_ty);
    const v = try builder.genericOp("cute.gather", &.{ .{ .value = source.value }, .{ .value = indices.value } }, &.{}, &.{ mlir.Type.raw(src_ty.slice()), mlir.Type.raw(idx_ty.slice()) }, &.{mlir.Type.raw(src_ty.slice())});
    return TensorSsa.init(v, source.shape_value, source.dtype);
}

pub fn scatter(builder: anytype, source: TensorSsa, indices: TensorSsa, values: TensorSsa) Error!TensorSsa {
    var src_ty: mlir.TextBuffer(128) = .{};
    var idx_ty: mlir.TextBuffer(128) = .{};
    var val_ty: mlir.TextBuffer(128) = .{};
    try source.vectorType(&src_ty);
    try indices.vectorType(&idx_ty);
    try values.vectorType(&val_ty);
    const v = try builder.genericOp("cute.scatter", &.{ .{ .value = source.value }, .{ .value = indices.value }, .{ .value = values.value } }, &.{}, &.{ mlir.Type.raw(src_ty.slice()), mlir.Type.raw(idx_ty.slice()), mlir.Type.raw(val_ty.slice()) }, &.{mlir.Type.raw(src_ty.slice())});
    return TensorSsa.init(v, source.shape_value, source.dtype);
}

fn inferBroadcastShape(lhs: layout.Tree, rhs: layout.Tree) Error!layout.Tree {
    if (lhs.equals(&rhs)) return lhs;
    const lprod = try lhs.product();
    const rprod = try rhs.product();
    if (lprod == 1) return rhs;
    if (rprod == 1) return lhs;
    return Error.BroadcastMismatch;
}

test "tensor: make tensor and domain offset" {
    const shape = layout.Tree.fromComptime(.{ 4, 4 });
    const l = try layout.Layout.makeCompact(shape);
    const ptr = try runtime.Pointer.init(0x1000, typing.Float32, .gmem, null);
    const t = makeTensor(.{ .pointer = ptr }, l, typing.Float32, .gmem);
    const shifted = try t.domainOffset(layout.Tree.fromComptime(.{ 2, 1 }));
    try std.testing.expectEqual(@as(usize, 0x1000 + 6 * 4), shifted.engine.pointer.address);
}

test "tensor: TensorSSA emits binary and reduction operations" {
    var b: mlir.Builder(2048) = .{};
    const shape = layout.Tree.fromComptime(.{4});
    const c0 = try b.constantI(0, mlir.Type.i(32));
    const lhs = try full(&b, shape, typing.Int32, .{ .value = c0 });
    const rhs = try onesLike(&b, lhs);
    const sum = try lhs.applyBinary(&b, .add, rhs);
    _ = try sum.reduce(&b, .add);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "arith.addi") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "vector.reduction.add") != null);
}

test "tensor: reshape preserves element count" {
    var b: mlir.Builder(1024) = .{};
    const c0 = try b.constantI(0, mlir.Type.i(32));
    const src = try full(&b, layout.Tree.fromComptime(.{ 2, 3 }), typing.Int32, .{ .value = c0 });
    const reshaped = try src.reshape(&b, layout.Tree.fromComptime(.{6}));
    try std.testing.expectEqual(@as(layout.Unsigned, 6), try reshaped.shape_value.product());
    try std.testing.expectError(Error.InvalidTensorShape, src.reshape(&b, layout.Tree.fromComptime(.{5})));
}
