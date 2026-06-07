const std = @import("std");
const layout = @import("layout.zig");
const layout_algebra = @import("layout_algebra.zig");
const layout_core = @import("layout_core.zig");
const typing = @import("typing.zig");
const mlir = @import("mlir_text.zig");
const runtime = @import("runtime.zig");
const tensor = @import("tensor.zig");
const cutlass_emit = @import("cutlass_emit.zig");

pub const Scalar = layout.Scalar;
pub const Unsigned = layout.Unsigned;
pub const Tree = layout.Tree;
pub const Layout = layout.Layout;
pub const Selector = layout_algebra.Selector;
pub const keep = layout_algebra.keep;

const attr_cmpi_eq = [_]mlir.Attribute{.{ .key = "predicate", .value = "eq" }};
const attr_cmpi_ne = [_]mlir.Attribute{.{ .key = "predicate", .value = "ne" }};
const attr_cmpi_lt = [_]mlir.Attribute{.{ .key = "predicate", .value = "slt" }};
const attr_cmpi_le = [_]mlir.Attribute{.{ .key = "predicate", .value = "sle" }};
const attr_cmpi_gt = [_]mlir.Attribute{.{ .key = "predicate", .value = "sgt" }};
const attr_cmpi_ge = [_]mlir.Attribute{.{ .key = "predicate", .value = "sge" }};
const attr_cmpui_lt = [_]mlir.Attribute{.{ .key = "predicate", .value = "ult" }};
const attr_cmpui_le = [_]mlir.Attribute{.{ .key = "predicate", .value = "ule" }};
const attr_cmpui_gt = [_]mlir.Attribute{.{ .key = "predicate", .value = "ugt" }};
const attr_cmpui_ge = [_]mlir.Attribute{.{ .key = "predicate", .value = "uge" }};
const attr_cmpf_eq = [_]mlir.Attribute{.{ .key = "predicate", .value = "oeq" }};
const attr_cmpf_ne = [_]mlir.Attribute{.{ .key = "predicate", .value = "one" }};
const attr_cmpf_lt = [_]mlir.Attribute{.{ .key = "predicate", .value = "olt" }};
const attr_cmpf_le = [_]mlir.Attribute{.{ .key = "predicate", .value = "ole" }};
const attr_cmpf_gt = [_]mlir.Attribute{.{ .key = "predicate", .value = "ogt" }};
const attr_cmpf_ge = [_]mlir.Attribute{.{ .key = "predicate", .value = "oge" }};
const attr_none = [_]mlir.Attribute{};

pub const VectorOrder = enum {
    /// Cute TensorSSA internal order: logical column-major flattened vector.
    column_major,
    /// MLIR n-D vector operation order: row-major shaped vector.
    row_major,
};

pub const Error = tensor.Error || layout_core.Error || cutlass_emit.Error || error{
    EmptyTensorSsa,
    InvalidMaskShape,
    InvalidReductionProfile,
    InvalidElementType,
    InvalidTensorAccess,
    InvalidGatherScatterMode,
    IncompatibleTensorShapes,
    NarrowPrecisionAlignment,
    InvalidVectorOrder,
    InvalidVectorSlice,
    InvalidVectorRank,
};

/// Concrete tensor metadata used by Zig-native front-end code.
///
/// This deliberately separates semantic tensor metadata from any particular
/// MLIR SSA handle so the same value can be used at comptime for validation,
/// runtime for launch descriptors, and during textual MLIR generation.
pub const TensorMeta = struct {
    engine: tensor.Engine,
    layout_value: Layout,
    dtype: typing.Numeric,
    memspace: typing.AddressSpace = .generic,

    pub fn init(engine: tensor.Engine, layout_value: Layout, dtype: typing.Numeric, memspace: typing.AddressSpace) Error!TensorMeta {
        try layout_value.shape.assertPositive();
        if (!layout_value.shape.sameProfile(&layout_value.stride)) return Error.ProfileMismatch;
        return .{ .engine = engine, .layout_value = layout_value, .dtype = dtype, .memspace = memspace };
    }

    pub fn fromTensor(source: tensor.Tensor) TensorMeta {
        return .{ .engine = source.engine, .layout_value = source.layout_value, .dtype = source.dtype, .memspace = source.memspace };
    }

    pub fn toTensor(self: TensorMeta) tensor.Tensor {
        return tensor.Tensor.init(self.engine, self.layout_value, self.dtype, self.memspace);
    }

    pub fn shape(self: TensorMeta) Tree {
        return self.layout_value.shape;
    }

    pub fn stride(self: TensorMeta) Tree {
        return self.layout_value.stride;
    }

    pub fn rank(self: TensorMeta) usize {
        return self.layout_value.rank();
    }

    pub fn size(self: TensorMeta) Error!Unsigned {
        return self.layout_value.size();
    }

    pub fn cosize(self: TensorMeta) Error!Unsigned {
        return self.layout_value.cosize();
    }

    pub fn elementType(self: TensorMeta) typing.Numeric {
        return self.dtype;
    }

    pub fn typedTensor(self: TensorMeta) Error!typing.TypedTensor {
        return typing.TypedTensor.init(self.dtype, self.layout_value.shape, self.layout_value.stride, self.memspace, self.dtype.bytes());
    }

    pub fn tensorTypeText(self: TensorMeta, out: anytype) Error!void {
        try self.cutlassTensorTypeText(out);
    }

    pub fn cutlassTensorTypeText(self: TensorMeta, out: anytype) Error!void {
        try cutlass_emit.writeMemRefTypeForLayout(out, self.dtype, self.memspace, self.dtype.bytes(), &self.layout_value);
    }

    /// Byte/element reinterpretation of the tensor payload.  Static layout
    /// scaling follows CuteDSL's recast_tensor rule: the total bit capacity is
    /// preserved and the logical element count changes by old_width/new_width.
    /// Nested profiles are conservatively flattened when element count changes.
    pub fn recast(self: TensorMeta, new_dtype: typing.Numeric) Error!TensorMeta {
        if (new_dtype.width == 0 or self.dtype.width == 0) return Error.InvalidElementType;
        var out = self;
        out.dtype = new_dtype;
        if (self.dtype.width != new_dtype.width) {
            const old_bits = (try self.layout_value.size()) * self.dtype.width;
            if (old_bits % new_dtype.width != 0) return Error.InvalidTensorShape;
            const new_elems: Scalar = @intCast(old_bits / new_dtype.width);
            out.layout_value = try Layout.makeCompact(try Tree.initLeaf(new_elems));
        }
        switch (out.engine) {
            .pointer => |p| {
                var p2 = p;
                p2.dtype = new_dtype;
                p2.assumed_align = @min(p2.assumed_align, new_dtype.bytes());
                out.engine = .{ .pointer = p2 };
            },
            else => {},
        }
        return out;
    }

    /// Apply a coordinate/slice selector.  If the selector fixes every mode,
    /// the result is an element access with the physical offset.  If at least
    /// one mode is kept, the result is a sub-tensor whose engine has been
    /// domain-offset when the engine is a pointer.
    pub fn access(self: TensorMeta, selector: *const Selector) Error!TensorAccess {
        try validateSelectorAgainstLayout(&self.layout_value, selector);
        const offset = try selectorOffset(&self.layout_value, selector);
        if ((try selectorStats(selector)).kept == 0) {
            return .{ .element = .{ .source = self, .offset = offset } };
        }
        const sliced = try layout_algebra.sliceAndOffset(&self.layout_value, selector);
        var out = self;
        out.layout_value = sliced.layout;
        switch (out.engine) {
            .pointer => |p| out.engine = .{ .pointer = p.add(@intCast(sliced.offset)) },
            else => {},
        }
        return .{ .tensor = out };
    }

    pub fn slice(self: TensorMeta, selector: *const Selector) Error!TensorMeta {
        return switch (try self.access(selector)) {
            .tensor => |t| t,
            .element => Error.InvalidTensorAccess,
        };
    }

    pub fn elementOffset(self: TensorMeta, coord: Tree) Error!Scalar {
        return self.layout_value.crd2idx(coord);
    }
};

pub const ElementAccess = struct {
    source: TensorMeta,
    offset: Scalar,
};

pub const TensorAccess = union(enum) {
    element: ElementAccess,
    tensor: TensorMeta,
};

/// SSA handle paired with tensor metadata.  `value` is the MLIR value for the
/// tensor/memref itself; element/vector operations are emitted through the
/// textual MLIR builder.
pub const TensorValue = struct {
    meta: TensorMeta,
    value: mlir.Value,
    /// Optional borrowed static type text. Prefer initFromMeta for values returned
    /// from generated ops so we do not smuggle fake placeholder types like
    /// `!cute.tensor` back into later emitters.
    type_text: []const u8 = "",
    owned_type_text: [512]u8 = undefined,
    owned_type_len: usize = 0,

    pub fn init(meta: TensorMeta, value: mlir.Value, type_text: []const u8) TensorValue {
        return .{ .meta = meta, .value = value, .type_text = type_text, .owned_type_len = 0 };
    }

    pub fn initFromMeta(meta: TensorMeta, value: mlir.Value) Error!TensorValue {
        var out = TensorValue.init(meta, value, "");
        var type_buf: mlir.TextBuffer(512) = .{};
        try meta.tensorTypeText(&type_buf);
        @memcpy(out.owned_type_text[0..type_buf.len], type_buf.slice());
        out.owned_type_len = type_buf.len;
        return out;
    }

    pub fn typeText(self: *const TensorValue) []const u8 {
        return if (self.owned_type_len != 0) self.owned_type_text[0..self.owned_type_len] else self.type_text;
    }

    pub fn type_(self: *const TensorValue) mlir.Type {
        return mlir.Type.raw(self.typeText());
    }

    pub fn access(self: TensorValue, builder: anytype, selector: *const Selector) Error!AccessValue {
        return switch (try self.meta.access(selector)) {
            .tensor => |sub| blk: {
                const offset_val = try builder.constantIndex(try selectorOffset(&self.meta.layout_value, selector));
                var sub_type_buf: mlir.TextBuffer(512) = .{};
                try sub.tensorTypeText(&sub_type_buf);
                const result = try builder.genericOp(
                    "cute.tensor_slice",
                    &.{ .{ .value = self.value }, .{ .value = offset_val } },
                    &.{},
                    &.{ self.type_(), mlir.Type.index() },
                    &.{mlir.Type.raw(sub_type_buf.slice())},
                );
                break :blk .{ .tensor = try TensorValue.initFromMeta(sub, result) };
            },
            .element => |elt| blk: {
                var memref_ty_buf: mlir.TextBuffer(512) = .{};
                try self.meta.cutlassTensorTypeText(&memref_ty_buf);
                const coord = try cutlass_emit.makeCoordFromScalar(builder, elt.offset);
                const result = try cutlass_emit.emitMemrefLoad(
                    builder,
                    self.value,
                    coord.value,
                    mlir.Type.raw(memref_ty_buf.slice()),
                    coord.ty,
                    mlir.Type.raw(self.meta.dtype.mlir_type),
                );
                break :blk .{ .element = result };
            },
        };
    }

    pub fn storeElement(self: TensorValue, builder: anytype, selector: *const Selector, data: SsaValue) Error!void {
        const access_result = try self.meta.access(selector);
        switch (access_result) {
            .element => |elt| {
                const casted = try data.castTo(builder, self.meta.dtype);
                var memref_ty_buf: mlir.TextBuffer(512) = .{};
                try self.meta.cutlassTensorTypeText(&memref_ty_buf);
                const coord = try cutlass_emit.makeCoordFromScalar(builder, elt.offset);
                try cutlass_emit.emitMemrefStore(
                    builder,
                    self.value,
                    coord.value,
                    casted.value,
                    mlir.Type.raw(memref_ty_buf.slice()),
                    coord.ty,
                    mlir.Type.raw(self.meta.dtype.mlir_type),
                );
            },
            .tensor => return Error.InvalidTensorAccess,
        }
    }

    pub fn load(self: TensorValue, builder: anytype, mask: ?SsaTensor, pass_thru: ?SsaTensor) Error!SsaTensor {
        try checkVectorLoadStore(self.meta);
        if (mask) |m| try validateMaskShape(self.meta.layout_value.shape, m.shape_value);
        if (pass_thru) |p| try expectSameShape(self.meta.layout_value.shape, p.shape_value);

        const memory_dtype = memoryNumeric(self.meta.dtype);
        var memory_vec_ty: mlir.TextBuffer(128) = .{};
        try writeVectorType(&memory_vec_ty, self.meta.layout_value.shape, memory_dtype);
        var memref_ty_buf: mlir.TextBuffer(512) = .{};
        try self.meta.cutlassTensorTypeText(&memref_ty_buf);
        const raw_value = try cutlass_emit.emitMemrefLoadVec(builder, self.value, mlir.Type.raw(memref_ty_buf.slice()), mlir.Type.raw(memory_vec_ty.slice()));
        if (self.meta.dtype.kind == .boolean) {
            const raw = try SsaTensor.init(raw_value, self.meta.layout_value.shape, typing.Int8);
            const zero = try SsaTensor.zeros(builder, self.meta.layout_value.shape, typing.Int8);
            return raw.binary(builder, .ne, zero);
        }
        return SsaTensor.init(raw_value, self.meta.layout_value.shape, self.meta.dtype);
    }

    pub fn store(self: TensorValue, builder: anytype, data: SsaTensor, mask: ?SsaTensor) Error!void {
        try checkVectorLoadStore(self.meta);
        try expectSameShape(self.meta.layout_value.shape, data.shape_value);
        if (mask) |m| try validateMaskShape(self.meta.layout_value.shape, m.shape_value);
        const memory_dtype = memoryNumeric(self.meta.dtype);
        try checkNarrowStoreAlignment(memory_dtype, self.meta.layout_value.shape);
        const casted = try data.castTo(builder, memory_dtype);
        var vec_ty_buf: mlir.TextBuffer(128) = .{};
        try writeVectorType(&vec_ty_buf, self.meta.layout_value.shape, memory_dtype);
        var memref_ty_buf: mlir.TextBuffer(512) = .{};
        try self.meta.cutlassTensorTypeText(&memref_ty_buf);
        try cutlass_emit.emitMemrefStoreVec(builder, casted.value, self.value, mlir.Type.raw(vec_ty_buf.slice()), mlir.Type.raw(memref_ty_buf.slice()));
    }

    pub fn fill(self: TensorValue, builder: anytype, scalar: SsaValue) Error!void {
        const filled = try SsaTensor.full(builder, self.meta.layout_value.shape, self.meta.dtype, .{ .value = scalar.value });
        try self.store(builder, filled, null);
    }
};

pub const AccessValue = union(enum) {
    element: mlir.Value,
    tensor: TensorValue,
};

pub const SsaValue = struct {
    value: mlir.Value,
    dtype: typing.Numeric,

    pub fn init(value: mlir.Value, dtype: typing.Numeric) SsaValue {
        return .{ .value = value, .dtype = dtype };
    }

    pub fn castTo(self: SsaValue, builder: anytype, target: typing.Numeric) Error!SsaValue {
        if (sameNumeric(self.dtype, target)) return self;
        const op_name = conversionOp(self.dtype, target);
        const result = try builder.genericOp(
            op_name,
            &.{.{ .value = self.value }},
            &.{},
            &.{mlir.Type.raw(self.dtype.mlir_type)},
            &.{mlir.Type.raw(target.mlir_type)},
        );
        return .{ .value = result, .dtype = target };
    }

    pub fn irValueInt8(self: SsaValue, builder: anytype) Error!SsaValue {
        if (self.dtype.kind != .boolean) return self.castTo(builder, typing.Int8);
        return self.castTo(builder, typing.Int8);
    }
};

pub const SsaTensor = struct {
    value: mlir.Value,
    shape_value: Tree,
    dtype: typing.Numeric,

    pub fn init(value: mlir.Value, shape_value: Tree, dtype: typing.Numeric) Error!SsaTensor {
        try shape_value.assertPositive();
        return .{ .value = value, .shape_value = shape_value, .dtype = dtype };
    }

    pub fn vectorType(self: SsaTensor, out: anytype) Error!void {
        try writeVectorType(out, self.shape_value, self.dtype);
    }

    pub fn scalar(self: SsaTensor) Error!SsaValue {
        if (try self.shape_value.product() != 1) return Error.InvalidTensorShape;
        return .{ .value = self.value, .dtype = self.dtype };
    }

    pub fn full(builder: anytype, shape_value: Tree, dtype: typing.Numeric, scalar_operand: mlir.Operand) Error!SsaTensor {
        var vec_ty: mlir.TextBuffer(128) = .{};
        try writeVectorType(&vec_ty, shape_value, dtype);
        const result = try mlir.vector.broadcast(builder, scalar_operand, mlir.Type.raw(dtype.mlir_type), mlir.Type.raw(vec_ty.slice()));
        return SsaTensor.init(result, shape_value, dtype);
    }

    pub fn empty(builder: anytype, shape_value: Tree, dtype: typing.Numeric) Error!SsaTensor {
        var vec_ty: mlir.TextBuffer(128) = .{};
        try writeVectorType(&vec_ty, shape_value, dtype);
        const result = try builder.genericOp("cute.undef", &.{}, &.{}, &.{}, &.{mlir.Type.raw(vec_ty.slice())});
        return SsaTensor.init(result, shape_value, dtype);
    }

    pub fn fullLike(builder: anytype, source: SsaTensor, scalar_operand: mlir.Operand, dtype: ?typing.Numeric) Error!SsaTensor {
        return SsaTensor.full(builder, source.shape_value, dtype orelse source.dtype, scalar_operand);
    }

    pub fn emptyLike(builder: anytype, source: SsaTensor, dtype: ?typing.Numeric) Error!SsaTensor {
        return SsaTensor.empty(builder, source.shape_value, dtype orelse source.dtype);
    }

    pub fn irValueInt8(self: SsaTensor, builder: anytype) Error!SsaTensor {
        return self.castTo(builder, typing.Int8);
    }

    pub fn zeros(builder: anytype, shape_value: Tree, dtype: typing.Numeric) Error!SsaTensor {
        const zero = if (dtype.isFloat()) try builder.constantF(0.0, mlir.Type.raw(dtype.mlir_type)) else try builder.constantI(0, mlir.Type.raw(dtype.mlir_type));
        return SsaTensor.full(builder, shape_value, dtype, .{ .value = zero });
    }

    pub fn ones(builder: anytype, shape_value: Tree, dtype: typing.Numeric) Error!SsaTensor {
        const one = if (dtype.isFloat()) try builder.constantF(1.0, mlir.Type.raw(dtype.mlir_type)) else try builder.constantI(1, mlir.Type.raw(dtype.mlir_type));
        return SsaTensor.full(builder, shape_value, dtype, .{ .value = one });
    }

    pub fn reshape(self: SsaTensor, builder: anytype, new_shape: Tree) Error!SsaTensor {
        if (try self.shape_value.product() != try new_shape.product()) return Error.InvalidTensorShape;
        var old_ty: mlir.TextBuffer(128) = .{};
        var new_ty: mlir.TextBuffer(128) = .{};
        try self.vectorType(&old_ty);
        try writeVectorType(&new_ty, new_shape, self.dtype);
        const result = try builder.genericOp("vector.shape_cast", &.{.{ .value = self.value }}, &.{}, &.{mlir.Type.raw(old_ty.slice())}, &.{mlir.Type.raw(new_ty.slice())});
        return SsaTensor.init(result, new_shape, self.dtype);
    }

    pub fn bitcast(self: SsaTensor, builder: anytype, target: typing.Numeric) Error!SsaTensor {
        if (sameNumeric(self.dtype, target)) return self;
        const bits = (try self.shape_value.product()) * self.dtype.width;
        if (bits % target.width != 0) return Error.InvalidTensorShape;
        const new_count: Scalar = @intCast(bits / target.width);
        const new_shape = try Tree.initLeaf(new_count);
        var old_ty: mlir.TextBuffer(128) = .{};
        var new_ty: mlir.TextBuffer(128) = .{};
        try self.vectorType(&old_ty);
        try writeVectorType(&new_ty, new_shape, target);
        const result = try builder.genericOp("vector.bitcast", &.{.{ .value = self.value }}, &.{}, &.{mlir.Type.raw(old_ty.slice())}, &.{mlir.Type.raw(new_ty.slice())});
        return SsaTensor.init(result, new_shape, target);
    }

    pub fn castTo(self: SsaTensor, builder: anytype, target: typing.Numeric) Error!SsaTensor {
        if (sameNumeric(self.dtype, target)) return self;
        var old_ty: mlir.TextBuffer(128) = .{};
        var new_ty: mlir.TextBuffer(128) = .{};
        try self.vectorType(&old_ty);
        try writeVectorType(&new_ty, self.shape_value, target);
        const op_name = conversionOp(self.dtype, target);
        const result = try builder.genericOp(op_name, &.{.{ .value = self.value }}, &.{}, &.{mlir.Type.raw(old_ty.slice())}, &.{mlir.Type.raw(new_ty.slice())});
        return SsaTensor.init(result, self.shape_value, target);
    }

    pub fn binary(self: SsaTensor, builder: anytype, op: BinaryOp, rhs: SsaTensor) Error!SsaTensor {
        const result_shape = try inferBroadcastShape(self.shape_value, rhs.shape_value);
        const lhs_b = if (self.shape_value.equals(&result_shape)) self else try self.broadcastTo(builder, result_shape);
        const rhs_casted = try rhs.castTo(builder, lhs_b.dtype);
        const rhs_b = if (rhs_casted.shape_value.equals(&result_shape)) rhs_casted else try rhs_casted.broadcastTo(builder, result_shape);
        var vec_ty: mlir.TextBuffer(128) = .{};
        try lhs_b.vectorType(&vec_ty);
        const result_dtype = op.resultType(lhs_b.dtype);
        var result_ty: mlir.TextBuffer(128) = .{};
        try writeVectorType(&result_ty, result_shape, result_dtype);
        const value = try builder.genericOp(op.mlirName(lhs_b.dtype), &.{ .{ .value = lhs_b.value }, .{ .value = rhs_b.value } }, op.attrs(lhs_b.dtype), &.{ mlir.Type.raw(vec_ty.slice()), mlir.Type.raw(vec_ty.slice()) }, &.{mlir.Type.raw(result_ty.slice())});
        return SsaTensor.init(value, result_shape, result_dtype);
    }

    pub fn unary(self: SsaTensor, builder: anytype, op: UnaryOp) Error!SsaTensor {
        var vec_ty: mlir.TextBuffer(128) = .{};
        try self.vectorType(&vec_ty);
        const value = try builder.genericOp(op.mlirName(self.dtype), &.{.{ .value = self.value }}, &.{}, &.{mlir.Type.raw(vec_ty.slice())}, &.{mlir.Type.raw(vec_ty.slice())});
        return SsaTensor.init(value, self.shape_value, self.dtype);
    }

    pub fn broadcastTo(self: SsaTensor, builder: anytype, shape_value: Tree) Error!SsaTensor {
        if (self.shape_value.equals(&shape_value)) return self;
        const checked = try inferBroadcastShape(self.shape_value, shape_value);
        if (!checked.equals(&shape_value)) return Error.BroadcastMismatch;
        const src = try self.shape_value.product();
        var result_ty: mlir.TextBuffer(128) = .{};
        try writeVectorType(&result_ty, shape_value, self.dtype);
        if (src == 1) {
            const value = try mlir.vector.broadcast(builder, .{ .value = self.value }, mlir.Type.raw(self.dtype.mlir_type), mlir.Type.raw(result_ty.slice()));
            return SsaTensor.init(value, shape_value, self.dtype);
        }
        var src_ty: mlir.TextBuffer(128) = .{};
        try self.vectorType(&src_ty);
        const value = try builder.genericOp(
            "cute.tensor_broadcast",
            &.{.{ .value = self.value }},
            &.{},
            &.{mlir.Type.raw(src_ty.slice())},
            &.{mlir.Type.raw(result_ty.slice())},
        );
        return SsaTensor.init(value, shape_value, self.dtype);
    }

    pub fn where_(builder: anytype, cond: SsaTensor, if_value: SsaTensor, else_value: SsaTensor) Error!SsaTensor {
        const result_shape = try inferBroadcastShape(if_value.shape_value, else_value.shape_value);
        const c = if (cond.shape_value.equals(&result_shape)) cond else try cond.broadcastTo(builder, result_shape);
        const a = if (if_value.shape_value.equals(&result_shape)) if_value else try if_value.broadcastTo(builder, result_shape);
        var b = if (else_value.shape_value.equals(&result_shape)) else_value else try else_value.broadcastTo(builder, result_shape);
        b = try b.castTo(builder, a.dtype);
        var data_ty: mlir.TextBuffer(128) = .{};
        var pred_ty: mlir.TextBuffer(128) = .{};
        try writeVectorType(&data_ty, result_shape, a.dtype);
        try writeVectorType(&pred_ty, result_shape, typing.Boolean);
        const value = try builder.genericOp("arith.select", &.{ .{ .value = c.value }, .{ .value = a.value }, .{ .value = b.value } }, &.{}, &.{ mlir.Type.raw(pred_ty.slice()), mlir.Type.raw(data_ty.slice()), mlir.Type.raw(data_ty.slice()) }, &.{mlir.Type.raw(data_ty.slice())});
        return SsaTensor.init(value, result_shape, a.dtype);
    }

    pub fn reduceAll(self: SsaTensor, builder: anytype, op: ReduceOp) Error!SsaValue {
        var vec_ty: mlir.TextBuffer(128) = .{};
        try self.vectorType(&vec_ty);
        const value = try builder.genericOp(op.mlirName(self.dtype), &.{.{ .value = self.value }}, &.{}, &.{mlir.Type.raw(vec_ty.slice())}, &.{mlir.Type.raw(self.dtype.mlir_type)});
        return SsaValue.init(value, self.dtype);
    }

    pub fn reduceProfile(self: SsaTensor, builder: anytype, op: ReduceOp, profile: *const Selector) Error!ReduceResult {
        const stats = try selectorStats(profile);
        if (stats.fixed == 0) return .{ .tensor = self };
        try validateSelectorAgainstShape(&self.shape_value, profile);
        if (stats.kept == 0) return .{ .scalar = try self.reduceAll(builder, op) };
        const result_shape = try layout_algebra.sliceTree(&self.shape_value, profile);
        var in_ty: mlir.TextBuffer(128) = .{};
        var out_ty: mlir.TextBuffer(128) = .{};
        try self.vectorType(&in_ty);
        try writeVectorType(&out_ty, result_shape, self.dtype);
        const dims_attr = try reductionDimsAttribute(profile);
        const value = try builder.genericOp(
            "vector.multi_reduction",
            &.{.{ .value = self.value }},
            &.{ .{ .key = "kind", .value = op.attributeValue(self.dtype) }, .{ .key = "dims", .value = dims_attr.slice() } },
            &.{mlir.Type.raw(in_ty.slice())},
            &.{mlir.Type.raw(out_ty.slice())},
        );
        return .{ .tensor = try SsaTensor.init(value, result_shape, self.dtype) };
    }

    pub fn extract(self: SsaTensor, builder: anytype, linear_index: usize) Error!SsaValue {
        if (linear_index >= try self.shape_value.product()) return Error.CoordinateOutOfBounds;
        var vec_ty: mlir.TextBuffer(128) = .{};
        try self.vectorType(&vec_ty);
        const idx = try builder.constantIndex(@intCast(linear_index));
        const value = try builder.genericOp("vector.extractelement", &.{ .{ .value = self.value }, .{ .value = idx } }, &.{}, &.{ mlir.Type.raw(vec_ty.slice()), mlir.Type.index() }, &.{mlir.Type.raw(self.dtype.mlir_type)});
        return SsaValue.init(value, self.dtype);
    }

    pub fn extractCoord(self: SsaTensor, builder: anytype, coord: Tree) Error!SsaValue {
        const index = try columnMajorIndex(self.shape_value, coord);
        return self.extract(builder, @intCast(index));
    }

    pub fn access(self: SsaTensor, builder: anytype, selector: *const Selector) Error!SsaAccess {
        try validateSelectorAgainstShape(&self.shape_value, selector);
        const stats = try selectorStats(selector);
        if (stats.kept == 0) {
            const coord = try selectorToCoord(selector);
            return .{ .element = try self.extractCoord(builder, coord) };
        }
        const result_shape = try layout_algebra.sliceTree(&self.shape_value, selector);
        const slice_spec = try vectorSliceSpec(&self.shape_value, selector);
        var src_col_ty: mlir.TextBuffer(128) = .{};
        var src_row_ty: mlir.TextBuffer(128) = .{};
        var result_row_ty: mlir.TextBuffer(128) = .{};
        try self.vectorType(&src_col_ty);
        try writeRowMajorVectorType(&src_row_ty, self.shape_value, self.dtype);
        try writeRowMajorVectorType(&result_row_ty, slice_spec.sizes_tree, self.dtype);
        const row_value = try self.toVector(builder, .row_major);
        const offsets = try slice_spec.writeOffsetsAttr();
        const sizes = try slice_spec.writeSizesAttr();
        const strides = try slice_spec.writeStridesAttr();
        const raw_slice = try builder.genericOp(
            "vector.extract_strided_slice",
            &.{.{ .value = row_value }},
            &.{ .{ .key = "offsets", .value = offsets.slice() }, .{ .key = "sizes", .value = sizes.slice() }, .{ .key = "strides", .value = strides.slice() } },
            &.{mlir.Type.raw(src_row_ty.slice())},
            &.{mlir.Type.raw(result_row_ty.slice())},
        );
        var final_row_ty: mlir.TextBuffer(128) = .{};
        try writeRowMajorVectorType(&final_row_ty, result_shape, self.dtype);
        const slice_value = if (std.mem.eql(u8, result_row_ty.slice(), final_row_ty.slice()))
            raw_slice
        else
            try builder.genericOp("vector.shape_cast", &.{.{ .value = raw_slice }}, &.{}, &.{mlir.Type.raw(result_row_ty.slice())}, &.{mlir.Type.raw(final_row_ty.slice())});
        const col_slice = try fromVector(builder, slice_value, self.dtype, result_shape, .row_major);
        return .{ .tensor = col_slice };
    }

    pub fn withElement(self: SsaTensor, builder: anytype, coord: Tree, scalar_value: SsaValue) Error!SsaTensor {
        const index = try columnMajorIndex(self.shape_value, coord);
        return self.insert(builder, @intCast(index), try scalar_value.castTo(builder, self.dtype));
    }

    pub fn insert(self: SsaTensor, builder: anytype, linear_index: usize, scalar_value: SsaValue) Error!SsaTensor {
        if (linear_index >= try self.shape_value.product()) return Error.CoordinateOutOfBounds;
        var vec_ty: mlir.TextBuffer(128) = .{};
        try self.vectorType(&vec_ty);
        const idx = try builder.constantIndex(@intCast(linear_index));
        const value = try builder.genericOp(
            "vector.insertelement",
            &.{ .{ .value = scalar_value.value }, .{ .value = self.value }, .{ .value = idx } },
            &.{},
            &.{ mlir.Type.raw(self.dtype.mlir_type), mlir.Type.raw(vec_ty.slice()), mlir.Type.index() },
            &.{mlir.Type.raw(vec_ty.slice())},
        );
        return SsaTensor.init(value, self.shape_value, self.dtype);
    }

    pub fn withSlice(self: SsaTensor, builder: anytype, selector: *const Selector, replacement: SsaTensor) Error!SsaTensor {
        try validateSelectorAgainstShape(&self.shape_value, selector);
        const result_shape = try layout_algebra.sliceTree(&self.shape_value, selector);
        try expectSameShape(result_shape, replacement.shape_value);
        const casted = try replacement.castTo(builder, self.dtype);
        const slice_spec = try vectorSliceSpec(&self.shape_value, selector);
        var dst_row_ty: mlir.TextBuffer(128) = .{};
        var src_row_ty: mlir.TextBuffer(128) = .{};
        try writeRowMajorVectorType(&dst_row_ty, self.shape_value, self.dtype);
        try writeRowMajorVectorType(&src_row_ty, casted.shape_value, casted.dtype);
        const dst_row = try self.toVector(builder, .row_major);
        const src_row = try casted.toVector(builder, .row_major);
        const offsets = try slice_spec.writeOffsetsAttr();
        const strides = try slice_spec.writeStridesAttr();
        const value = try builder.genericOp(
            "vector.insert_strided_slice",
            &.{ .{ .value = src_row }, .{ .value = dst_row } },
            &.{ .{ .key = "offsets", .value = offsets.slice() }, .{ .key = "strides", .value = strides.slice() } },
            &.{ mlir.Type.raw(src_row_ty.slice()), mlir.Type.raw(dst_row_ty.slice()) },
            &.{mlir.Type.raw(dst_row_ty.slice())},
        );
        return fromVector(builder, value, self.dtype, self.shape_value, .row_major);
    }

    pub fn toVector(self: SsaTensor, builder: anytype, order: VectorOrder) Error!mlir.Value {
        switch (order) {
            .column_major => return self.value,
            .row_major => return columnToRowMajor(builder, self),
        }
    }

    pub fn fromVector(builder: anytype, value: mlir.Value, dtype: typing.Numeric, shape_value: Tree, source_order: VectorOrder) Error!SsaTensor {
        const tmp = try SsaTensor.init(value, shape_value, dtype);
        switch (source_order) {
            .column_major => return tmp,
            .row_major => return rowToColumnMajor(builder, tmp),
        }
    }

    pub fn add(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .add, rhs);
    }

    pub fn sub(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .sub, rhs);
    }

    pub fn mul(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .mul, rhs);
    }

    pub fn div(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .div, rhs);
    }

    pub fn mod(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .rem, rhs);
    }

    pub fn min(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .min, rhs);
    }

    pub fn max(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .max, rhs);
    }

    pub fn eql(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .eq, rhs);
    }

    pub fn neql(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .ne, rhs);
    }

    pub fn lt(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .lt, rhs);
    }

    pub fn le(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .le, rhs);
    }

    pub fn gt(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .gt, rhs);
    }

    pub fn ge(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .ge, rhs);
    }

    pub fn bitAnd(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .bit_and, rhs);
    }

    pub fn bitOr(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .bit_or, rhs);
    }

    pub fn bitXor(self: SsaTensor, builder: anytype, rhs: SsaTensor) Error!SsaTensor {
        return self.binary(builder, .bit_xor, rhs);
    }

    pub fn neg(self: SsaTensor, builder: anytype) Error!SsaTensor {
        return self.unary(builder, .neg);
    }

    pub fn abs(self: SsaTensor, builder: anytype) Error!SsaTensor {
        return self.unary(builder, .abs);
    }

    pub fn reduceAllWithInit(self: SsaTensor, builder: anytype, op: ReduceOp, initial: SsaValue) Error!SsaValue {
        const casted_init = try initial.castTo(builder, self.dtype);
        var vec_ty: mlir.TextBuffer(128) = .{};
        try self.vectorType(&vec_ty);
        const value = try builder.genericOp(
            op.mlirName(self.dtype),
            &.{ .{ .value = self.value }, .{ .value = casted_init.value } },
            &.{},
            &.{ mlir.Type.raw(vec_ty.slice()), mlir.Type.raw(self.dtype.mlir_type) },
            &.{mlir.Type.raw(self.dtype.mlir_type)},
        );
        return SsaValue.init(value, self.dtype);
    }

    pub fn reduceProfileWithInit(self: SsaTensor, builder: anytype, op: ReduceOp, profile: *const Selector, initial: SsaTensor) Error!ReduceResult {
        const stats = try selectorStats(profile);
        if (stats.fixed == 0) return .{ .tensor = self };
        try validateSelectorAgainstShape(&self.shape_value, profile);
        if (stats.kept == 0) return .{ .scalar = try self.reduceAllWithInit(builder, op, try initial.scalar()) };
        const result_shape = try layout_algebra.sliceTree(&self.shape_value, profile);
        try expectSameShape(result_shape, initial.shape_value);
        const casted_init = try initial.castTo(builder, self.dtype);
        var in_ty: mlir.TextBuffer(128) = .{};
        var out_ty: mlir.TextBuffer(128) = .{};
        try self.vectorType(&in_ty);
        try writeVectorType(&out_ty, result_shape, self.dtype);
        const dims_attr = try reductionDimsAttribute(profile);
        const value = try builder.genericOp(
            "vector.multi_reduction",
            &.{ .{ .value = self.value }, .{ .value = casted_init.value } },
            &.{ .{ .key = "kind", .value = op.attributeValue(self.dtype) }, .{ .key = "dims", .value = dims_attr.slice() } },
            &.{ mlir.Type.raw(in_ty.slice()), mlir.Type.raw(out_ty.slice()) },
            &.{mlir.Type.raw(out_ty.slice())},
        );
        return .{ .tensor = try SsaTensor.init(value, result_shape, self.dtype) };
    }
};

pub const SsaAccess = union(enum) {
    element: SsaValue,
    tensor: SsaTensor,
};

pub const ReduceResult = union(enum) {
    scalar: SsaValue,
    tensor: SsaTensor,
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
        const unsigned = dtype.kind == .unsigned_int or dtype.kind == .boolean;
        return switch (self) {
            .add => if (dtype.isFloat()) "arith.addf" else "arith.addi",
            .sub => if (dtype.isFloat()) "arith.subf" else "arith.subi",
            .mul => if (dtype.isFloat()) "arith.mulf" else "arith.muli",
            .div => if (dtype.isFloat()) "arith.divf" else if (unsigned) "arith.divui" else "arith.divsi",
            .rem => if (unsigned) "arith.remui" else "arith.remsi",
            .min => if (dtype.isFloat()) "arith.minimumf" else if (unsigned) "arith.minui" else "arith.minsi",
            .max => if (dtype.isFloat()) "arith.maximumf" else if (unsigned) "arith.maxui" else "arith.maxsi",
            .eq, .ne, .lt, .le, .gt, .ge => if (dtype.isFloat()) "arith.cmpf" else "arith.cmpi",
            .bit_and => "arith.andi",
            .bit_or => "arith.ori",
            .bit_xor => "arith.xori",
        };
    }

    pub fn attrs(self: BinaryOp, dtype: typing.Numeric) []const mlir.Attribute {
        const fp = dtype.isFloat();
        const unsigned = dtype.kind == .unsigned_int or dtype.kind == .boolean;
        return switch (self) {
            .eq => if (fp) attr_cmpf_eq[0..] else attr_cmpi_eq[0..],
            .ne => if (fp) attr_cmpf_ne[0..] else attr_cmpi_ne[0..],
            .lt => if (fp) attr_cmpf_lt[0..] else if (unsigned) attr_cmpui_lt[0..] else attr_cmpi_lt[0..],
            .le => if (fp) attr_cmpf_le[0..] else if (unsigned) attr_cmpui_le[0..] else attr_cmpi_le[0..],
            .gt => if (fp) attr_cmpf_gt[0..] else if (unsigned) attr_cmpui_gt[0..] else attr_cmpi_gt[0..],
            .ge => if (fp) attr_cmpf_ge[0..] else if (unsigned) attr_cmpui_ge[0..] else attr_cmpi_ge[0..],
            else => attr_none[0..],
        };
    }

    pub fn resultType(self: BinaryOp, dtype: typing.Numeric) typing.Numeric {
        return switch (self) {
            .eq, .ne, .lt, .le, .gt, .ge => typing.Boolean,
            else => dtype,
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

    pub fn mlirName(self: ReduceOp, dtype: typing.Numeric) []const u8 {
        _ = dtype;
        return switch (self) {
            .add => "vector.reduction.add",
            .mul => "vector.reduction.mul",
            .min => "vector.reduction.min",
            .max => "vector.reduction.max",
            .any => "vector.reduction.or",
            .all => "vector.reduction.and",
        };
    }

    pub fn attributeValue(self: ReduceOp, dtype: typing.Numeric) []const u8 {
        return switch (self) {
            .add => "#vector.kind<add>",
            .mul => "#vector.kind<mul>",
            .min => if (dtype.isFloat()) "#vector.kind<minimumf>" else "#vector.kind<minsi>",
            .max => if (dtype.isFloat()) "#vector.kind<maximumf>" else "#vector.kind<maxsi>",
            .any => "#vector.kind<or>",
            .all => "#vector.kind<and>",
        };
    }
};

pub const GatherScatterPlan = struct {
    mode: usize,
    rank: usize,
    vectorized: bool,
};

pub fn validateGather(input: TensorMeta, mode: usize, index: SsaTensor) Error!GatherScatterPlan {
    try validateGatherScatterStatic(input, mode, index, null);
    return .{ .mode = mode, .rank = input.rank(), .vectorized = try canUseVectorGatherScatter(input, mode) };
}

pub fn validateScatter(output: TensorMeta, mode: usize, index: SsaTensor, data: SsaTensor) Error!GatherScatterPlan {
    try validateGatherScatterStatic(output, mode, index, data);
    return .{ .mode = mode, .rank = output.rank(), .vectorized = try canUseVectorGatherScatter(output, mode) };
}

pub fn emitGather(builder: anytype, input: TensorValue, mode: usize, index: SsaTensor) Error!SsaTensor {
    _ = try validateGather(input.meta, mode, index);
    var input_ty: mlir.TextBuffer(512) = .{};
    var idx_ty: mlir.TextBuffer(128) = .{};
    var out_ty: mlir.TextBuffer(128) = .{};
    try input.meta.tensorTypeText(&input_ty);
    try index.vectorType(&idx_ty);
    try writeVectorType(&out_ty, index.shape_value, input.meta.dtype);
    const mode_attr = try modeAttr(mode);
    const value = try builder.genericOp(
        "cute.gather",
        &.{ .{ .value = input.value }, .{ .value = index.value } },
        &.{.{ .key = "mode", .value = mode_attr.slice() }},
        &.{ mlir.Type.raw(input_ty.slice()), mlir.Type.raw(idx_ty.slice()) },
        &.{mlir.Type.raw(out_ty.slice())},
    );
    return SsaTensor.init(value, index.shape_value, input.meta.dtype);
}

pub fn emitScatter(builder: anytype, output: TensorValue, mode: usize, index: SsaTensor, data: SsaTensor) Error!void {
    _ = try validateScatter(output.meta, mode, index, data);
    var out_ty: mlir.TextBuffer(512) = .{};
    var idx_ty: mlir.TextBuffer(128) = .{};
    var data_ty: mlir.TextBuffer(128) = .{};
    try output.meta.tensorTypeText(&out_ty);
    try index.vectorType(&idx_ty);
    try data.vectorType(&data_ty);
    const mode_attr = try modeAttr(mode);
    try builder.operationNoResult(.{
        .name = "cute.scatter",
        .operands = &.{ .{ .value = output.value }, .{ .value = index.value }, .{ .value = data.value } },
        .attrs = &.{.{ .key = "mode", .value = mode_attr.slice() }},
        .operand_types = &.{ mlir.Type.raw(out_ty.slice()), mlir.Type.raw(idx_ty.slice()), mlir.Type.raw(data_ty.slice()) },
        .result_types = &.{},
    });
}

pub fn makePointerTensor(pointer: runtime.Pointer, layout_value: Layout) Error!TensorMeta {
    return TensorMeta.init(.{ .pointer = pointer }, layout_value, pointer.dtype, pointer.memspace);
}

pub fn makeIdentityTensor(shape: Tree) Error!TensorMeta {
    return TensorMeta.init(.{ .identity = {} }, try Layout.makeCompact(shape), typing.Int32, .generic);
}

pub fn makeFragment(dtype: typing.Numeric, shape: Tree) Error!TensorMeta {
    return TensorMeta.init(.{ .fragment = {} }, try Layout.makeCompact(shape), dtype, .generic);
}

pub fn makeRmemTensor(dtype: typing.Numeric, shape: Tree) Error!TensorMeta {
    return TensorMeta.init(.{ .rmem = {} }, try Layout.makeCompact(shape), dtype, .generic);
}

pub fn full(builder: anytype, shape_value: Tree, dtype: typing.Numeric, scalar_operand: mlir.Operand) Error!SsaTensor {
    return SsaTensor.full(builder, shape_value, dtype, scalar_operand);
}

pub fn fullLike(builder: anytype, source: SsaTensor, scalar_operand: mlir.Operand, dtype: ?typing.Numeric) Error!SsaTensor {
    return SsaTensor.fullLike(builder, source, scalar_operand, dtype);
}

pub fn empty(builder: anytype, shape_value: Tree, dtype: typing.Numeric) Error!SsaTensor {
    return SsaTensor.empty(builder, shape_value, dtype);
}

pub fn emptyLike(builder: anytype, source: SsaTensor, dtype: ?typing.Numeric) Error!SsaTensor {
    return SsaTensor.emptyLike(builder, source, dtype);
}

pub fn zerosLike(builder: anytype, source: SsaTensor) Error!SsaTensor {
    return SsaTensor.zeros(builder, source.shape_value, source.dtype);
}

pub fn onesLike(builder: anytype, source: SsaTensor) Error!SsaTensor {
    return SsaTensor.ones(builder, source.shape_value, source.dtype);
}

pub fn where(builder: anytype, cond: SsaTensor, if_value: SsaTensor, else_value: SsaTensor) Error!SsaTensor {
    return SsaTensor.where_(builder, cond, if_value, else_value);
}

pub fn writeVectorType(out: anytype, shape_value: Tree, dtype: typing.Numeric) Error!void {
    try out.append("vector<");
    try out.appendUnsigned(@intCast(try shape_value.product()));
    try out.append("x");
    try out.append(dtype.mlir_type);
    try out.append(">");
}

pub fn writeRowMajorVectorType(out: anytype, shape_value: Tree, dtype: typing.Numeric) Error!void {
    const flat = try shape_value.flattenLeaves();
    try out.append("vector<");
    for (flat.slice(), 0..) |extent, i| {
        if (i != 0) try out.append("x");
        if (extent <= 0) return Error.InvalidTensorShape;
        try out.appendUnsigned(@intCast(extent));
    }
    if (flat.len != 0) try out.append("x");
    try out.append(dtype.mlir_type);
    try out.append(">");
}

fn memoryNumeric(dtype: typing.Numeric) typing.Numeric {
    return if (dtype.kind == .boolean) typing.Int8 else dtype;
}

fn columnMajorIndex(shape_value: Tree, coord: Tree) Error!Scalar {
    const l = try Layout.makeCompact(shape_value);
    return l.crd2idx(coord);
}

fn rowMajorIndex(shape_value: Tree, coord: Tree) Error!Scalar {
    const l = try Layout.makeCompactRight(shape_value);
    return l.crd2idx(coord);
}

fn coordFromFlatIndexColumn(shape_value: Tree, index: Scalar) Error!Tree {
    return layout_algebra.idx2crdShape(index, &shape_value);
}

fn selectorToCoord(selector: *const Selector) Error!Tree {
    var out: Tree = .{};
    out.root = try selectorToCoordSub(&out, selector, selector.root);
    return out;
}

fn selectorToCoordSub(out: *Tree, selector: *const Selector, id: u16) Error!u16 {
    switch (selector.nodes.at(id)) {
        .leaf => |leaf| switch (leaf) {
            .fixed => |value| {
                const index = out.nodes.len;
                try out.nodes.append(.{ .leaf = value });
                return @intCast(index);
            },
            .keep => return Error.InvalidVectorSlice,
        },
        .tuple => |span| {
            var child_ids: [layout.max_children]u16 = undefined;
            if (span.len > layout.max_children) return Error.OutOfCapacity;
            for (0..span.len) |i| child_ids[i] = try selectorToCoordSub(out, selector, selector.children.at(span.start + i));
            const child_start = out.children.len;
            for (child_ids[0..span.len]) |child_id| try out.children.append(child_id);
            const index = out.nodes.len;
            try out.nodes.append(.{ .tuple = .{ .start = child_start, .len = span.len } });
            return @intCast(index);
        },
    }
}

const VectorSliceSpec = struct {
    offsets: layout.Flat = .{},
    sizes: layout.Flat = .{},
    strides: layout.Flat = .{},
    sizes_tree: Tree,

    pub fn writeOffsetsAttr(self: VectorSliceSpec) Error!mlir.TextBuffer(256) {
        return writeFlatArrayAttr(self.offsets.slice());
    }

    pub fn writeSizesAttr(self: VectorSliceSpec) Error!mlir.TextBuffer(256) {
        return writeFlatArrayAttr(self.sizes.slice());
    }

    pub fn writeStridesAttr(self: VectorSliceSpec) Error!mlir.TextBuffer(256) {
        return writeFlatArrayAttr(self.strides.slice());
    }
};

fn vectorSliceSpec(shape_value: *const Tree, selector: *const Selector) Error!VectorSliceSpec {
    var spec = VectorSliceSpec{ .sizes_tree = undefined };
    try collectVectorSliceSpec(shape_value, shape_value.root, selector, selector.root, &spec);
    spec.sizes_tree = try Tree.fromProfileAndLeaves(shape_value, spec.sizes.slice());
    return spec;
}

fn collectVectorSliceSpec(shape_value: *const Tree, shape_id: u16, selector: *const Selector, selector_id: u16, spec: *VectorSliceSpec) Error!void {
    switch (selector.nodes.at(selector_id)) {
        .leaf => |leaf| {
            const extent = switch (shape_value.nodes.at(shape_id)) {
                .leaf => |v| v,
                .tuple => return Error.ProfileMismatch,
            };
            switch (leaf) {
                .fixed => |value| {
                    if (value < 0 or value >= extent) return Error.CoordinateOutOfBounds;
                    try spec.offsets.append(value);
                    try spec.sizes.append(1);
                    try spec.strides.append(1);
                },
                .keep => {
                    try spec.offsets.append(0);
                    try spec.sizes.append(extent);
                    try spec.strides.append(1);
                },
            }
        },
        .tuple => |sel_span| {
            const shape_span = switch (shape_value.nodes.at(shape_id)) {
                .tuple => |span| span,
                .leaf => return Error.ProfileMismatch,
            };
            if (shape_span.len != sel_span.len) return Error.ProfileMismatch;
            for (0..sel_span.len) |i| try collectVectorSliceSpec(shape_value, shape_value.children.at(shape_span.start + i), selector, selector.children.at(sel_span.start + i), spec);
        },
    }
}

fn writeFlatArrayAttr(values: []const Scalar) Error!mlir.TextBuffer(256) {
    var out: mlir.TextBuffer(256) = .{};
    try out.append("[");
    for (values, 0..) |value, i| {
        if (i != 0) try out.append(", ");
        try out.appendSigned(value);
    }
    try out.append("]");
    return out;
}

fn writePermutationAttr(perm: []const usize) Error!mlir.TextBuffer(1024) {
    var out: mlir.TextBuffer(1024) = .{};
    try out.append("[");
    for (perm, 0..) |value, i| {
        if (i != 0) try out.append(", ");
        try out.appendUnsigned(value);
    }
    try out.append("]");
    return out;
}

fn columnToRowMajor(builder: anytype, src: SsaTensor) Error!mlir.Value {
    return reorderVector(builder, src, .column_major, .row_major);
}

fn rowToColumnMajor(builder: anytype, src: SsaTensor) Error!SsaTensor {
    const value = try reorderVector(builder, src, .row_major, .column_major);
    return SsaTensor.init(value, src.shape_value, src.dtype);
}

fn reorderVector(builder: anytype, src: SsaTensor, from: VectorOrder, to: VectorOrder) Error!mlir.Value {
    if (from == to) return src.value;
    const count_u = try src.shape_value.product();
    if (count_u > layout.max_leaves * layout.max_leaves) return Error.OutOfCapacity;
    const count: usize = @intCast(count_u);
    var perm_buf: [layout.max_leaves * layout.max_leaves]usize = undefined;
    for (0..count) |out_i| {
        const out_coord = switch (to) {
            .column_major => try coordFromFlatIndexColumn(src.shape_value, @intCast(out_i)),
            .row_major => try coordFromFlatIndexRow(src.shape_value, @intCast(out_i)),
        };
        const in_index = switch (from) {
            .column_major => try columnMajorIndex(src.shape_value, out_coord),
            .row_major => try rowMajorIndex(src.shape_value, out_coord),
        };
        if (in_index < 0 or in_index >= count) return Error.InvalidVectorOrder;
        perm_buf[out_i] = @intCast(in_index);
    }
    var flat_ty: mlir.TextBuffer(128) = .{};
    var result_ty: mlir.TextBuffer(128) = .{};
    try writeVectorType(&flat_ty, src.shape_value, src.dtype);
    if (to == .row_major) {
        try writeRowMajorVectorType(&result_ty, src.shape_value, src.dtype);
    } else {
        try writeVectorType(&result_ty, src.shape_value, src.dtype);
    }
    const mask = try writePermutationAttr(perm_buf[0..count]);
    const shuffled = try builder.genericOp(
        "vector.shuffle",
        &.{ .{ .value = src.value }, .{ .value = src.value } },
        &.{.{ .key = "mask", .value = mask.slice() }},
        &.{ mlir.Type.raw(flat_ty.slice()), mlir.Type.raw(flat_ty.slice()) },
        &.{mlir.Type.raw(flat_ty.slice())},
    );
    if (to == .row_major and !std.mem.eql(u8, flat_ty.slice(), result_ty.slice())) {
        return builder.genericOp("vector.shape_cast", &.{.{ .value = shuffled }}, &.{}, &.{mlir.Type.raw(flat_ty.slice())}, &.{mlir.Type.raw(result_ty.slice())});
    }
    if (to == .column_major and !std.mem.eql(u8, flat_ty.slice(), result_ty.slice())) {
        return builder.genericOp("vector.shape_cast", &.{.{ .value = shuffled }}, &.{}, &.{mlir.Type.raw(flat_ty.slice())}, &.{mlir.Type.raw(result_ty.slice())});
    }
    return shuffled;
}

fn coordFromFlatIndexRow(shape_value: Tree, index: Scalar) Error!Tree {
    if (index < 0) return Error.CoordinateOutOfBounds;
    const shapes = try shape_value.flattenLeaves();
    var coords: layout.Flat = .{};
    for (0..shapes.len) |_| try coords.append(0);
    var remaining = index;
    var i = shapes.len;
    while (i > 0) {
        i -= 1;
        const extent = shapes.at(i);
        if (extent <= 0) return Error.InvalidShape;
        coords.set(i, @mod(remaining, extent));
        remaining = @divFloor(remaining, extent);
    }
    if (remaining != 0) return Error.CoordinateOutOfBounds;
    return Tree.fromProfileAndLeaves(&shape_value, coords.slice());
}

fn selectorStats(selector: *const Selector) Error!struct { kept: usize, fixed: usize } {
    var kept: usize = 0;
    var fixed: usize = 0;
    try selectorStatsSub(selector, selector.root, &kept, &fixed);
    return .{ .kept = kept, .fixed = fixed };
}

fn selectorStatsSub(selector: *const Selector, id: u16, kept: *usize, fixed: *usize) Error!void {
    switch (selector.nodes.at(id)) {
        .leaf => |leaf| switch (leaf) {
            .keep => kept.* += 1,
            .fixed => fixed.* += 1,
        },
        .tuple => |span| for (0..span.len) |i| try selectorStatsSub(selector, selector.children.at(span.start + i), kept, fixed),
    }
}

fn validateSelectorAgainstLayout(l: *const Layout, selector: *const Selector) Error!void {
    try validateSelectorAgainstShape(&l.shape, selector);
}

fn validateSelectorAgainstShape(shape: *const Tree, selector: *const Selector) Error!void {
    try validateSelectorShapeSub(shape, shape.root, selector, selector.root);
}

fn validateSelectorShapeSub(shape: *const Tree, shape_id: u16, selector: *const Selector, selector_id: u16) Error!void {
    switch (selector.nodes.at(selector_id)) {
        .leaf => |leaf| switch (leaf) {
            .keep => {},
            .fixed => |value| {
                const extent = switch (shape.nodes.at(shape_id)) {
                    .leaf => |v| v,
                    .tuple => return Error.ProfileMismatch,
                };
                if (value < 0 or value >= extent) return Error.CoordinateOutOfBounds;
            },
        },
        .tuple => |sel_span| {
            const shape_span = switch (shape.nodes.at(shape_id)) {
                .tuple => |span| span,
                .leaf => return Error.ProfileMismatch,
            };
            if (shape_span.len != sel_span.len) return Error.ProfileMismatch;
            for (0..sel_span.len) |i| try validateSelectorShapeSub(shape, shape.children.at(shape_span.start + i), selector, selector.children.at(sel_span.start + i));
        },
    }
}

fn selectorOffset(l: *const Layout, selector: *const Selector) Error!Scalar {
    const sliced = try layout_algebra.sliceAndOffset(l, selector);
    return sliced.offset;
}

fn expectSameShape(lhs: Tree, rhs: Tree) Error!void {
    if (!lhs.equals(&rhs)) return Error.IncompatibleTensorShapes;
}

fn validateMaskShape(data_shape: Tree, mask_shape: Tree) Error!void {
    if (!data_shape.equals(&mask_shape)) return Error.InvalidMaskShape;
}

fn checkVectorLoadStore(meta: TensorMeta) Error!void {
    switch (meta.engine) {
        .pointer, .mlir_value => {},
        else => return Error.InvalidTensorEngine,
    }
    try meta.layout_value.shape.assertPositive();
}

fn checkNarrowStoreAlignment(dtype: typing.Numeric, shape: Tree) Error!void {
    if (dtype.width >= 8) return;
    const bits = (try shape.product()) * dtype.width;
    if (bits % 32 != 0) return Error.NarrowPrecisionAlignment;
}

fn inferBroadcastShape(lhs: Tree, rhs: Tree) Error!Tree {
    if (lhs.equals(&rhs)) return lhs;
    const lprod = try lhs.product();
    const rprod = try rhs.product();
    if (lprod == 1) return rhs;
    if (rprod == 1) return lhs;

    const lflat = try lhs.flattenLeaves();
    const rflat = try rhs.flattenLeaves();
    const n = @max(lflat.len, rflat.len);
    var leaves: [layout.max_leaves]Scalar = undefined;
    if (n > leaves.len) return Error.OutOfCapacity;
    for (0..n) |i| {
        const lv = if (i < lflat.len) lflat.at(i) else 1;
        const rv = if (i < rflat.len) rflat.at(i) else 1;
        if (lv == rv) {
            leaves[i] = lv;
        } else if (lv == 1) {
            leaves[i] = rv;
        } else if (rv == 1) {
            leaves[i] = lv;
        } else {
            return Error.BroadcastMismatch;
        }
    }
    return treeFromScalars(leaves[0..n]);
}

fn treeFromScalars(leaves: []const Scalar) Error!Tree {
    if (leaves.len == 0) return Error.InvalidShape;
    if (leaves.len == 1) return Tree.initLeaf(leaves[0]);
    var parts: [layout.max_leaves]Tree = undefined;
    if (leaves.len > parts.len) return Error.OutOfCapacity;
    for (leaves, 0..) |v, i| parts[i] = try Tree.initLeaf(v);
    return Tree.initTuple(parts[0..leaves.len]);
}

fn sameNumeric(a: typing.Numeric, b: typing.Numeric) bool {
    return a.width == b.width and a.kind == b.kind and std.mem.eql(u8, a.mlir_type, b.mlir_type);
}

fn conversionOp(src: typing.Numeric, dst: typing.Numeric) []const u8 {
    if (src.isFloat() and dst.isFloat()) return if (dst.width >= src.width) "arith.extf" else "arith.truncf";
    if (src.isFloat() and dst.isInteger()) return if (dst.kind == .unsigned_int or dst.kind == .boolean) "arith.fptoui" else "arith.fptosi";
    if (src.isInteger() and dst.isFloat()) return if (src.kind == .unsigned_int or src.kind == .boolean) "arith.uitofp" else "arith.sitofp";
    if (dst.width >= src.width) return if (src.kind == .unsigned_int or src.kind == .boolean) "arith.extui" else "arith.extsi";
    return "arith.trunci";
}

fn reductionDimsAttribute(selector: *const Selector) Error!mlir.TextBuffer(128) {
    var dims: mlir.TextBuffer(128) = .{};
    try dims.append("[");
    var leaf_index: usize = 0;
    var first = true;
    try reductionDimsAttributeSub(selector, selector.root, &dims, &leaf_index, &first);
    try dims.append("]");
    return dims;
}

fn reductionDimsAttributeSub(selector: *const Selector, id: u16, out: *mlir.TextBuffer(128), leaf_index: *usize, first: *bool) Error!void {
    switch (selector.nodes.at(id)) {
        .leaf => |leaf| {
            switch (leaf) {
                .fixed => {
                    if (!first.*) try out.append(", ");
                    first.* = false;
                    try out.appendUnsigned(leaf_index.*);
                },
                .keep => {},
            }
            leaf_index.* += 1;
        },
        .tuple => |span| for (0..span.len) |i| try reductionDimsAttributeSub(selector, selector.children.at(span.start + i), out, leaf_index, first),
    }
}

fn modeAttr(mode: usize) Error!mlir.TextBuffer(32) {
    var out: mlir.TextBuffer(32) = .{};
    try out.appendUnsigned(mode);
    try out.append(" : i64");
    return out;
}

fn validateGatherScatterStatic(base: TensorMeta, mode: usize, index: SsaTensor, data: ?SsaTensor) Error!void {
    const n_modes = base.rank();
    if (mode >= n_modes) return Error.InvalidGatherScatterMode;
    if (index.shape_value.rank() != n_modes) return Error.RankMismatch;
    if (data) |d| try expectSameShape(index.shape_value, d.shape_value);
    const base_shape = try base.layout_value.shape.flattenLeaves();
    const idx_shape = try index.shape_value.flattenLeaves();
    for (0..n_modes) |i| {
        if (i != mode and idx_shape.at(i) > base_shape.at(i)) return Error.InvalidTensorShape;
    }
    if (!index.dtype.isInteger()) return Error.InvalidElementType;
}

fn canUseVectorGatherScatter(base: TensorMeta, mode: usize) Error!bool {
    if (mode != 0) return false;
    const st = try base.layout_value.stride.flattenLeaves();
    return st.len != 0 and st.at(0) == 1;
}

pub const TensorLikeSource = union(enum) {
    meta: TensorMeta,
    ssa: SsaTensor,
    layout_value: Layout,
    shape_value: Tree,
};

pub const MakeTensorInput = union(enum) {
    pointer: runtime.Pointer,
    identity_scalar: Scalar,
    identity_tuple: Tree,
    mlir_value: struct { value: mlir.Value, dtype: typing.Numeric, memspace: typing.AddressSpace = .generic },
};

pub const TensorBuild = struct {
    meta: TensorMeta,
    value: ?TensorValue = null,
};

pub fn makeTensor(input: MakeTensorInput, layout_value: Layout) Error!TensorMeta {
    return switch (input) {
        .pointer => |p| makePointerTensor(p, layout_value),
        .identity_scalar => |base| makeIdentityIteratorTensor(try Tree.initLeaf(base), layout_value),
        .identity_tuple => |base| makeIdentityIteratorTensor(base, layout_value),
        .mlir_value => |v| TensorMeta.init(.{ .mlir_value = v.value }, layout_value, v.dtype, v.memspace),
    };
}

pub fn emitMakeTensor(builder: anytype, input: MakeTensorInput, layout_value: Layout) Error!TensorBuild {
    const meta = try makeTensor(input, layout_value);
    var type_buf: mlir.TextBuffer(512) = .{};
    try meta.tensorTypeText(&type_buf);
    const result_value = switch (input) {
        .pointer => try builder.genericOp("cute.make_tensor", &.{.{ .value = mlir.Value.arg(0) }}, &.{}, &.{mlir.Type.raw("!cute.ptr")}, &.{mlir.Type.raw(type_buf.slice())}),
        .identity_scalar, .identity_tuple => try builder.genericOp("cute.make_coord_tensor", &.{}, &.{}, &.{}, &.{mlir.Type.raw(type_buf.slice())}),
        .mlir_value => |v| try builder.genericOp("cute.make_tensor", &.{.{ .value = v.value }}, &.{}, &.{mlir.Type.raw(v.dtype.mlir_type)}, &.{mlir.Type.raw(type_buf.slice())}),
    };
    return .{ .meta = meta, .value = try TensorValue.initFromMeta(meta, result_value) };
}

pub fn makeIdentityIteratorTensor(base: Tree, layout_value: Layout) Error!TensorMeta {
    // Coordinate tensors carry an arithmetic tuple iterator.  The dtype is the
    // scalar type used to materialize coordinates when an element is selected.
    _ = base;
    return TensorMeta.init(.{ .identity = {} }, layout_value, typing.Int32, .generic);
}

pub fn makeRmemTensorFromLayout(layout_value: Layout, dtype: typing.Numeric) Error!TensorMeta {
    var meta = try TensorMeta.init(.{ .rmem = {} }, layout_value, dtype, .generic);
    if (meta.dtype.kind == .boolean) meta.dtype = typing.Boolean;
    return meta;
}

pub fn makeRmemTensorLike(src: TensorLikeSource, dtype: ?typing.Numeric) Error!TensorMeta {
    const shape_value = switch (src) {
        .meta => |m| m.layout_value.shape,
        .ssa => |s| s.shape_value,
        .layout_value => |l| l.shape,
        .shape_value => |shape| shape,
    };
    const source_dtype = switch (src) {
        .meta => |m| m.dtype,
        .ssa => |s| s.dtype,
        else => typing.Float32,
    };
    return makeRmemTensorFromLayout(try Layout.makeCompact(shape_value), dtype orelse source_dtype);
}

pub fn makeFragmentLike(src: TensorLikeSource, dtype: ?typing.Numeric) Error!TensorMeta {
    // Fragment is kept as a distinct engine so later MMA lowering can
    // distinguish it from general rmem allocation.
    const shape_value = switch (src) {
        .meta => |m| m.layout_value.shape,
        .ssa => |s| s.shape_value,
        .layout_value => |l| l.shape,
        .shape_value => |shape| shape,
    };
    const source_dtype = switch (src) {
        .meta => |m| m.dtype,
        .ssa => |s| s.dtype,
        else => typing.Float32,
    };
    return TensorMeta.init(.{ .fragment = {} }, try Layout.makeCompact(shape_value), dtype orelse source_dtype, .generic);
}

pub fn makeTensorSsa(value: mlir.Value, shape_value: Tree, dtype: typing.Numeric) Error!SsaTensor {
    return SsaTensor.init(value, shape_value, dtype);
}

pub fn makeFragmentLikeSsa(builder: anytype, src: SsaTensor, dtype: ?typing.Numeric) Error!SsaTensor {
    return SsaTensor.empty(builder, src.shape_value, dtype orelse src.dtype);
}

pub fn makeRmemTensorLikeSsa(builder: anytype, src: SsaTensor, dtype: ?typing.Numeric) Error!SsaTensor {
    return SsaTensor.empty(builder, src.shape_value, dtype orelse src.dtype);
}

pub fn domainOffset(meta: TensorMeta, coord: Tree) Error!TensorMeta {
    var out = meta;
    const offset = try meta.elementOffset(coord);
    switch (out.engine) {
        .pointer => |p| out.engine = .{ .pointer = p.add(@intCast(offset)) },
        else => {},
    }
    return out;
}

pub fn recastTensor(meta: TensorMeta, dtype: typing.Numeric) Error!TensorMeta {
    return meta.recast(dtype);
}

pub fn any(builder: anytype, src: SsaTensor) Error!SsaValue {
    if (src.dtype.kind != .boolean) return Error.InvalidElementType;
    return src.reduceAll(builder, .any);
}

pub fn all(builder: anytype, src: SsaTensor) Error!SsaValue {
    if (src.dtype.kind != .boolean) return Error.InvalidElementType;
    return src.reduceAll(builder, .all);
}

pub fn reduce(builder: anytype, src: SsaTensor, op: ReduceOp, profile: ?*const Selector, init: ?SsaTensor) Error!ReduceResult {
    if (profile) |p| {
        if (init) |i| return src.reduceProfileWithInit(builder, op, p, i);
        return src.reduceProfile(builder, op, p);
    }
    if (init) |i| return .{ .scalar = try src.reduceAllWithInit(builder, op, try i.scalar()) };
    return .{ .scalar = try src.reduceAll(builder, op) };
}

pub fn emitGatherOptimized(builder: anytype, input: TensorValue, mode: usize, index: SsaTensor) Error!SsaTensor {
    const plan = try validateGather(input.meta, mode, index);
    if (!plan.vectorized) return emitGather(builder, input, mode, index);

    var idx_ty: mlir.TextBuffer(128) = .{};
    var out_ty: mlir.TextBuffer(128) = .{};
    var mask_ty: mlir.TextBuffer(128) = .{};
    try index.vectorType(&idx_ty);
    try writeVectorType(&out_ty, index.shape_value, input.meta.dtype);
    try writeVectorType(&mask_ty, index.shape_value, typing.Boolean);
    const mask = try builder.genericOp("vector.constant_mask", &.{}, &.{.{ .key = "dim_sizes", .value = "all" }}, &.{}, &.{mlir.Type.raw(mask_ty.slice())});
    const pass = try builder.genericOp("llvm.mlir.poison", &.{}, &.{}, &.{}, &.{mlir.Type.raw(out_ty.slice())});
    const value = try builder.genericOp(
        "vector.gather",
        &.{ .{ .value = input.value }, .{ .value = index.value }, .{ .value = mask }, .{ .value = pass } },
        &.{.{ .key = "alignment", .value = "1 : i64" }},
        &.{ input.type_(), mlir.Type.raw(idx_ty.slice()), mlir.Type.raw(mask_ty.slice()), mlir.Type.raw(out_ty.slice()) },
        &.{mlir.Type.raw(out_ty.slice())},
    );
    return SsaTensor.init(value, index.shape_value, input.meta.dtype);
}

pub fn emitScatterOptimized(builder: anytype, output: TensorValue, mode: usize, index: SsaTensor, data: SsaTensor) Error!void {
    const plan = try validateScatter(output.meta, mode, index, data);
    if (!plan.vectorized) return emitScatter(builder, output, mode, index, data);

    var idx_ty: mlir.TextBuffer(128) = .{};
    var data_ty: mlir.TextBuffer(128) = .{};
    var mask_ty: mlir.TextBuffer(128) = .{};
    try index.vectorType(&idx_ty);
    try data.vectorType(&data_ty);
    try writeVectorType(&mask_ty, index.shape_value, typing.Boolean);
    const mask = try builder.genericOp("vector.constant_mask", &.{}, &.{.{ .key = "dim_sizes", .value = "all" }}, &.{}, &.{mlir.Type.raw(mask_ty.slice())});
    try builder.operationNoResult(.{
        .name = "vector.scatter",
        .operands = &.{ .{ .value = output.value }, .{ .value = index.value }, .{ .value = mask }, .{ .value = data.value } },
        .attrs = &.{.{ .key = "alignment", .value = "1 : i64" }},
        .operand_types = &.{ output.type_(), mlir.Type.raw(idx_ty.slice()), mlir.Type.raw(mask_ty.slice()), mlir.Type.raw(data_ty.slice()) },
        .result_types = &.{},
    });
}

pub fn promoteNumeric(lhs: typing.Numeric, rhs: typing.Numeric) typing.Numeric {
    if (sameNumeric(lhs, rhs)) return lhs;
    if (lhs.isFloat() or rhs.isFloat()) {
        return if (lhs.width >= rhs.width and lhs.isFloat()) lhs else if (rhs.isFloat()) rhs else lhs;
    }
    if (lhs.width > rhs.width) return lhs;
    if (rhs.width > lhs.width) return rhs;
    if (lhs.kind == .unsigned_int) return lhs;
    return rhs;
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

test "tensor_ssa: tensor metadata selector access offsets pointer slices" {
    const lay = layout.makeLayout(.{ 4, 5 }, .{ 5, 1 });
    const ptr = try runtime.Pointer.init(0x1000, typing.Float32, .gmem, null);
    const meta = try makePointerTensor(ptr, lay);

    const element_sel = Selector.fromComptime(.{ 2, 3 });
    const element = (try meta.access(&element_sel)).element;
    try std.testing.expectEqual(@as(Scalar, 13), element.offset);

    const slice_sel = Selector.fromComptime(.{ 2, keep });
    const sub = (try meta.access(&slice_sel)).tensor;
    try std.testing.expectEqual(@as(usize, 0x1000 + 10 * 4), sub.engine.pointer.address);
    try std.testing.expectEqualSlices(Scalar, &.{5}, (try sub.layout_value.shape.flattenLeaves()).slice());
}

test "tensor_ssa: TensorValue emits scalar load and vector store" {
    var b: mlir.Builder(4096) = .{};
    const lay = layout.makeLayout(.{ 4, 4 }, .{ 4, 1 });
    const ptr = try runtime.Pointer.init(0x2000, typing.Float32, .gmem, null);
    const meta = try makePointerTensor(ptr, lay);
    const tv = TensorValue.init(meta, mlir.Value.arg(0), "!cute.memref<ptr<f32, gmem, align=4>, layout<shape=(4,4), stride=(4,1)>>");
    const sel = Selector.fromComptime(.{ 1, 2 });
    const access_value = try tv.access(&b, &sel);
    switch (access_value) {
        .element => {},
        .tensor => return Error.InvalidTensorAccess,
    }
    const zero = try b.constantF(0.0, mlir.Type.f(32));
    const data = try SsaTensor.full(&b, meta.layout_value.shape, typing.Float32, .{ .value = zero });
    try tv.store(&b, data, null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.memref.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.memref.store_vec") != null);
}

test "tensor_ssa: SSA tensor arithmetic cast reshape bitcast where reduce" {
    var b: mlir.Builder(8192) = .{};
    const shape = Tree.fromComptime(.{ 2, 2 });
    const zero = try b.constantI(0, mlir.Type.i(32));
    const one = try b.constantI(1, mlir.Type.i(32));
    const lhs = try SsaTensor.full(&b, shape, typing.Int32, .{ .value = zero });
    const rhs = try SsaTensor.full(&b, Tree.fromComptime(.{1}), typing.Int32, .{ .value = one });
    const sum = try lhs.binary(&b, .add, rhs);
    const cmp = try sum.binary(&b, .gt, lhs);
    _ = try SsaTensor.where_(&b, cmp, sum, lhs);
    _ = try sum.reshape(&b, Tree.fromComptime(.{4}));
    _ = try sum.bitcast(&b, typing.Int16);
    _ = try sum.reduceAll(&b, .add);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "arith.addi") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "arith.cmpi") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "vector.bitcast") != null);
}

test "tensor_ssa: profile reduction emits multi reduction and preserves kept shape" {
    var b: mlir.Builder(4096) = .{};
    const shape = Tree.fromComptime(.{ 4, 8 });
    const zero = try b.constantF(0.0, mlir.Type.f(32));
    const src = try SsaTensor.full(&b, shape, typing.Float32, .{ .value = zero });
    const prof = Selector.fromComptime(.{ keep, 0 });
    const reduced = try src.reduceProfile(&b, .add, &prof);
    const out = reduced.tensor;
    try std.testing.expectEqualSlices(Scalar, &.{4}, (try out.shape_value.flattenLeaves()).slice());
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "vector.multi_reduction") != null);
}

test "tensor_ssa: gather scatter validation and emission" {
    var b: mlir.Builder(8192) = .{};
    const lay = layout.makeLayout(.{ 8, 4 }, .{ 1, 8 });
    const ptr = try runtime.Pointer.init(0x4000, typing.Float32, .gmem, null);
    const meta = try makePointerTensor(ptr, lay);
    const tv = TensorValue.init(meta, mlir.Value.arg(0), "!cute.memref<ptr<f32, gmem, align=4>, layout<shape=(8,4), stride=(1,8)>>");
    const idx_zero = try b.constantI(0, mlir.Type.i(32));
    const index = try SsaTensor.full(&b, Tree.fromComptime(.{ 8, 4 }), typing.Int32, .{ .value = idx_zero });
    const gathered = try emitGather(&b, tv, 0, index);
    try emitScatter(&b, tv, 0, index, gathered);
    try std.testing.expect((try validateGather(meta, 0, index)).vectorized);
    try std.testing.expectError(Error.InvalidGatherScatterMode, validateGather(meta, 2, index));
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.gather") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.scatter") != null);
}

test "tensor_ssa+: TensorSSA vector order conversion and coordinate access" {
    var b: mlir.Builder(8192) = .{};
    const shape = Tree.fromComptime(.{ 2, 3 });
    const zero = try b.constantI(0, mlir.Type.i(32));
    const src = try SsaTensor.full(&b, shape, typing.Int32, .{ .value = zero });
    _ = try src.toVector(&b, .row_major);
    const coord = Tree.fromComptime(.{ 1, 2 });
    const elem = try src.extractCoord(&b, coord);
    try std.testing.expectEqualStrings("Int32", elem.dtype.name);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "vector.shuffle") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "vector.extractelement") != null);
}

test "tensor_ssa+: TensorSSA slicing and slice insertion emit strided vector ops" {
    var b: mlir.Builder(16384) = .{};
    const shape = Tree.fromComptime(.{ 4, 5 });
    const zero = try b.constantF(0.0, mlir.Type.f(32));
    const src = try SsaTensor.full(&b, shape, typing.Float32, .{ .value = zero });
    const sel = Selector.fromComptime(.{ 2, keep });
    const sliced = try src.access(&b, &sel);
    const sub = switch (sliced) {
        .tensor => |t| t,
        .element => return Error.InvalidVectorSlice,
    };
    try std.testing.expectEqualSlices(Scalar, &.{5}, (try sub.shape_value.flattenLeaves()).slice());
    _ = try src.withSlice(&b, &sel, sub);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "vector.extract_strided_slice") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "vector.insert_strided_slice") != null);
}

test "tensor_ssa+: TensorSSA element replacement and rank broadcast" {
    var b: mlir.Builder(8192) = .{};
    const zero = try b.constantI(0, mlir.Type.i(32));
    const one = try b.constantI(1, mlir.Type.i(32));
    const a = try SsaTensor.full(&b, Tree.fromComptime(.{ 2, 1 }), typing.Int32, .{ .value = zero });
    const c = try SsaTensor.full(&b, Tree.fromComptime(.{ 1, 3 }), typing.Int32, .{ .value = one });
    const sum = try a.binary(&b, .add, c);
    try std.testing.expectEqualSlices(Scalar, &.{ 2, 3 }, (try sum.shape_value.flattenLeaves()).slice());
    const replaced = try sum.withElement(&b, Tree.fromComptime(.{ 1, 2 }), SsaValue.init(one, typing.Int32));
    try std.testing.expectEqualSlices(Scalar, &.{ 2, 3 }, (try replaced.shape_value.flattenLeaves()).slice());
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "vector.insertelement") != null);
}

test "tensor_ssa+: boolean vector load store uses i8 memory representation" {
    var b: mlir.Builder(16384) = .{};
    const lay = layout.makeLayout(.{32}, .{1});
    const ptr = try runtime.Pointer.init(0x8000, typing.Boolean, .gmem, null);
    const meta = try makePointerTensor(ptr, lay);
    const tv = TensorValue.init(meta, mlir.Value.arg(0), "!cute.memref<ptr<i8, gmem, align=1>, layout<shape=32, stride=1>>");
    const data = try tv.load(&b, null, null);
    try std.testing.expectEqual(typing.NumericKind.boolean, data.dtype.kind);
    try tv.store(&b, data, null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "vector<32xi8>") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "arith.cmpi") != null);
}

test "tensor_ssa+: full_like empty_like and bool ir_value_int8 helpers" {
    var b: mlir.Builder(8192) = .{};
    const flag = try b.constantI(1, mlir.Type.i(1));
    const src = try SsaTensor.full(&b, Tree.fromComptime(.{4}), typing.Boolean, .{ .value = flag });
    _ = try src.irValueInt8(&b);
    _ = try fullLike(&b, src, .{ .value = flag }, null);
    _ = try emptyLike(&b, src, typing.Int8);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.undef") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "arith.extui") != null);
}

test "tensor_ssa finish: make_tensor variants and tensor-like constructors" {
    var b: mlir.Builder(8192) = .{};
    const lay = layout.makeLayout(.{ 2, 3 }, .{ 3, 1 });
    const ptr = try runtime.Pointer.init(0x9000, typing.Float16, .gmem, null);
    const ptr_meta = try makeTensor(.{ .pointer = ptr }, lay);
    try std.testing.expectEqual(typing.NumericKind.float, ptr_meta.dtype.kind);
    const id_meta = try makeTensor(.{ .identity_scalar = 7 }, lay);
    try std.testing.expectEqual(typing.NumericKind.signed_int, id_meta.dtype.kind);
    const mlir_meta = try makeTensor(.{ .mlir_value = .{ .value = mlir.Value.arg(1), .dtype = typing.Int32, .memspace = .smem } }, lay);
    try std.testing.expectEqual(typing.AddressSpace.smem, mlir_meta.memspace);
    _ = try emitMakeTensor(&b, .{ .mlir_value = .{ .value = mlir.Value.arg(1), .dtype = typing.Int32, .memspace = .smem } }, lay);

    const ssa = try SsaTensor.empty(&b, Tree.fromComptime(.{ 2, 3 }), typing.Float32);
    const rmem_like = try makeRmemTensorLike(.{ .ssa = ssa }, typing.Float16);
    try std.testing.expectEqualStrings("Float16", rmem_like.dtype.name);
    const frag_like = try makeFragmentLike(.{ .meta = ptr_meta }, null);
    try std.testing.expectEqualStrings("Float16", frag_like.dtype.name);
    _ = try makeFragmentLikeSsa(&b, ssa, typing.Int32);
    _ = try makeRmemTensorLikeSsa(&b, ssa, null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.make_tensor") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.undef") != null);
}

test "tensor_ssa finish: unsigned arithmetic and conversion lower with unsigned ops" {
    var b: mlir.Builder(8192) = .{};
    const z = try b.constantI(0, mlir.Type.i(32));
    const a = try SsaTensor.full(&b, Tree.fromComptime(.{4}), typing.Uint32, .{ .value = z });
    const b2 = try SsaTensor.ones(&b, Tree.fromComptime(.{4}), typing.Uint32);
    _ = try a.div(&b, b2);
    _ = try a.mod(&b, b2);
    _ = try a.lt(&b, b2);
    _ = try a.castTo(&b, typing.Float32);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "arith.divui") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "arith.remui") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "predicate = ult") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "arith.uitofp") != null);
}

test "tensor_ssa finish: reductions support explicit scalar and profile initializers" {
    var b: mlir.Builder(16384) = .{};
    const shape = Tree.fromComptime(.{ 2, 4 });
    const zero = try b.constantF(0.0, mlir.Type.f(32));
    const one = try b.constantF(1.0, mlir.Type.f(32));
    const src = try SsaTensor.full(&b, shape, typing.Float32, .{ .value = one });
    const scalar_init = SsaValue.init(zero, typing.Float32);
    _ = try src.reduceAllWithInit(&b, .add, scalar_init);
    const init_tensor = try SsaTensor.full(&b, Tree.fromComptime(.{2}), typing.Float32, .{ .value = zero });
    const prof = Selector.fromComptime(.{ keep, 0 });
    const reduced = try reduce(&b, src, .max, &prof, init_tensor);
    switch (reduced) {
        .tensor => |t| try std.testing.expectEqualSlices(Scalar, &.{2}, (try t.shape_value.flattenLeaves()).slice()),
        .scalar => return Error.InvalidReductionProfile,
    }
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "vector.multi_reduction") != null);
}

test "tensor_ssa finish: optimized vector gather scatter paths are available" {
    var b: mlir.Builder(16384) = .{};
    const lay = layout.makeLayout(.{ 8, 2 }, .{ 1, 8 });
    const ptr = try runtime.Pointer.init(0xa000, typing.Float32, .gmem, null);
    const meta = try makePointerTensor(ptr, lay);
    const tv = TensorValue.init(meta, mlir.Value.arg(0), "!cute.memref<ptr<f32, gmem, align=4>, layout<shape=(8,2), stride=(1,8)>>");
    const idx_zero = try b.constantI(0, mlir.Type.i(32));
    const index = try SsaTensor.full(&b, Tree.fromComptime(.{ 8, 2 }), typing.Int32, .{ .value = idx_zero });
    const gathered = try emitGatherOptimized(&b, tv, 0, index);
    try emitScatterOptimized(&b, tv, 0, index, gathered);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "vector.gather") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "vector.scatter") != null);
}

test "tensor_ssa finish: public convenience wrappers cover unary/binary/any/all" {
    var b: mlir.Builder(16384) = .{};
    const one = try b.constantI(1, mlir.Type.i(32));
    const a = try SsaTensor.full(&b, Tree.fromComptime(.{2}), typing.Int32, .{ .value = one });
    const b2 = try SsaTensor.ones(&b, Tree.fromComptime(.{2}), typing.Int32);
    _ = try a.add(&b, b2);
    _ = try a.sub(&b, b2);
    _ = try a.mul(&b, b2);
    _ = try a.min(&b, b2);
    _ = try a.max(&b, b2);
    _ = try a.neg(&b);
    _ = try a.abs(&b);
    const flags = try a.gt(&b, b2);
    _ = try any(&b, flags);
    _ = try all(&b, flags);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "arith.addi") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.neg_int") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "vector.reduction.or") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "vector.reduction.and") != null);
}
