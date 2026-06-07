const std = @import("std");
const layout = @import("layout.zig");
const typing = @import("typing.zig");
const mlir = @import("mlir.zig");
const atom = @import("atom.zig");
const tensor = @import("tensor.zig");
const cutlass = @import("cutlass.zig");

pub const Tree = layout.Tree;
pub const Layout = layout.Layout;
pub const TensorMeta = tensor.TensorMeta;
pub const TensorValue = tensor.TensorValue;
pub const SsaValue = tensor.SsaValue;
pub const SsaTensor = tensor.SsaTensor;

pub const Error = tensor.Error || atom.Error || cutlass.Error || error{
    IncompatibleElementType,
    IncompatibleTensorShape,
    IncompatibleMemorySpace,
    InvalidPredicateType,
    InvalidCopyOperand,
    InvalidCopyMode,
    InvalidAsyncCopy,
    InvalidMmaOperand,
    InvalidMmaShape,
    InvalidFragmentShape,
    InvalidAccumulatorType,
    InvalidThreadValueLayout,
    MissingInstructionShape,
    MissingTensorValue,
    UnsupportedCopyDirection,
    UnsupportedMmaDirection,
};

pub const FragmentRole = enum {
    A,
    B,
    C,
    D,

    pub fn atomOperand(self: FragmentRole) Error!atom.Operand {
        return switch (self) {
            .A => .A,
            .B => .B,
            .C, .D => .C,
        };
    }

    pub fn irName(self: FragmentRole) []const u8 {
        return switch (self) {
            .A => "A",
            .B => "B",
            .C => "C",
            .D => "D",
        };
    }
};

pub const CopyLoweringKind = enum {
    atom,
    conditional,
    tiled,
    async,
    tma,
};

pub const CopyLoweringPlan = struct {
    kind: CopyLoweringKind,
    source_space: typing.AddressSpace,
    destination_space: typing.AddressSpace,
    element: typing.Numeric,
    element_count: layout.Unsigned,
    predicated: bool = false,
    vector_bits: u16 = 0,

    pub fn isVectorized(self: CopyLoweringPlan) bool {
        return self.vector_bits != 0 and self.vector_bits > self.element.width;
    }
};

pub const CopyLoweringResult = struct {
    plan: CopyLoweringPlan,
    emitted_predicate: bool = false,
};

pub const TiledCopyPartition = struct {
    source: TensorValue,
    destination: TensorValue,
    thread_id: mlir.Value,
};

pub const MmaLoweringPlan = struct {
    m: layout.Unsigned,
    n: layout.Unsigned,
    k: layout.Unsigned,
    element_a: typing.Numeric,
    element_b: typing.Numeric,
    element_c: typing.Numeric,
    accumulator: typing.Numeric,
    accumulate: bool = true,
};

pub const MmaFragments = struct {
    d: TensorValue,
    a: TensorValue,
    b: TensorValue,
    c: TensorValue,
};

pub fn validateCopy(
    atom_value: atom.CopyAtom,
    src: TensorMeta,
    dst: TensorMeta,
    pred: ?SsaValue,
) Error!CopyLoweringPlan {
    if (!sameNumeric(src.dtype, atom_value.valueType()))
        return Error.IncompatibleElementType;
    if (!sameNumeric(dst.dtype, atom_value.valueType()))
        return Error.IncompatibleElementType;
    if (!src.layout_value.shape.equals(&dst.layout_value.shape))
        return Error.IncompatibleTensorShape;
    if (atom_value.op().source_space) |space| if (src.memspace != space) return Error.IncompatibleMemorySpace;
    if (atom_value.op().destination_space) |space| if (dst.memspace != space) return Error.IncompatibleMemorySpace;
    if (pred) |p| if (p.dtype.kind != .boolean) return Error.InvalidPredicateType;
    const count = try src.layout_value.size();
    return .{
        .kind = if (pred != null) .conditional else .atom,
        .source_space = src.memspace,
        .destination_space = dst.memspace,
        .element = src.dtype,
        .element_count = count,
        .predicated = pred != null,
        .vector_bits = atom_value.op().num_bits_per_copy orelse src.dtype.width,
    };
}

pub fn lowerCopyAtom(
    builder: anytype,
    atom_value: atom.CopyAtom,
    src: TensorValue,
    dst: TensorValue,
    pred: ?SsaValue,
) Error!CopyLoweringResult {
    const plan = try validateCopy(atom_value, src.meta, dst.meta, pred);
    if (pred) |_| {
        const src_op = try tensorOperand(src);
        const dst_op = try tensorOperand(dst);
        const pred_op: ?atom.TensorOperand = if (pred) |p| .{
            .value = .{ .value = p.value },
            .ty = mlir.Type.raw(p.dtype.mlir_type),
            .element = p.dtype,
        } else null;
        try atom.copyAtomCall(builder, atom_value, &.{src_op}, &.{dst_op}, pred_op);
    } else {
        var src_ty_buf: mlir.TextBuffer(512) = .{};
        var dst_ty_buf: mlir.TextBuffer(512) = .{};
        var atom_ty_buf: mlir.TextBuffer(256) = .{};
        try src.meta.cutlassTensorTypeText(&src_ty_buf);
        try dst.meta.cutlassTensorTypeText(&dst_ty_buf);
        try cutlass.writeUniversalCopyAtomType(
            &atom_ty_buf,
            atom_value.valueType().mlir_type,
            plan.vector_bits,
        );
        const atom_handle = try emitMakeAtomWithType(builder, atom_ty_buf.slice());
        try cutlass.emitCopyAtomCall(
            builder,
            atom_handle,
            mlir.Type.raw(atom_ty_buf.slice()),
            src.value,
            dst.value,
            mlir.Type.raw(src_ty_buf.slice()),
            mlir.Type.raw(dst_ty_buf.slice()),
        );
    }
    return .{ .plan = plan, .emitted_predicate = pred != null };
}

pub fn copyTensor(
    builder: anytype,
    src: TensorValue,
    dst: TensorValue,
) Error!CopyLoweringResult {
    try validatePlainCopy(src.meta, dst.meta, null);
    try builder.operationNoResult(.{
        .name = "cute.copy",
        .operands = &.{ .{ .value = src.value }, .{ .value = dst.value } },
        .operand_types = &.{ src.type_(), dst.type_() },
        .result_types = &.{},
    });
    return .{ .plan = .{
        .kind = .atom,
        .source_space = src.meta.memspace,
        .destination_space = dst.meta.memspace,
        .element = src.meta.dtype,
        .element_count = try src.meta.size(),
        .vector_bits = src.meta.dtype.width,
    } };
}

pub fn copyIf(
    builder: anytype,
    src: TensorValue,
    dst: TensorValue,
    pred: SsaValue,
) Error!CopyLoweringResult {
    try validatePlainCopy(src.meta, dst.meta, pred);
    try builder.operationNoResult(.{
        .name = "cute.copy_if",
        .operands = &.{
            .{ .value = src.value },
            .{ .value = dst.value },
            .{ .value = pred.value },
        },
        .operand_types = &.{
            src.type_(),
            dst.type_(),
            mlir.Type.raw(pred.dtype.mlir_type),
        },
        .result_types = &.{},
    });
    return .{ .plan = .{
        .kind = .conditional,
        .source_space = src.meta.memspace,
        .destination_space = dst.meta.memspace,
        .element = src.meta.dtype,
        .element_count = try src.meta.size(),
        .predicated = true,
        .vector_bits = src.meta.dtype.width,
    }, .emitted_predicate = true };
}

pub fn retileCopy(
    builder: anytype,
    tiled: atom.TiledCopy,
    tensor_value: TensorValue,
) Error!TensorValue {
    const out_layout = try tiled.retile(tensor_value.meta.layout_value);
    var out_meta = tensor_value.meta;
    out_meta.layout_value = out_layout;
    return emitTensorTransform(
        builder,
        "cute.tiled.copy.retile",
        tensor_value,
        out_meta,
        &.{},
    );
}

pub fn partitionSrc(
    builder: anytype,
    tiled: atom.TiledCopy,
    src: TensorValue,
    thread_id: mlir.Value,
) Error!TensorValue {
    const thr = try tiled.getSlice(0);
    const out_layout = try thr.partitionS(src.meta.layout_value);
    var out_meta = src.meta;
    out_meta.layout_value = out_layout;
    const tid_ty = mlir.Type.index();
    return emitTensorTransform(
        builder,
        "cute.tiled.copy.partition_S",
        src,
        out_meta,
        &.{.{ .value = thread_id }},
        &.{tid_ty},
    );
}

pub fn partitionDst(
    builder: anytype,
    tiled: atom.TiledCopy,
    dst: TensorValue,
    thread_id: mlir.Value,
) Error!TensorValue {
    const thr = try tiled.getSlice(0);
    const out_layout = try thr.partitionD(dst.meta.layout_value);
    var out_meta = dst.meta;
    out_meta.layout_value = out_layout;
    const tid_ty = mlir.Type.index();
    return emitTensorTransform(
        builder,
        "cute.tiled.copy.partition_D",
        dst,
        out_meta,
        &.{.{ .value = thread_id }},
        &.{tid_ty},
    );
}

pub fn lowerTiledCopy(
    builder: anytype,
    tiled: atom.TiledCopy,
    src: TensorValue,
    dst: TensorValue,
    thread_id: mlir.Value,
    pred: ?SsaValue,
) Error!CopyLoweringResult {
    _ = try validateCopy(tiled.base, src.meta, dst.meta, pred);
    const ps = try partitionSrc(builder, tiled, src, thread_id);
    const pd = try partitionDst(builder, tiled, dst, thread_id);
    var result = try lowerCopyAtom(builder, tiled.base, ps, pd, pred);
    result.plan.kind = .tiled;
    return result;
}

pub fn lowerAsyncCopy(
    builder: anytype,
    atom_value: atom.CopyAtom,
    src: TensorValue,
    dst: TensorValue,
    barrier: ?mlir.Value,
    pred: ?SsaValue,
) Error!CopyLoweringResult {
    const plan = try validateCopy(atom_value, src.meta, dst.meta, pred);
    if (dst.meta.memspace != .smem and dst.meta.memspace != .tmem)
        return Error.InvalidAsyncCopy;
    var operands: [5]mlir.Operand = undefined;
    var types: [5]mlir.Type = undefined;
    var count: usize = 0;
    operands[count] = .{ .value = src.value };
    types[count] = src.type_();
    count += 1;
    operands[count] = .{ .value = dst.value };
    types[count] = dst.type_();
    count += 1;
    if (barrier) |b| {
        operands[count] = .{ .value = b };
        types[count] = mlir.Type.raw("!cute.barrier");
        count += 1;
    }
    if (pred) |p| {
        operands[count] = .{ .value = p.value };
        types[count] = mlir.Type.raw(p.dtype.mlir_type);
        count += 1;
    }
    try builder.operationNoResult(.{
        .name = "cute_nvgpu.copy_async",
        .operands = operands[0..count],
        .attrs = &.{.{ .key = "op", .value = atomNameAttr(atom_value.op().name) }},
        .operand_types = types[0..count],
        .result_types = &.{},
    });
    return .{ .plan = .{
        .kind = .async,
        .source_space = plan.source_space,
        .destination_space = plan.destination_space,
        .element = plan.element,
        .element_count = plan.element_count,
        .predicated = pred != null,
        .vector_bits = plan.vector_bits,
    }, .emitted_predicate = pred != null };
}

pub fn lowerTmaCopy(
    builder: anytype,
    atom_value: atom.CopyAtom,
    src: TensorValue,
    dst: TensorValue,
    descriptor: mlir.Value,
    coords: []const mlir.Value,
    pred: ?SsaValue,
) Error!CopyLoweringResult {
    const plan = try validateCopy(atom_value, src.meta, dst.meta, pred);
    var operands: [18]mlir.Operand = undefined;
    var types: [18]mlir.Type = undefined;
    var count: usize = 0;
    operands[count] = .{ .value = src.value };
    types[count] = src.type_();
    count += 1;
    operands[count] = .{ .value = dst.value };
    types[count] = dst.type_();
    count += 1;
    operands[count] = .{ .value = descriptor };
    types[count] = mlir.Type.raw("!cute.tma_descriptor");
    count += 1;
    for (coords) |coord| {
        if (count >= operands.len) return Error.OutOfCapacity;
        operands[count] = .{ .value = coord };
        types[count] = mlir.Type.index();
        count += 1;
    }
    if (pred) |p| {
        if (count >= operands.len) return Error.OutOfCapacity;
        operands[count] = .{ .value = p.value };
        types[count] = mlir.Type.raw(p.dtype.mlir_type);
        count += 1;
    }
    try builder.operationNoResult(.{
        .name = "cute_nvgpu.tma_copy",
        .operands = operands[0..count],
        .attrs = &.{.{ .key = "op", .value = atomNameAttr(atom_value.op().name) }},
        .operand_types = types[0..count],
        .result_types = &.{},
    });
    return .{ .plan = .{
        .kind = .tma,
        .source_space = plan.source_space,
        .destination_space = plan.destination_space,
        .element = plan.element,
        .element_count = plan.element_count,
        .predicated = pred != null,
        .vector_bits = plan.vector_bits,
    }, .emitted_predicate = pred != null };
}

pub fn validateMma(
    atom_value: atom.MmaAtom,
    d: TensorMeta,
    a: TensorMeta,
    b: TensorMeta,
    c: TensorMeta,
) Error!MmaLoweringPlan {
    const shape = atom_value.op().instruction_shape_mnk orelse atom_value.shapeMnk();
    if (shape.rank() != 3) return Error.InvalidMmaShape;
    const leaves = try shape.flattenLeaves();
    if (leaves.len != 3) return Error.InvalidMmaShape;
    const m: layout.Unsigned = @intCast(leaves.at(0));
    const n: layout.Unsigned = @intCast(leaves.at(1));
    const k: layout.Unsigned = @intCast(leaves.at(2));
    if (!shapeMatches2(a.layout_value.shape, m, k)) return Error.InvalidMmaOperand;
    if (!shapeMatches2(b.layout_value.shape, n, k)) return Error.InvalidMmaOperand;
    if (!shapeMatches2(c.layout_value.shape, m, n)) return Error.InvalidMmaOperand;
    if (!shapeMatches2(d.layout_value.shape, m, n)) return Error.InvalidMmaOperand;
    if (atom_value.op().a_type) |dt| if (!sameNumeric(a.dtype, dt)) return Error.IncompatibleElementType;
    if (atom_value.op().b_type) |dt| if (!sameNumeric(b.dtype, dt)) return Error.IncompatibleElementType;
    if (atom_value.op().c_type) |dt| if (!sameNumeric(c.dtype, dt)) return Error.InvalidAccumulatorType;
    if (!sameNumeric(c.dtype, d.dtype)) return Error.InvalidAccumulatorType;
    return .{
        .m = m,
        .n = n,
        .k = k,
        .element_a = a.dtype,
        .element_b = b.dtype,
        .element_c = c.dtype,
        .accumulator = d.dtype,
        .accumulate = true,
    };
}

pub fn makeFragment(
    builder: anytype,
    atom_value: atom.MmaAtom,
    comptime role: FragmentRole,
    input: TensorValue,
) Error!TensorValue {
    const operand = try role.atomOperand();
    const result_shape = try mmaFragmentShape(atom_value, role);
    var meta = try tensor.makeFragment(input.meta.dtype, result_shape);
    if (role == .C or role == .D) {
        meta.dtype = atom_value.op().c_type orelse input.meta.dtype;
    }
    var type_buf: mlir.TextBuffer(512) = .{};
    try meta.tensorTypeText(&type_buf);
    const result = try atom_value.makeFragmentMlir(
        operand,
        builder,
        .{ .value = input.value },
        input.type_(),
        mlir.Type.raw(type_buf.slice()),
    );
    return TensorValue.initFromMeta(meta, result);
}

pub fn makeFragmentA(
    builder: anytype,
    atom_value: atom.MmaAtom,
    input: TensorValue,
) Error!TensorValue {
    return makeFragment(builder, atom_value, .A, input);
}

pub fn makeFragmentB(
    builder: anytype,
    atom_value: atom.MmaAtom,
    input: TensorValue,
) Error!TensorValue {
    return makeFragment(builder, atom_value, .B, input);
}

pub fn makeFragmentC(
    builder: anytype,
    atom_value: atom.MmaAtom,
    input: TensorValue,
) Error!TensorValue {
    return makeFragment(builder, atom_value, .C, input);
}

pub fn partitionMma(
    builder: anytype,
    tiled: atom.TiledMma,
    comptime role: FragmentRole,
    tensor_value: TensorValue,
    thread_id: mlir.Value,
) Error!TensorValue {
    const operand: atom.Operand = switch (role) {
        .A => .A,
        .B => .B,
        .C, .D => .C,
    };
    var out_meta = tensor_value.meta;
    out_meta.layout_value = switch (role) {
        .A => try tiled.thrfrgLayout(.A, tensor_value.meta.layout_value),
        .B => try tiled.thrfrgLayout(.B, tensor_value.meta.layout_value),
        .C, .D => try tiled.thrfrgLayout(.C, tensor_value.meta.layout_value),
    };
    const result_type = try tensorTypeBuffer(out_meta);
    const value = switch (operand) {
        .A => try tiled.emitPartition(
            .A,
            builder,
            .{ .value = tensor_value.value },
            .{ .value = thread_id },
            tensor_value.type_(),
            mlir.Type.index(),
            mlir.Type.raw(result_type.slice()),
        ),
        .B => try tiled.emitPartition(
            .B,
            builder,
            .{ .value = tensor_value.value },
            .{ .value = thread_id },
            tensor_value.type_(),
            mlir.Type.index(),
            mlir.Type.raw(result_type.slice()),
        ),
        .C => try tiled.emitPartition(
            .C,
            builder,
            .{ .value = tensor_value.value },
            .{ .value = thread_id },
            tensor_value.type_(),
            mlir.Type.index(),
            mlir.Type.raw(result_type.slice()),
        ),
        else => unreachable,
    };
    return TensorValue.initFromMeta(out_meta, value);
}

pub fn partitionMmaA(
    builder: anytype,
    tiled: atom.TiledMma,
    tensor_value: TensorValue,
    thread_id: mlir.Value,
) Error!TensorValue {
    return partitionMma(builder, tiled, .A, tensor_value, thread_id);
}

pub fn partitionMmaB(
    builder: anytype,
    tiled: atom.TiledMma,
    tensor_value: TensorValue,
    thread_id: mlir.Value,
) Error!TensorValue {
    return partitionMma(builder, tiled, .B, tensor_value, thread_id);
}

pub fn partitionMmaC(
    builder: anytype,
    tiled: atom.TiledMma,
    tensor_value: TensorValue,
    thread_id: mlir.Value,
) Error!TensorValue {
    return partitionMma(builder, tiled, .C, tensor_value, thread_id);
}

pub fn lowerMmaAtom(
    builder: anytype,
    atom_value: atom.MmaAtom,
    d: TensorValue,
    a: TensorValue,
    b: TensorValue,
    c: TensorValue,
) Error!MmaLoweringPlan {
    const plan = try validateMma(atom_value, d.meta, a.meta, b.meta, c.meta);
    var d_ty_buf: mlir.TextBuffer(512) = .{};
    var a_ty_buf: mlir.TextBuffer(512) = .{};
    var b_ty_buf: mlir.TextBuffer(512) = .{};
    var c_ty_buf: mlir.TextBuffer(512) = .{};
    var atom_ty_buf: mlir.TextBuffer(256) = .{};
    try d.meta.cutlassTensorTypeText(&d_ty_buf);
    try a.meta.cutlassTensorTypeText(&a_ty_buf);
    try b.meta.cutlassTensorTypeText(&b_ty_buf);
    try c.meta.cutlassTensorTypeText(&c_ty_buf);
    try cutlass.writeUniversalFmaAtomType(
        &atom_ty_buf,
        plan.accumulator.mlir_type,
        @intCast(plan.m),
        @intCast(plan.n),
        @intCast(plan.k),
    );
    const atom_handle = try emitMakeAtomWithType(builder, atom_ty_buf.slice());
    try cutlass.emitMmaAtomCall(
        builder,
        atom_handle,
        mlir.Type.raw(atom_ty_buf.slice()),
        d.value,
        a.value,
        b.value,
        c.value,
        mlir.Type.raw(d_ty_buf.slice()),
        mlir.Type.raw(a_ty_buf.slice()),
        mlir.Type.raw(b_ty_buf.slice()),
        mlir.Type.raw(c_ty_buf.slice()),
    );
    return plan;
}

pub fn lowerTiledMma(
    builder: anytype,
    tiled: atom.TiledMma,
    d: TensorValue,
    a: TensorValue,
    b: TensorValue,
    c: TensorValue,
    thread_id: mlir.Value,
) Error!MmaLoweringPlan {
    const plan = try validateMma(tiled.base, d.meta, a.meta, b.meta, c.meta);
    const pa = try partitionMmaA(builder, tiled, a, thread_id);
    const pb = try partitionMmaB(builder, tiled, b, thread_id);
    const pc = try partitionMmaC(builder, tiled, c, thread_id);
    const pd = try partitionMmaC(builder, tiled, d, thread_id);
    var d_ty_buf: mlir.TextBuffer(512) = .{};
    var a_ty_buf: mlir.TextBuffer(512) = .{};
    var b_ty_buf: mlir.TextBuffer(512) = .{};
    var c_ty_buf: mlir.TextBuffer(512) = .{};
    var atom_ty_buf: mlir.TextBuffer(256) = .{};
    try pd.meta.cutlassTensorTypeText(&d_ty_buf);
    try pa.meta.cutlassTensorTypeText(&a_ty_buf);
    try pb.meta.cutlassTensorTypeText(&b_ty_buf);
    try pc.meta.cutlassTensorTypeText(&c_ty_buf);
    try cutlass.writeUniversalFmaAtomType(
        &atom_ty_buf,
        plan.accumulator.mlir_type,
        @intCast(plan.m),
        @intCast(plan.n),
        @intCast(plan.k),
    );
    const atom_handle = try emitMakeAtomWithType(builder, atom_ty_buf.slice());
    try cutlass.emitMmaAtomCall(
        builder,
        atom_handle,
        mlir.Type.raw(atom_ty_buf.slice()),
        pd.value,
        pa.value,
        pb.value,
        pc.value,
        mlir.Type.raw(d_ty_buf.slice()),
        mlir.Type.raw(a_ty_buf.slice()),
        mlir.Type.raw(b_ty_buf.slice()),
        mlir.Type.raw(c_ty_buf.slice()),
    );
    return plan;
}

pub fn gemmStep(
    builder: anytype,
    tiled: atom.TiledMma,
    d: TensorValue,
    a: TensorValue,
    b: TensorValue,
    c: TensorValue,
    thread_id: mlir.Value,
) Error!MmaLoweringPlan {
    return lowerTiledMma(builder, tiled, d, a, b, c, thread_id);
}

pub fn lowerMmaAccumulate(
    builder: anytype,
    atom_value: atom.MmaAtom,
    acc: SsaTensor,
    a: SsaTensor,
    b: SsaTensor,
) Error!SsaTensor {
    const expected = try mmaFragmentShape(atom_value, .C);
    if (!acc.shape_value.equals(&expected)) return Error.InvalidFragmentShape;
    const shape_a = try mmaFragmentShape(atom_value, .A);
    const shape_b = try mmaFragmentShape(atom_value, .B);
    if (!a.shape_value.equals(&shape_a) or !b.shape_value.equals(&shape_b))
        return Error.InvalidFragmentShape;
    var acc_ty: mlir.TextBuffer(256) = .{};
    var a_ty: mlir.TextBuffer(256) = .{};
    var b_ty: mlir.TextBuffer(256) = .{};
    try acc.vectorType(&acc_ty);
    try a.vectorType(&a_ty);
    try b.vectorType(&b_ty);
    const result = try builder.genericOp(
        "cute.mma.ssa",
        &.{ .{ .value = acc.value }, .{ .value = a.value }, .{ .value = b.value } },
        &.{.{ .key = "op", .value = atomNameAttr(atom_value.op().name) }},
        &.{
            mlir.Type.raw(acc_ty.slice()),
            mlir.Type.raw(a_ty.slice()),
            mlir.Type.raw(b_ty.slice()),
        },
        &.{mlir.Type.raw(acc_ty.slice())},
    );
    return SsaTensor.init(result, acc.shape_value, acc.dtype);
}

fn emitMakeAtomWithType(builder: anytype, atom_ty: []const u8) Error!mlir.Value {
    const result = builder.freshValue();
    try builder.writeResultPrefixFor(&.{mlir.Type.raw(atom_ty)}, result.id);
    try builder.append("cute.make_atom() : () -> ");
    try builder.append(atom_ty);
    try builder.newline();
    return result;
}

fn validatePlainCopy(src: TensorMeta, dst: TensorMeta, pred: ?SsaValue) Error!void {
    if (!sameNumeric(src.dtype, dst.dtype)) return Error.IncompatibleElementType;
    if (!src.layout_value.shape.equals(&dst.layout_value.shape))
        return Error.IncompatibleTensorShape;
    if (pred) |p| if (p.dtype.kind != .boolean) return Error.InvalidPredicateType;
}

fn tensorOperand(t: TensorValue) Error!atom.TensorOperand {
    return .{
        .value = .{ .value = t.value },
        .ty = t.type_(),
        .element = t.meta.dtype,
        .v_rank = t.meta.rank(),
    };
}

fn emitTensorTransform(
    builder: anytype,
    op_name: []const u8,
    input: TensorValue,
    out_meta: TensorMeta,
    extra_operands: []const mlir.Operand,
    extra_types: []const mlir.Type,
) Error!TensorValue {
    if (extra_operands.len != extra_types.len) return Error.RankMismatch;
    var operands: [8]mlir.Operand = undefined;
    var types: [8]mlir.Type = undefined;
    operands[0] = .{ .value = input.value };
    types[0] = input.type_();
    for (extra_operands, 0..) |operand, i| {
        operands[i + 1] = operand;
        types[i + 1] = extra_types[i];
    }
    var type_buf = try tensorTypeBuffer(out_meta);
    const result = try builder.genericOp(
        op_name,
        operands[0 .. 1 + extra_operands.len],
        &.{},
        types[0 .. 1 + extra_types.len],
        &.{mlir.Type.raw(type_buf.slice())},
    );
    return TensorValue.initFromMeta(out_meta, result);
}

fn tensorTypeBuffer(meta: TensorMeta) Error!mlir.TextBuffer(512) {
    var type_buf: mlir.TextBuffer(512) = .{};
    try meta.tensorTypeText(&type_buf);
    return type_buf;
}

fn sameNumeric(a: typing.Numeric, b: typing.Numeric) bool {
    return a.width == b.width and a.kind == b.kind and std.mem.eql(u8, a.mlir_type, b.mlir_type);
}

fn shapeMatches2(shape: Tree, d0: layout.Unsigned, d1: layout.Unsigned) bool {
    const flat = shape.flattenLeaves() catch return false;
    if (flat.len == 2) {
        return flat.at(0) == @as(layout.Scalar, @intCast(d0)) and flat.at(1) == @as(layout.Scalar, @intCast(d1));
    }
    if (flat.len == 1 and d0 == 1 and d1 == 1) {
        return flat.at(0) == 1;
    }
    return false;
}

fn mmaFragmentShape(atom_value: atom.MmaAtom, role: FragmentRole) Error!Tree {
    const shape = atom_value.op().instruction_shape_mnk orelse atom_value.shapeMnk();
    const leaves = try shape.flattenLeaves();
    if (leaves.len != 3) return Error.InvalidMmaShape;
    return switch (role) {
        .A => Tree.initTuple(&.{
            try Tree.initLeaf(leaves.at(0)),
            try Tree.initLeaf(leaves.at(2)),
        }),
        .B => Tree.initTuple(&.{
            try Tree.initLeaf(leaves.at(1)),
            try Tree.initLeaf(leaves.at(2)),
        }),
        .C, .D => Tree.initTuple(&.{
            try Tree.initLeaf(leaves.at(0)),
            try Tree.initLeaf(leaves.at(1)),
        }),
    };
}

fn atomNameAttr(name: []const u8) []const u8 {
    // The textual builder does not own heap memory.  Common names in tests and
    // descriptors are static, so return a safe quoted token for known/simple
    // names and a generic quoted fallback for dynamic names.
    if (std.mem.eql(u8, name, "copy")) return "\"copy\"";
    if (std.mem.eql(u8, name, "mma")) return "\"mma\"";
    if (std.mem.eql(u8, name, "cp.async")) return "\"cp.async\"";
    if (std.mem.eql(u8, name, "tma.copy")) return "\"tma.copy\"";
    return "\"custom\"";
}

fn makeTensorValue(meta: TensorMeta, value_name: []const u8) Error!TensorValue {
    return TensorValue.initFromMeta(meta, mlir.Value.named(value_name));
}

fn genericCopyAtom(
    dtype: typing.Numeric,
    src_space: typing.AddressSpace,
    dst_space: typing.AddressSpace,
) Error!atom.CopyAtom {
    const thr = layout.makeCompactLayout(.{4});
    const tv = layout.makeCompactLayout(.{ 4, 1 });
    var tr: atom.Trait = .{ .name = "copy", .thr_id = thr };
    tr = tr.withCopyLayouts(tv, tv);
    return atom.makeCopyAtom(
        atom.OpDescriptor.copyTyped("copy", "generic", "unit", dtype, src_space, dst_space, dtype.width, &.{}),
        tr,
    );
}

fn genericMmaAtom() Error!atom.MmaAtom {
    const thr = layout.makeCompactLayout(.{32});
    const tv = layout.makeCompactLayout(.{ 32, 1 });
    var tr: atom.Trait = .{
        .name = "mma",
        .thr_id = thr,
        .shape_mnk = Tree.fromComptime(.{ 16, 8, 8 }),
    };
    tr = tr.withMmaLayouts(tv, tv, tv);
    return atom.makeMmaAtom(
        atom.OpDescriptor.mmaTyped("mma", "generic", "unit", Tree.fromComptime(.{
            16,
            8,
            8,
        }), typing.Float16, typing.Float16, typing.Float32, &.{.accumulate}),
        tr,
    );
}

test "copy_mma: copy atom validates tensor metadata and emits copy call" {
    const shape = Tree.fromComptime(.{ 4, 4 });
    const l = try Layout.makeCompact(shape);
    const src_meta = try TensorMeta.init(
        .{ .pointer = @import("runtime.zig").Pointer.nullptr(typing.Float32, .gmem) },
        l,
        typing.Float32,
        .gmem,
    );
    const dst_meta = try TensorMeta.init(
        .{ .pointer = @import("runtime.zig").Pointer.nullptr(typing.Float32, .smem) },
        l,
        typing.Float32,
        .smem,
    );
    const src = try makeTensorValue(src_meta, "%src");
    const dst = try makeTensorValue(dst_meta, "%dst");
    const copy_atom = try genericCopyAtom(typing.Float32, .gmem, .smem);
    var b: mlir.Builder(2048) = .{};
    const result = try lowerCopyAtom(&b, copy_atom, src, dst, null);
    try std.testing.expectEqual(CopyLoweringKind.atom, result.plan.kind);
    try std.testing.expectEqual(@as(layout.Unsigned, 16), result.plan.element_count);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.copy_atom_call") != null);
}

test "copy_mma: copy rejects element, shape, memory and predicate mismatches" {
    const shape = Tree.fromComptime(.{ 4, 4 });
    const l = try Layout.makeCompact(shape);
    const copy_atom = try genericCopyAtom(typing.Float32, .gmem, .smem);
    const src = try TensorMeta.init(
        .{ .pointer = @import("runtime.zig").Pointer.nullptr(typing.Float32, .gmem) },
        l,
        typing.Float32,
        .gmem,
    );
    const dst_bad_type = try TensorMeta.init(
        .{ .pointer = @import("runtime.zig").Pointer.nullptr(typing.Float16, .smem) },
        l,
        typing.Float16,
        .smem,
    );
    try std.testing.expectError(
        Error.IncompatibleElementType,
        validateCopy(copy_atom, src, dst_bad_type, null),
    );
    const dst_bad_space = try TensorMeta.init(
        .{ .pointer = @import("runtime.zig").Pointer.nullptr(typing.Float32, .gmem) },
        l,
        typing.Float32,
        .gmem,
    );
    try std.testing.expectError(
        Error.IncompatibleMemorySpace,
        validateCopy(copy_atom, src, dst_bad_space, null),
    );
    const bad_pred = SsaValue.init(mlir.Value.named("%p"), typing.Int32);
    const dst = try TensorMeta.init(
        .{ .pointer = @import("runtime.zig").Pointer.nullptr(typing.Float32, .smem) },
        l,
        typing.Float32,
        .smem,
    );
    try std.testing.expectError(
        Error.InvalidPredicateType,
        validateCopy(copy_atom, src, dst, bad_pred),
    );
}

test "copy_mma: tiled copy partitions tensors before atom call" {
    const shape = Tree.fromComptime(.{ 8, 4 });
    const l = try Layout.makeCompact(shape);
    const src_meta = try TensorMeta.init(
        .{ .pointer = @import("runtime.zig").Pointer.nullptr(typing.Int32, .gmem) },
        l,
        typing.Int32,
        .gmem,
    );
    const dst_meta = try TensorMeta.init(
        .{ .pointer = @import("runtime.zig").Pointer.nullptr(typing.Int32, .smem) },
        l,
        typing.Int32,
        .smem,
    );
    const src = try makeTensorValue(src_meta, "%src");
    const dst = try makeTensorValue(dst_meta, "%dst");
    const copy_atom = try genericCopyAtom(typing.Int32, .gmem, .smem);
    const tv = layout.makeCompactLayout(.{ 4, 1 });
    const tiled = try atom.makeTiledCopy(copy_atom, tv, Tree.fromComptime(.{4}));
    var b: mlir.Builder(4096) = .{};
    const tid = try b.constantIndex(0);
    const result = try lowerTiledCopy(&b, tiled, src, dst, tid, null);
    try std.testing.expectEqual(CopyLoweringKind.tiled, result.plan.kind);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.tiled.copy.partition_S") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.tiled.copy.partition_D") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.copy_atom_call") != null);
}

test "copy_mma: async and tma copy emit distinct nvgpu hooks" {
    const shape = Tree.fromComptime(.{ 4, 4 });
    const l = try Layout.makeCompact(shape);
    const src_meta = try TensorMeta.init(
        .{ .pointer = @import("runtime.zig").Pointer.nullptr(typing.Float32, .gmem) },
        l,
        typing.Float32,
        .gmem,
    );
    const dst_meta = try TensorMeta.init(
        .{ .pointer = @import("runtime.zig").Pointer.nullptr(typing.Float32, .smem) },
        l,
        typing.Float32,
        .smem,
    );
    const src = try makeTensorValue(src_meta, "%src");
    const dst = try makeTensorValue(dst_meta, "%dst");
    const copy_atom = try genericCopyAtom(typing.Float32, .gmem, .smem);
    var b: mlir.Builder(4096) = .{};
    _ = try lowerAsyncCopy(&b, copy_atom, src, dst, null, null);
    _ = try lowerTmaCopy(
        &b,
        copy_atom,
        src,
        dst,
        mlir.Value.named("%desc"),
        &.{mlir.Value.named("%i")},
        null,
    );
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute_nvgpu.copy_async") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute_nvgpu.tma_copy") != null);
}

test "copy_mma: mma validates fragment shapes and emits call" {
    const mma = try genericMmaAtom();
    const la = try Layout.makeCompact(Tree.fromComptime(.{ 16, 8 }));
    const lb = try Layout.makeCompact(Tree.fromComptime(.{ 8, 8 }));
    const lc = try Layout.makeCompact(Tree.fromComptime(.{ 16, 8 }));
    const a_meta = try TensorMeta.init(
        .{ .fragment = {} },
        la,
        typing.Float16,
        .generic,
    );
    const b_meta = try TensorMeta.init(
        .{ .fragment = {} },
        lb,
        typing.Float16,
        .generic,
    );
    const c_meta = try TensorMeta.init(
        .{ .fragment = {} },
        lc,
        typing.Float32,
        .generic,
    );
    const d_meta = try TensorMeta.init(
        .{ .fragment = {} },
        lc,
        typing.Float32,
        .generic,
    );
    const a = try makeTensorValue(a_meta, "%a");
    const bval = try makeTensorValue(b_meta, "%b");
    const c = try makeTensorValue(c_meta, "%c");
    const d = try makeTensorValue(d_meta, "%d");
    var builder: mlir.Builder(2048) = .{};
    const plan = try lowerMmaAtom(&builder, mma, d, a, bval, c);
    try std.testing.expectEqual(@as(layout.Unsigned, 16), plan.m);
    try std.testing.expectEqual(@as(layout.Unsigned, 8), plan.n);
    try std.testing.expect(std.mem.indexOf(u8, builder.slice(), "cute.mma_atom_call") != null);
    const wrong_a_meta = try TensorMeta.init(
        .{ .fragment = {} },
        lb,
        typing.Float16,
        .generic,
    );
    try std.testing.expectError(
        Error.InvalidMmaOperand,
        validateMma(mma, d_meta, wrong_a_meta, b_meta, c_meta),
    );
}

test "copy_mma: tiled mma partitions A/B/C/D before lowering" {
    const mma = try genericMmaAtom();
    const tiled = try atom.makeTiledMma(
        mma,
        layout.makeCompactLayout(.{ 1, 1, 1 }),
        null,
    );
    const la = try Layout.makeCompact(Tree.fromComptime(.{ 16, 8 }));
    const lb = try Layout.makeCompact(Tree.fromComptime(.{ 8, 8 }));
    const lc = try Layout.makeCompact(Tree.fromComptime(.{ 16, 8 }));
    const a = try makeTensorValue(
        try TensorMeta.init(.{ .fragment = {} }, la, typing.Float16, .generic),
        "%a",
    );
    const b = try makeTensorValue(
        try TensorMeta.init(.{ .fragment = {} }, lb, typing.Float16, .generic),
        "%b",
    );
    const c = try makeTensorValue(
        try TensorMeta.init(.{ .fragment = {} }, lc, typing.Float32, .generic),
        "%c",
    );
    const d = try makeTensorValue(
        try TensorMeta.init(.{ .fragment = {} }, lc, typing.Float32, .generic),
        "%d",
    );
    var builder: mlir.Builder(8192) = .{};
    const tid = try builder.constantIndex(0);
    _ = try lowerTiledMma(&builder, tiled, d, a, b, c, tid);
    try std.testing.expect(std.mem.indexOf(u8, builder.slice(), "cute.tiled.mma.partition") != null);
    try std.testing.expect(std.mem.indexOf(u8, builder.slice(), "cute.mma_atom_call") != null);
}

test "copy_mma: SSA mma hook checks fragment shapes" {
    const mma = try genericMmaAtom();
    var builder: mlir.Builder(4096) = .{};
    const acc = try SsaTensor.empty(
        &builder,
        Tree.fromComptime(.{ 16, 8 }),
        typing.Float32,
    );
    const a = try SsaTensor.empty(
        &builder,
        Tree.fromComptime(.{ 16, 8 }),
        typing.Float16,
    );
    const b = try SsaTensor.empty(
        &builder,
        Tree.fromComptime(.{ 8, 8 }),
        typing.Float16,
    );
    const out = try lowerMmaAccumulate(&builder, mma, acc, a, b);
    try std.testing.expect(out.shape_value.equals(&Tree.fromComptime(.{ 16, 8 })));
    try std.testing.expect(std.mem.indexOf(u8, builder.slice(), "cute.mma.ssa") != null);
}
