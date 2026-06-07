const std = @import("std");
const layout = @import("layout.zig");
const layout_algebra = @import("layout_algebra.zig");
const typing = @import("typing.zig");
const mlir = @import("mlir_text.zig");

pub const Error = layout.Error || mlir.Error || layout_algebra.Error || error{
    WrongAtomKind,
    MissingTraitLayout,
    InvalidThreadIndex,
    InvalidModeIndex,
    InvalidOperand,
    InvalidOperandCount,
    TypeWidthMismatch,
    UnsupportedRank,
    UnsupportedField,
    MissingRuntimeField,
    InvalidCopyBits,
    InvalidAtomLayout,
    InvalidTiler,
    UnsupportedOperation,
};

pub const OpKind = enum {
    mma,
    copy,
};

pub const Operand = enum {
    A,
    B,
    C,
    S,
    D,

    pub fn irName(self: Operand) []const u8 {
        return switch (self) {
            .A => "A",
            .B => "B",
            .C => "C",
            .S => "S",
            .D => "D",
        };
    }
};

pub const RuntimeField = enum {
    accumulate,
    negate_a,
    negate_b,
    sfa,
    sfb,
    cache_policy,
    tma_barrier,
    multicast_mask,
    byte_mask,

    pub fn irName(self: RuntimeField) []const u8 {
        return switch (self) {
            .accumulate => "accum_c",
            .negate_a => "neg_a",
            .negate_b => "neg_b",
            .sfa => "sf_a",
            .sfb => "sf_b",
            .cache_policy => "cache_policy",
            .tma_barrier => "tma_barrier",
            .multicast_mask => "multicast_mask",
            .byte_mask => "byte_mask",
        };
    }

    pub fn fromIrName(name: []const u8) ?RuntimeField {
        inline for (@typeInfo(RuntimeField).@"enum".fields) |field_info| {
            const f: RuntimeField = @enumFromInt(field_info.value);
            if (std.mem.eql(u8, f.irName(), name)) return f;
        }
        return null;
    }
};

pub const RuntimeValue = union(enum) {
    bool: bool,
    i64: i64,
    u64: u64,
    symbol: []const u8,

    pub fn eql(a: RuntimeValue, b: RuntimeValue) bool {
        if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
        return switch (a) {
            .bool => |x| x == b.bool,
            .i64 => |x| x == b.i64,
            .u64 => |x| x == b.u64,
            .symbol => |x| std.mem.eql(u8, x, b.symbol),
        };
    }

    pub fn writeMlirAttr(self: RuntimeValue, out: anytype) Error!void {
        switch (self) {
            .bool => |v| try out.append(if (v) "true" else "false"),
            .i64 => |v| try out.appendSigned(v),
            .u64 => |v| try out.appendUnsigned(@intCast(v)),
            .symbol => |v| try out.append(v),
        }
    }
};

pub const RuntimeEntry = struct {
    field: RuntimeField,
    value: RuntimeValue,
};

pub const RuntimeState = layout.BoundedList(RuntimeEntry, 24);

pub const OpDescriptor = struct {
    name: []const u8,
    kind: OpKind,
    arch: []const u8 = "generic",
    family: []const u8 = "generic",
    instruction_shape_mnk: ?layout.Tree = null,
    a_type: ?typing.Numeric = null,
    b_type: ?typing.Numeric = null,
    c_type: ?typing.Numeric = null,
    value_type: ?typing.Numeric = null,
    num_bits_per_copy: ?u16 = null,
    source_space: ?typing.AddressSpace = null,
    destination_space: ?typing.AddressSpace = null,
    allowed_fields: []const RuntimeField = &.{},
    requires_runtime_unpack: bool = false,

    pub fn mma(name: []const u8, arch: []const u8) OpDescriptor {
        return .{ .name = name, .kind = .mma, .arch = arch };
    }

    pub fn mmaTyped(
        name: []const u8,
        arch: []const u8,
        family: []const u8,
        shape_mnk: layout.Tree,
        a: typing.Numeric,
        b: typing.Numeric,
        c: typing.Numeric,
        fields: []const RuntimeField,
    ) OpDescriptor {
        return .{
            .name = name,
            .kind = .mma,
            .arch = arch,
            .family = family,
            .instruction_shape_mnk = shape_mnk,
            .a_type = a,
            .b_type = b,
            .c_type = c,
            .allowed_fields = fields,
        };
    }

    pub fn copy(name: []const u8, arch: []const u8, value_type: typing.Numeric) OpDescriptor {
        return .{ .name = name, .kind = .copy, .arch = arch, .value_type = value_type };
    }

    pub fn copyTyped(
        name: []const u8,
        arch: []const u8,
        family: []const u8,
        value_type: typing.Numeric,
        source: typing.AddressSpace,
        destination: typing.AddressSpace,
        bits_per_copy: u16,
        fields: []const RuntimeField,
    ) OpDescriptor {
        return .{
            .name = name,
            .kind = .copy,
            .arch = arch,
            .family = family,
            .value_type = value_type,
            .source_space = source,
            .destination_space = destination,
            .num_bits_per_copy = bits_per_copy,
            .allowed_fields = fields,
        };
    }

    pub fn validate(self: OpDescriptor) Error!void {
        switch (self.kind) {
            .mma => {
                if (self.instruction_shape_mnk) |s| {
                    if (s.rank() != 3) return Error.InvalidAtomLayout;
                    try s.assertPositive();
                }
            },
            .copy => {
                if (self.num_bits_per_copy) |bits| {
                    if (bits == 0) return Error.InvalidCopyBits;
                }
            },
        }
    }
};

pub const Trait = struct {
    name: []const u8,
    thr_id: layout.Layout,
    shape_mnk: ?layout.Tree = null,
    tv_layout_a: ?layout.Layout = null,
    tv_layout_b: ?layout.Layout = null,
    tv_layout_c: ?layout.Layout = null,
    layout_src_tv: ?layout.Layout = null,
    layout_dst_tv: ?layout.Layout = null,
    admissible_fields: []const RuntimeField = &.{},
    state: RuntimeState = .{},
    type_name: []const u8 = "!cute.atom_trait",

    pub fn withMmaLayouts(self: Trait, a: layout.Layout, b: layout.Layout, c: layout.Layout) Trait {
        var out = self;
        out.tv_layout_a = a;
        out.tv_layout_b = b;
        out.tv_layout_c = c;
        return out;
    }

    pub fn withCopyLayouts(self: Trait, src: layout.Layout, dst: layout.Layout) Trait {
        var out = self;
        out.layout_src_tv = src;
        out.layout_dst_tv = dst;
        return out;
    }

    pub fn withFields(self: Trait, fields: []const RuntimeField) Trait {
        var out = self;
        out.admissible_fields = fields;
        return out;
    }

    pub fn allows(self: *const Trait, field: RuntimeField) bool {
        for (self.admissible_fields) |f| if (f == field) return true;
        return false;
    }

    pub fn set(self: *Trait, field: RuntimeField, value: RuntimeValue) Error!void {
        if (!self.allows(field)) return Error.UnsupportedField;
        for (self.state.mutableSlice()) |*entry| {
            if (entry.field == field) {
                entry.value = value;
                return;
            }
        }
        try self.state.append(.{ .field = field, .value = value });
    }

    pub fn get(self: *const Trait, field: RuntimeField) Error!RuntimeValue {
        if (!self.allows(field)) return Error.UnsupportedField;
        for (self.state.slice()) |entry| {
            if (entry.field == field) return entry.value;
        }
        return Error.MissingRuntimeField;
    }

    pub fn withRuntimeField(self: Trait, field: RuntimeField, value: RuntimeValue) Error!Trait {
        var out = self;
        try out.set(field, value);
        return out;
    }

    pub fn writeMlirType(self: *const Trait, out: anytype) Error!void {
        try out.append(self.type_name);
        try out.append("<");
        try out.append(self.name);
        try out.append(">");
    }

    pub fn writeRuntimeAttrs(self: *const Trait, out: anytype) Error!void {
        try out.append("{");
        for (self.state.slice(), 0..) |entry, i| {
            if (i != 0) try out.append(", ");
            try out.append(entry.field.irName());
            try out.append(" = ");
            try entry.value.writeMlirAttr(out);
        }
        try out.append("}");
    }
};

pub const Atom = struct {
    op: OpDescriptor,
    trait: Trait,

    pub fn kind(self: Atom) OpKind {
        return self.op.kind;
    }

    pub fn set(self: *Atom, field: RuntimeField, value: RuntimeValue) Error!void {
        try self.trait.set(field, value);
    }

    pub fn get(self: *const Atom, field: RuntimeField) Error!RuntimeValue {
        return self.trait.get(field);
    }

    pub fn withRuntimeField(self: Atom, field: RuntimeField, value: RuntimeValue) Error!Atom {
        var out = self;
        try out.set(field, value);
        return out;
    }

    pub fn writeMlirType(self: Atom, out: anytype) Error!void {
        try out.append("!cute.atom<");
        try out.append(@tagName(self.op.kind));
        try out.append(", ");
        try out.append(self.op.arch);
        try out.append(", ");
        try out.append(self.op.family);
        try out.append(", ");
        try out.append(self.op.name);
        try out.append(", trait=");
        try out.append(self.trait.name);
        try out.append(">");
    }
};

pub const MmaAtom = struct {
    atom: Atom,

    pub fn init(desc: OpDescriptor, tr: Trait) Error!MmaAtom {
        if (desc.kind != .mma) return Error.WrongAtomKind;
        try desc.validate();
        if (tr.shape_mnk == null or tr.tv_layout_a == null or tr.tv_layout_b == null or tr.tv_layout_c == null) return Error.MissingTraitLayout;
        if (tr.shape_mnk.?.rank() != 3) return Error.InvalidAtomLayout;
        return .{ .atom = .{ .op = desc, .trait = tr } };
    }

    pub fn op(self: MmaAtom) OpDescriptor {
        return self.atom.op;
    }

    pub fn trait(self: MmaAtom) Trait {
        return self.atom.trait;
    }

    pub fn set(self: *MmaAtom, field: RuntimeField, value: RuntimeValue) Error!void {
        try self.atom.set(field, value);
    }

    pub fn get(self: *const MmaAtom, field: RuntimeField) Error!RuntimeValue {
        return self.atom.get(field);
    }

    pub fn thrId(self: MmaAtom) layout.Layout {
        return self.atom.trait.thr_id;
    }

    pub fn shapeMnk(self: MmaAtom) layout.Tree {
        return self.atom.trait.shape_mnk.?;
    }

    pub fn tvLayoutA(self: MmaAtom) layout.Layout {
        return self.atom.trait.tv_layout_a.?;
    }

    pub fn tvLayoutB(self: MmaAtom) layout.Layout {
        return self.atom.trait.tv_layout_b.?;
    }

    pub fn tvLayoutC(self: MmaAtom) layout.Layout {
        return self.atom.trait.tv_layout_c.?;
    }

    pub fn makeFragmentMlir(self: MmaAtom, comptime operand: Operand, builder: anytype, input: mlir.Operand, input_type: mlir.Type, result_type: mlir.Type) Error!mlir.Value {
        if (operand != .A and operand != .B and operand != .C) return Error.InvalidOperand;
        _ = self;
        return try builder.genericOp("cute.mma.make_fragment", &.{input}, &.{mlir.Attribute.str("operand", operand.irName())}, &.{input_type}, &.{result_type});
    }
};

pub const CopyAtom = struct {
    atom: Atom,

    pub fn init(desc: OpDescriptor, tr: Trait) Error!CopyAtom {
        if (desc.kind != .copy) return Error.WrongAtomKind;
        try desc.validate();
        if (tr.layout_src_tv == null or tr.layout_dst_tv == null) return Error.MissingTraitLayout;
        return .{ .atom = .{ .op = desc, .trait = tr } };
    }

    pub fn op(self: CopyAtom) OpDescriptor {
        return self.atom.op;
    }

    pub fn trait(self: CopyAtom) Trait {
        return self.atom.trait;
    }

    pub fn set(self: *CopyAtom, field: RuntimeField, value: RuntimeValue) Error!void {
        try self.atom.set(field, value);
    }

    pub fn get(self: *const CopyAtom, field: RuntimeField) Error!RuntimeValue {
        return self.atom.get(field);
    }

    pub fn valueType(self: CopyAtom) typing.Numeric {
        return self.atom.op.value_type orelse typing.Int8;
    }

    pub fn thrId(self: CopyAtom) layout.Layout {
        return self.atom.trait.thr_id;
    }

    pub fn layoutSrcTv(self: CopyAtom) layout.Layout {
        return self.atom.trait.layout_src_tv.?;
    }

    pub fn layoutDstTv(self: CopyAtom) layout.Layout {
        return self.atom.trait.layout_dst_tv.?;
    }
};

pub const TiledMma = struct {
    base: MmaAtom,
    atom_layout_mnk: layout.Layout,
    permutation_mnk: ?layout.Tree = null,
    thr_layout_vmnk: layout.Layout,
    tv_layout_a_tiled: layout.Layout,
    tv_layout_b_tiled: layout.Layout,
    tv_layout_c_tiled: layout.Layout,

    pub fn init(base: MmaAtom, atom_layout_mnk: layout.Layout, permutation_mnk: ?layout.Tree) Error!TiledMma {
        if (atom_layout_mnk.rank() != 3) return Error.InvalidAtomLayout;
        const atom_shape = base.shapeMnk();
        const atom_layout_shape = atom_layout_mnk.shape;
        const vmnk_shape = try prependTree(&atom_layout_shape, layout.Tree.fromComptime(1));
        const thr_layout_vmnk = try layout.Layout.makeCompact(vmnk_shape);

        // Textual zero-dependency model: tiled TV layouts preserve the atom TV layout and extend the
        // value mode by the static product of the atom tiler for the participating MNK modes.
        const a_tiler = try layout.Tree.initTuple(&.{ try modeTile(&atom_shape, &atom_layout_shape, 0), try modeTile(&atom_shape, &atom_layout_shape, 2) });
        const b_tiler = try layout.Tree.initTuple(&.{ try modeTile(&atom_shape, &atom_layout_shape, 1), try modeTile(&atom_shape, &atom_layout_shape, 2) });
        const c_tiler = try layout.Tree.initTuple(&.{ try modeTile(&atom_shape, &atom_layout_shape, 0), try modeTile(&atom_shape, &atom_layout_shape, 1) });

        return .{
            .base = base,
            .atom_layout_mnk = atom_layout_mnk,
            .permutation_mnk = permutation_mnk,
            .thr_layout_vmnk = thr_layout_vmnk,
            .tv_layout_a_tiled = try retileTv(base.tvLayoutA(), a_tiler),
            .tv_layout_b_tiled = try retileTv(base.tvLayoutB(), b_tiler),
            .tv_layout_c_tiled = try retileTv(base.tvLayoutC(), c_tiler),
        };
    }

    pub fn size(self: TiledMma) Error!layout.Unsigned {
        return self.thr_layout_vmnk.size();
    }

    pub fn getTileSize(self: TiledMma, mode_idx: usize) Error!layout.Unsigned {
        if (mode_idx >= 3) return Error.InvalidModeIndex;
        if (self.permutation_mnk) |perm| {
            const m = try perm.topMode(mode_idx);
            if (!(m.rank() == 0 and (try m.product()) == 1)) return m.product();
        }
        return try (try modeTile(&self.base.shapeMnk(), &self.atom_layout_mnk.shape, mode_idx)).product();
    }

    pub fn getSlice(self: TiledMma, thread_index: layout.Scalar) Error!ThrMma {
        const n = try self.size();
        if (thread_index < 0 or @as(layout.Unsigned, @intCast(thread_index)) >= n) return Error.InvalidThreadIndex;
        return .{ .base = self, .thr_idx = thread_index };
    }

    pub fn partitionShape(self: TiledMma, operand: Operand, shape: layout.Tree) Error!layout.Tree {
        const m = try self.getTileSize(0);
        const n = try self.getTileSize(1);
        const k = try self.getTileSize(2);
        return switch (operand) {
            .A => try divideShape2(shape, m, k),
            .B => try divideShape2(shape, n, k),
            .C => try divideShape2(shape, m, n),
            else => Error.InvalidOperand,
        };
    }

    pub fn partitionShapeA(self: TiledMma, shape_mk: layout.Tree) Error!layout.Tree {
        return self.partitionShape(.A, shape_mk);
    }
    pub fn partitionShapeB(self: TiledMma, shape_nk: layout.Tree) Error!layout.Tree {
        return self.partitionShape(.B, shape_nk);
    }
    pub fn partitionShapeC(self: TiledMma, shape_mn: layout.Tree) Error!layout.Tree {
        return self.partitionShape(.C, shape_mn);
    }

    pub fn thrfrgLayout(self: TiledMma, operand: Operand, input: layout.Layout) Error!layout.Layout {
        const part_shape = try self.partitionShape(operand, input.shape);
        return layout.Layout.makeCompact(part_shape);
    }

    pub fn emitPartition(self: TiledMma, comptime operand: Operand, builder: anytype, tensor: mlir.Operand, thr_idx: mlir.Operand, tensor_type: mlir.Type, idx_type: mlir.Type, result_type: mlir.Type) Error!mlir.Value {
        _ = self;
        if (operand != .A and operand != .B and operand != .C) return Error.InvalidOperand;
        return try builder.genericOp("cute.tiled.mma.partition", &.{ tensor, thr_idx }, &.{mlir.Attribute.str("operand", operand.irName())}, &.{ tensor_type, idx_type }, &.{result_type});
    }
};

pub const ThrMma = struct {
    base: TiledMma,
    thr_idx: layout.Scalar,

    pub fn partitionA(self: ThrMma, input_mk: layout.Layout) Error!layout.Layout {
        return self.base.thrfrgLayout(.A, input_mk);
    }
    pub fn partitionB(self: ThrMma, input_nk: layout.Layout) Error!layout.Layout {
        return self.base.thrfrgLayout(.B, input_nk);
    }
    pub fn partitionC(self: ThrMma, input_mn: layout.Layout) Error!layout.Layout {
        return self.base.thrfrgLayout(.C, input_mn);
    }
};

pub const TiledCopy = struct {
    base: CopyAtom,
    layout_tv_tiled: layout.Layout,
    tiler_mn: layout.Tree,
    layout_src_tv_tiled: layout.Layout,
    layout_dst_tv_tiled: layout.Layout,

    pub fn init(base: CopyAtom, layout_tv_tiled: layout.Layout, tiler_mn: layout.Tree) Error!TiledCopy {
        if (layout_tv_tiled.rank() != 2) return Error.InvalidTiler;
        if (tiler_mn.rank() == 0 or tiler_mn.rank() > 2) return Error.InvalidTiler;
        try tiler_mn.assertPositive();
        return .{
            .base = base,
            .layout_tv_tiled = layout_tv_tiled,
            .tiler_mn = tiler_mn,
            .layout_src_tv_tiled = try retileTv(base.layoutSrcTv(), tiler_mn),
            .layout_dst_tv_tiled = try retileTv(base.layoutDstTv(), tiler_mn),
        };
    }

    pub fn size(self: TiledCopy) Error!layout.Unsigned {
        return layout_algebra.sizeOf(&self.layout_tv_tiled, 0);
    }

    pub fn getSlice(self: TiledCopy, thread_index: layout.Scalar) Error!ThrCopy {
        const n = try self.size();
        if (thread_index < 0 or @as(layout.Unsigned, @intCast(thread_index)) >= n) return Error.InvalidThreadIndex;
        return .{ .base = self, .thr_idx = thread_index };
    }

    pub fn retile(self: TiledCopy, input: layout.Layout) Error!layout.Layout {
        _ = self;
        return input;
    }

    pub fn emitRetile(self: TiledCopy, builder: anytype, input: mlir.Operand, input_type: mlir.Type, result_type: mlir.Type) Error!mlir.Value {
        _ = self;
        return try builder.genericOp("cute.tiled.copy.retile", &.{input}, &.{}, &.{input_type}, &.{result_type});
    }
};

pub const ThrCopy = struct {
    base: TiledCopy,
    thr_idx: layout.Scalar,

    pub fn partitionS(self: ThrCopy, src: layout.Layout) Error!layout.Layout {
        _ = self;
        return src;
    }

    pub fn partitionD(self: ThrCopy, dst: layout.Layout) Error!layout.Layout {
        _ = self;
        return dst;
    }

    pub fn emitPartition(self: ThrCopy, comptime operand: Operand, builder: anytype, tensor: mlir.Operand, thr_idx: mlir.Operand, tensor_type: mlir.Type, idx_type: mlir.Type, result_type: mlir.Type) Error!mlir.Value {
        _ = self;
        if (operand != .S and operand != .D) return Error.InvalidOperand;
        return try builder.genericOp(if (operand == .S) "cute.tiled.copy.partition_S" else "cute.tiled.copy.partition_D", &.{ tensor, thr_idx }, &.{}, &.{ tensor_type, idx_type }, &.{result_type});
    }
};

pub const TensorOperand = struct {
    value: mlir.Operand,
    ty: mlir.Type,
    element: typing.Numeric,
    v_rank: usize = 1,
};

pub fn makeMmaAtom(op: OpDescriptor, trait: Trait) Error!MmaAtom {
    return MmaAtom.init(op, trait);
}

pub fn makeCopyAtom(op: OpDescriptor, trait: Trait) Error!CopyAtom {
    return CopyAtom.init(op, trait);
}

pub fn makeTiledMma(atom: MmaAtom, atom_layout_mnk: layout.Layout, permutation_mnk: ?layout.Tree) Error!TiledMma {
    return TiledMma.init(atom, atom_layout_mnk, permutation_mnk);
}

pub fn makeTiledCopy(atom: CopyAtom, layout_tv_tiled: layout.Layout, tiler_mn: layout.Tree) Error!TiledCopy {
    return TiledCopy.init(atom, layout_tv_tiled, tiler_mn);
}

pub fn makeTiledCopyTv(atom: CopyAtom, thr_layout: layout.Layout, val_layout: layout.Layout) Error!TiledCopy {
    const tv_shape = try layout.Tree.initTuple(&.{ thr_layout.shape, val_layout.shape });
    const layout_tv = try layout.Layout.makeCompact(tv_shape);
    const tiler = try productEachTuple(&.{ thr_layout.shape, val_layout.shape });
    return makeTiledCopy(atom, layout_tv, tiler);
}

pub fn makeTiledCopyA(atom: CopyAtom, tiled_mma: TiledMma) Error!TiledCopy {
    return makeTiledCopy(atom, tiled_mma.tv_layout_a_tiled, try makeTile2(try tiled_mma.getTileSize(0), try tiled_mma.getTileSize(2)));
}

pub fn makeTiledCopyB(atom: CopyAtom, tiled_mma: TiledMma) Error!TiledCopy {
    return makeTiledCopy(atom, tiled_mma.tv_layout_b_tiled, try makeTile2(try tiled_mma.getTileSize(1), try tiled_mma.getTileSize(2)));
}

pub fn makeTiledCopyC(atom: CopyAtom, tiled_mma: TiledMma) Error!TiledCopy {
    return makeTiledCopy(atom, tiled_mma.tv_layout_c_tiled, try makeTile2(try tiled_mma.getTileSize(0), try tiled_mma.getTileSize(1)));
}

pub fn makeTiledCopyS(atom: CopyAtom, tiled_copy: TiledCopy) Error!TiledCopy {
    return makeTiledCopy(atom, tiled_copy.layout_src_tv_tiled, tiled_copy.tiler_mn);
}

pub fn makeTiledCopyD(atom: CopyAtom, tiled_copy: TiledCopy) Error!TiledCopy {
    return makeTiledCopy(atom, tiled_copy.layout_dst_tv_tiled, tiled_copy.tiler_mn);
}

pub fn copyAtomCall(builder: anytype, atom: CopyAtom, src: []const TensorOperand, dst: []const TensorOperand, pred: ?TensorOperand) Error!void {
    _ = atom;
    if (src.len == 0 or dst.len == 0) return Error.InvalidOperandCount;
    if (dst.len == 1 and src[0].element.width != dst[0].element.width) return Error.TypeWidthMismatch;
    if (src[0].v_rank > 2 or dst[0].v_rank > 2) return Error.UnsupportedRank;
    var operands: [17]mlir.Operand = undefined;
    var types: [17]mlir.Type = undefined;
    var count: usize = 0;
    for (src) |t| {
        operands[count] = t.value;
        types[count] = t.ty;
        count += 1;
    }
    for (dst) |t| {
        operands[count] = t.value;
        types[count] = t.ty;
        count += 1;
    }
    if (pred) |p| {
        operands[count] = p.value;
        types[count] = p.ty;
        count += 1;
    }
    try builder.operationNoResult(.{ .name = "cute.copy_atom_call", .operands = operands[0..count], .operand_types = types[0..count], .result_types = &.{} });
}

pub fn mmaAtomCall(builder: anytype, atom: MmaAtom, d: TensorOperand, a: []const TensorOperand, b: []const TensorOperand, c: TensorOperand) Error!void {
    _ = atom;
    if (a.len == 0 or b.len == 0) return Error.InvalidOperandCount;
    var operands: [33]mlir.Operand = undefined;
    var types: [33]mlir.Type = undefined;
    var count: usize = 0;
    operands[count] = d.value;
    types[count] = d.ty;
    count += 1;
    for (a) |t| {
        operands[count] = t.value;
        types[count] = t.ty;
        count += 1;
    }
    for (b) |t| {
        operands[count] = t.value;
        types[count] = t.ty;
        count += 1;
    }
    operands[count] = c.value;
    types[count] = c.ty;
    count += 1;
    try builder.operationNoResult(.{ .name = "cute.mma_atom_call", .operands = operands[0..count], .operand_types = types[0..count], .result_types = &.{} });
}

fn makeTile2(a: layout.Unsigned, b: layout.Unsigned) Error!layout.Tree {
    return layout.Tree.initTuple(&.{
        try layout.Tree.initLeaf(@intCast(a)),
        try layout.Tree.initLeaf(@intCast(b)),
    });
}

fn prependTree(tree: *const layout.Tree, elem: layout.Tree) Error!layout.Tree {
    var parts: [128]layout.Tree = undefined;
    parts[0] = elem;
    const r = tree.rank();
    for (0..r) |i| parts[i + 1] = try tree.topMode(i);
    return layout.Tree.initTuple(parts[0 .. r + 1]);
}

fn modeTile(atom_shape: *const layout.Tree, atom_layout_shape: *const layout.Tree, mode_idx: usize) Error!layout.Tree {
    const a = try atom_shape.topMode(mode_idx);
    const b = try atom_layout_shape.topMode(mode_idx);
    const prod = (try a.product()) * (try b.product());
    return layout.Tree.initLeaf(@intCast(prod));
}

fn retileTv(tv: layout.Layout, tiler: layout.Tree) Error!layout.Layout {
    const threads = try tv.shape.topMode(0);
    const values = try tv.shape.topMode(1);
    const new_values = layout.Tree.initLeaf(@intCast((try values.product()) * (try tiler.product()))) catch unreachable;
    const shape = try layout.Tree.initTuple(&.{ threads, new_values });
    return layout.Layout.makeCompact(shape);
}

fn divideCeil(a: layout.Unsigned, b: layout.Unsigned) layout.Unsigned {
    return (a + b - 1) / b;
}

fn divideShape2(shape: layout.Tree, d0: layout.Unsigned, d1: layout.Unsigned) Error!layout.Tree {
    if (shape.rank() != 2) return Error.InvalidAtomLayout;
    const s0 = try shape.topMode(0);
    const s1 = try shape.topMode(1);
    return layout.Tree.initTuple(&.{
        try layout.Tree.initTuple(&.{ layout.Tree.initLeaf(@intCast(d0)) catch unreachable, layout.Tree.initLeaf(@intCast(divideCeil(try s0.product(), d0))) catch unreachable }),
        try layout.Tree.initTuple(&.{ layout.Tree.initLeaf(@intCast(d1)) catch unreachable, layout.Tree.initLeaf(@intCast(divideCeil(try s1.product(), d1))) catch unreachable }),
    });
}

fn productEachTuple(parts: []const layout.Tree) Error!layout.Tree {
    var leaves: [128]layout.Tree = undefined;
    for (parts, 0..) |p, i| leaves[i] = try layout.Tree.initLeaf(@intCast(try p.product()));
    return layout.Tree.initTuple(leaves[0..parts.len]);
}

test "atom: mma and copy descriptors validate trait completeness and runtime fields" {
    const thr = layout.makeCompactLayout(.{32});
    const tv = layout.makeCompactLayout(.{ 2, 2 });
    var trait_value: Trait = .{
        .name = "generic_mma_trait",
        .thr_id = thr,
        .shape_mnk = layout.Tree.fromComptime(.{ 16, 8, 8 }),
        .admissible_fields = &.{ .accumulate, .negate_a },
    };
    trait_value = trait_value.withMmaLayouts(tv, tv, tv);
    const mma_atom = try makeMmaAtom(OpDescriptor.mmaTyped("mma.sync.placeholder", "generic", "test", layout.Tree.fromComptime(.{ 16, 8, 8 }), typing.Float16, typing.Float16, typing.Float32, &.{ .accumulate, .negate_a }), trait_value);
    try std.testing.expectEqual(@as(layout.Scalar, 16), (try mma_atom.shapeMnk().flattenLeaves()).at(0));

    var mutable = mma_atom;
    try mutable.set(.accumulate, .{ .bool = true });
    try std.testing.expect((try mutable.get(.accumulate)).eql(.{ .bool = true }));
    try std.testing.expectError(Error.UnsupportedField, mutable.set(.sfa, .{ .u64 = 1 }));

    var out: mlir.TextBuffer(256) = .{};
    try mma_atom.atom.writeMlirType(&out);
    try std.testing.expectEqualStrings("!cute.atom<mma, generic, test, mma.sync.placeholder, trait=generic_mma_trait>", out.slice());

    var copy_trait: Trait = .{ .name = "generic_copy_trait", .thr_id = thr };
    copy_trait = copy_trait.withCopyLayouts(tv, tv);
    const copy_atom = try makeCopyAtom(OpDescriptor.copy("copy.placeholder", "generic", typing.Float32), copy_trait);
    try std.testing.expectEqualStrings("Float32", copy_atom.valueType().name);
}

test "atom: tiled mma computes tile sizes, partitions, and copy adapters" {
    const thr = layout.makeCompactLayout(.{32});
    const tv = layout.makeCompactLayout(.{ 32, 1 });
    var trait_value: Trait = .{ .name = "mma", .thr_id = thr, .shape_mnk = layout.Tree.fromComptime(.{ 16, 8, 8 }) };
    trait_value = trait_value.withMmaLayouts(tv, tv, tv);
    const atom_value = try makeMmaAtom(OpDescriptor.mma("mma", "generic"), trait_value);
    const tiled = try makeTiledMma(atom_value, layout.makeCompactLayout(.{ 2, 3, 4 }), null);
    try std.testing.expectEqual(@as(layout.Unsigned, 32), try tiled.getTileSize(0));
    try std.testing.expectEqual(@as(layout.Unsigned, 24), try tiled.getTileSize(1));
    try std.testing.expectEqual(@as(layout.Unsigned, 32), try tiled.getTileSize(2));
    const part = try tiled.partitionShapeA(layout.Tree.fromComptime(.{ 64, 64 }));
    try std.testing.expect(part.rank() == 2);

    var copy_trait: Trait = .{ .name = "copy", .thr_id = thr };
    copy_trait = copy_trait.withCopyLayouts(tv, tv);
    const copy_atom = try makeCopyAtom(OpDescriptor.copy("copy", "generic", typing.Int32), copy_trait);
    const copy_a = try makeTiledCopyA(copy_atom, tiled);
    const slice = try copy_a.getSlice(0);
    try std.testing.expectEqual(@as(layout.Scalar, 0), slice.thr_idx);
}

test "atom: tiled copy thread slices and MLIR call emission are checked" {
    const thr = layout.makeCompactLayout(.{4});
    const tv = layout.makeCompactLayout(.{ 4, 1 });
    var copy_trait: Trait = .{ .name = "copy", .thr_id = thr };
    copy_trait = copy_trait.withCopyLayouts(tv, tv);
    const atom_value = try makeCopyAtom(OpDescriptor.copy("copy", "generic", typing.Int32), copy_trait);
    const tiled = try makeTiledCopy(atom_value, tv, layout.Tree.fromComptime(.{4}));
    const slice = try tiled.getSlice(3);
    try std.testing.expectEqual(@as(layout.Scalar, 3), slice.thr_idx);
    try std.testing.expectError(Error.InvalidThreadIndex, tiled.getSlice(4));

    var b: mlir.Builder(1024) = .{};
    const t: TensorOperand = .{ .value = .named("src"), .ty = .raw("!cute.memref<i32, gmem, \"(1):(1)\">"), .element = typing.Int32 };
    const d: TensorOperand = .{ .value = .named("dst"), .ty = .raw("!cute.memref<i32, gmem, \"(1):(1)\">"), .element = typing.Int32 };
    try copyAtomCall(&b, atom_value, &.{t}, &.{d}, null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cute.copy_atom_call") != null);
}
