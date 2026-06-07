const std = @import("std");
const typing = @import("typing.zig");
const mlir = @import("mlir_text.zig");
const runtime = @import("runtime.zig");

pub const Error = mlir.Error || runtime.Error || error{ InvalidArchOperation, InvalidMemorySpace, InvalidBarrierPhase };

pub const SmemAllocation = struct {
    bytes: usize,
    alignment: usize = 16,
    dtype: typing.Numeric = typing.Uint8,

    pub fn init(bytes: usize, alignment: usize, dtype: typing.Numeric) Error!SmemAllocation {
        if (bytes == 0 or alignment == 0) return Error.InvalidMemorySpace;
        return .{ .bytes = bytes, .alignment = alignment, .dtype = dtype };
    }

    pub fn emitAlloc(self: SmemAllocation, builder: anytype) Error!mlir.Value {
        _ = self;
        return builder.genericOp("cute.arch.alloc_smem", &.{}, &.{ .{ .key = "bytes", .value = "dynamic" }, .{ .key = "alignment", .value = "dynamic" } }, &.{}, &.{mlir.Type.raw("!cute.ptr")});
    }
};

pub const TmemAllocation = struct {
    columns: u16,
    alignment: u16 = 32,

    pub fn init(columns: u16, alignment: u16) Error!TmemAllocation {
        if (columns == 0 or alignment == 0) return Error.InvalidMemorySpace;
        return .{ .columns = columns, .alignment = alignment };
    }
};

pub const Barrier = struct {
    address: mlir.Operand,
    ty: mlir.Type = mlir.Type.raw("!cute.ptr"),

    pub fn emitInit(self: Barrier, builder: anytype, thread_count: mlir.Operand) Error!void {
        try builder.operationNoResult(.{ .name = "cute.arch.mbarrier_init", .operands = &.{ self.address, thread_count }, .operand_types = &.{ self.ty, mlir.Type.i(32) }, .result_types = &.{} });
    }

    pub fn emitArrive(self: Barrier, builder: anytype) Error!mlir.Value {
        return builder.genericOp("cute.arch.mbarrier_arrive", &.{self.address}, &.{}, &.{self.ty}, &.{mlir.Type.i(64)});
    }

    pub fn emitWait(self: Barrier, builder: anytype, phase: mlir.Operand) Error!void {
        try builder.operationNoResult(.{ .name = "cute.arch.mbarrier_wait", .operands = &.{ self.address, phase }, .operand_types = &.{ self.ty, mlir.Type.i(32) }, .result_types = &.{} });
    }

    pub fn emitExpectTx(self: Barrier, builder: anytype, bytes: mlir.Operand) Error!void {
        try builder.operationNoResult(.{ .name = "cute.arch.mbarrier_expect_tx", .operands = &.{ self.address, bytes }, .operand_types = &.{ self.ty, mlir.Type.i(32) }, .result_types = &.{} });
    }
};

pub const ElectOne = struct {
    pub fn emit(builder: anytype) Error!mlir.Value {
        return builder.genericOp("cute.arch.elect_one", &.{}, &.{}, &.{}, &.{mlir.Type.i(1)});
    }
};

pub const ClusterQuery = enum { cta_rank, cluster_shape_x, cluster_shape_y, cluster_shape_z };

pub fn issueClcQuery(builder: anytype, query: ClusterQuery) Error!mlir.Value {
    return builder.genericOp("cute.arch.issue_clc_query", &.{}, &.{.{ .key = "query", .value = clcName(query) }}, &.{}, &.{mlir.Type.i(32)});
}

pub fn clcResponse(builder: anytype) Error!mlir.Value {
    return builder.genericOp("cute.arch.clc_response", &.{}, &.{}, &.{}, &.{mlir.Type.i(32)});
}

pub fn getDynSmem(builder: anytype) Error!mlir.Value {
    return builder.genericOp("cute.arch.get_dyn_smem", &.{}, &.{}, &.{}, &.{mlir.Type.raw("!cute.ptr")});
}

pub fn getDynSmemSize(builder: anytype) Error!mlir.Value {
    return builder.genericOp("cute.arch.get_dyn_smem_size", &.{}, &.{}, &.{}, &.{mlir.Type.i(32)});
}

pub fn mapDsmemPtr(builder: anytype, ptr: mlir.Operand) Error!mlir.Value {
    return builder.genericOp("cute.arch.map_dsmem_ptr", &.{ptr}, &.{}, &.{mlir.Type.raw("!cute.ptr")}, &.{mlir.Type.raw("!cute.ptr")});
}

pub fn getMaxTmemAllocCols(builder: anytype) Error!mlir.Value {
    return builder.genericOp("cute.arch.get_max_tmem_alloc_cols", &.{}, &.{}, &.{}, &.{mlir.Type.i(32)});
}

pub fn getMinTmemAllocCols(builder: anytype) Error!mlir.Value {
    return builder.genericOp("cute.arch.get_min_tmem_alloc_cols", &.{}, &.{}, &.{}, &.{mlir.Type.i(32)});
}

pub fn allocTmem(builder: anytype, columns: mlir.Operand) Error!mlir.Value {
    return builder.genericOp("cute.arch.alloc_tmem", &.{columns}, &.{}, &.{mlir.Type.i(32)}, &.{mlir.Type.raw("!cute.ptr")});
}

pub fn retrieveTmemPtr(builder: anytype, handle: mlir.Operand) Error!mlir.Value {
    return builder.genericOp("cute.arch.retrieve_tmem_ptr", &.{handle}, &.{}, &.{mlir.Type.i(32)}, &.{mlir.Type.raw("!cute.ptr")});
}

pub fn deallocTmem(builder: anytype, ptr: mlir.Operand) Error!void {
    try builder.operationNoResult(.{ .name = "cute.arch.dealloc_tmem", .operands = &.{ptr}, .operand_types = &.{mlir.Type.raw("!cute.ptr")}, .result_types = &.{} });
}

pub fn relinquishTmemAllocPermit(builder: anytype) Error!void {
    try builder.operationNoResult(.{ .name = "cute.arch.relinquish_tmem_alloc_permit", .operands = &.{}, .operand_types = &.{}, .result_types = &.{} });
}

pub const NumericConversion = enum {
    cvt_i8_bf16,
    cvt_i4_bf16,
    sext_unpacked_i4_i8,

    pub fn opName(self: NumericConversion) []const u8 {
        return switch (self) {
            .cvt_i8_bf16 => "cute.arch.cvt_i8_bf16_intrinsic",
            .cvt_i4_bf16 => "cute.arch.cvt_i4_bf16_intrinsic",
            .sext_unpacked_i4_i8 => "cute.arch.sext_unpacked_i4_i8_intrinsic",
        };
    }
};

pub fn numericConvert(builder: anytype, conversion: NumericConversion, value: mlir.Operand, input_type: mlir.Type, result_type: mlir.Type) Error!mlir.Value {
    return builder.genericOp(conversion.opName(), &.{value}, &.{}, &.{input_type}, &.{result_type});
}

pub fn inlineAsm(builder: anytype, asm_text: []const u8, constraints: []const u8, operands: []const mlir.Operand, operand_types: []const mlir.Type, result_types: []const mlir.Type) Error!mlir.ValueRange {
    return mlir.llvm.inlineAsm(builder, asm_text, constraints, operands, operand_types, result_types, true);
}

fn clcName(q: ClusterQuery) []const u8 {
    return switch (q) {
        .cta_rank => "cta_rank",
        .cluster_shape_x => "cluster_shape_x",
        .cluster_shape_y => "cluster_shape_y",
        .cluster_shape_z => "cluster_shape_z",
    };
}

test "arch: emits barrier, smem, tmem, elect, and conversion ops" {
    var b: mlir.Builder(4096) = .{};
    _ = try getDynSmem(&b);
    const barrier: Barrier = .{ .address = .arg(0) };
    try barrier.emitInit(&b, .arg(1));
    _ = try barrier.emitArrive(&b);
    _ = try ElectOne.emit(&b);
    _ = try allocTmem(&b, .arg(2));
    _ = try numericConvert(&b, .cvt_i8_bf16, .arg(3), mlir.Type.i(32), mlir.Type.bf16());
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "mbarrier_init") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "alloc_tmem") != null);
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cvt_i8_bf16") != null);
}
