const std = @import("std");
const mlir = @import("mlir_text.zig");
const typing = @import("typing.zig");

pub const Error = mlir.Error || typing.Error || error{
    InvalidNvvmWrapper,
    InvalidOperandCount,
};

pub const ReductionOp = enum { add, min, max, and_, or_, xor };
pub const VoteOp = enum { any, all, uni, ballot };
pub const FenceProxyKind = enum { async_shared, async_tmem };
pub const MatchKind = enum { any, all };
pub const ShiftKind = enum { left, right, wrap };

pub const IntrinsicCall = struct {
    name: []const u8,
    operands: []const mlir.Operand = &.{},
    operand_types: []const mlir.Type = &.{},
    result_type: ?mlir.Type = null,
    attrs: []const mlir.Attribute = &.{},

    pub fn emit(self: IntrinsicCall, builder: anytype) Error!?mlir.Value {
        if (self.operands.len != self.operand_types.len)
            return Error.InvalidOperandCount;
        if (self.result_type) |rt| {
            return try builder.genericOp(
                self.name,
                self.operands,
                self.attrs,
                self.operand_types,
                &.{rt},
            );
        }
        try builder.operationNoResult(.{
            .name = self.name,
            .operands = self.operands,
            .attrs = self.attrs,
            .operand_types = self.operand_types,
            .result_types = &.{},
        });
        return null;
    }
};

fn opName(comptime source_name: []const u8) []const u8 {
    return "cutlass.nvvm." ++ source_name;
}
fn emitValue(
    builder: anytype,
    comptime source_name: []const u8,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    const maybe = try (IntrinsicCall{
        .name = opName(source_name),
        .operands = operands,
        .operand_types = operand_types,
        .result_type = result_type,
    }).emit(builder);
    return maybe.?;
}
fn emitVoid(
    builder: anytype,
    comptime source_name: []const u8,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    _ = try (IntrinsicCall{
        .name = opName(source_name),
        .operands = operands,
        .operand_types = operand_types,
    }).emit(builder);
}

pub fn emit(
    builder: anytype,
    source_name: []const u8,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: ?mlir.Type,
) Error!?mlir.Value {
    var buf: mlir.TextBuffer(128) = .{};
    try buf.append("cutlass.nvvm.");
    try buf.append(source_name);
    return (IntrinsicCall{
        .name = buf.slice(),
        .operands = operands,
        .operand_types = operand_types,
        .result_type = result_type,
    }).emit(builder);
}

pub fn laneIdx(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "lane_idx", &.{}, &.{}, mlir.Type.i(32));
}

pub fn warpIdx(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "warp_idx", &.{}, &.{}, mlir.Type.i(32));
}

pub fn physicalWarpId(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "physical_warp_id", &.{}, &.{}, mlir.Type.i(32));
}

pub fn threadIdx(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "thread_idx", &.{}, &.{}, mlir.Type.i(32));
}

pub fn blockDim(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "block_dim", &.{}, &.{}, mlir.Type.i(32));
}

pub fn blockIdx(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "block_idx", &.{}, &.{}, mlir.Type.i(32));
}

pub fn gridDim(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "grid_dim", &.{}, &.{}, mlir.Type.i(32));
}

pub fn clusterIdx(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "cluster_idx", &.{}, &.{}, mlir.Type.i(32));
}

pub fn clusterDim(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "cluster_dim", &.{}, &.{}, mlir.Type.i(32));
}

pub fn blockInClusterIdx(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "block_in_cluster_idx", &.{}, &.{}, mlir.Type.i(32));
}

pub fn blockInClusterDim(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "block_in_cluster_dim", &.{}, &.{}, mlir.Type.i(32));
}

pub fn clusterSize(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "cluster_size", &.{}, &.{}, mlir.Type.i(32));
}

pub fn blockIdxInCluster(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "block_idx_in_cluster", &.{}, &.{}, mlir.Type.i(32));
}

pub fn dynamicSmemSize(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "dynamic_smem_size", &.{}, &.{}, mlir.Type.i(32));
}

pub fn shuffleSyncOp(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "shuffle_sync_op", operands, operand_types, result_type);
}

pub fn warpReduction(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "warp_reduction", operands, operand_types, result_type);
}

pub fn barrier(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "barrier", operands, operand_types);
}

pub fn barrierArrive(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "barrier_arrive", operands, operand_types);
}

pub fn syncThreads(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "sync_threads", operands, operand_types);
}

pub fn syncWarp(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "sync_warp", operands, operand_types, result_type);
}

pub fn fenceAcqRelCta(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "fence_acq_rel_cta", operands, operand_types);
}

pub fn fenceAcqRelCluster(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "fence_acq_rel_cluster", operands, operand_types);
}

pub fn fenceAcqRelGpu(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "fence_acq_rel_gpu", operands, operand_types);
}

pub fn fenceAcqRelSys(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "fence_acq_rel_sys", operands, operand_types);
}

pub fn cpAsyncCommitGroup(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "cp_async_commit_group", operands, operand_types);
}

pub fn cpAsyncWaitGroup(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "cp_async_wait_group", operands, operand_types);
}

pub fn cpAsyncBulkCommitGroup(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "cp_async_bulk_commit_group", operands, operand_types);
}

pub fn cpAsyncBulkWaitGroup(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "cp_async_bulk_wait_group", operands, operand_types);
}

pub fn clusterWait(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "cluster_wait", operands, operand_types);
}

pub fn clusterArrive(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "cluster_arrive", operands, operand_types);
}

pub fn clusterArriveRelaxed(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "cluster_arrive_relaxed", operands, operand_types);
}

pub fn fenceProxy(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "fence_proxy", operands, operand_types);
}

pub fn voteSyncOp(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "vote_sync_op", operands, operand_types, result_type);
}

pub fn voteBallotSync(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "vote_ballot_sync", operands, operand_types, result_type);
}

pub fn voteAnySync(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "vote_any_sync", operands, operand_types, result_type);
}

pub fn voteAllSync(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "vote_all_sync", operands, operand_types, result_type);
}

pub fn voteUniSync(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "vote_uni_sync", operands, operand_types, result_type);
}

pub fn popc(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "popc", operands, operand_types, result_type);
}

pub fn fenceViewAsyncTmemOp(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "fence_view_async_tmem_op",
        operands,
        operand_types,
        result_type,
    );
}

pub fn fenceViewAsyncShared(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "fence_view_async_shared", operands, operand_types);
}

pub fn setmaxregisterIncrease(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "setmaxregister_increase", operands, operand_types);
}

pub fn setmaxregisterDecrease(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "setmaxregister_decrease", operands, operand_types);
}

pub fn warpgroupRegAlloc(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "warpgroup_reg_alloc", operands, operand_types);
}

pub fn warpgroupRegDealloc(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "warpgroup_reg_dealloc", operands, operand_types);
}

pub fn calcPackedF32x2Op(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "calc_packed_f32x2_op",
        operands,
        operand_types,
        result_type,
    );
}

pub fn fmax(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "fmax", operands, operand_types, result_type);
}

pub fn fmin(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "fmin", operands, operand_types, result_type);
}

pub fn rcpApprox(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "rcp_approx", operands, operand_types, result_type);
}

pub fn exp2(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "exp2", operands, operand_types, result_type);
}

pub fn cvtI8Bf16(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "cvt_i8_bf16", operands, operand_types, result_type);
}

pub fn cvtI8x2ToBf16x2(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "cvt_i8x2_to_bf16x2",
        operands,
        operand_types,
        result_type,
    );
}

pub fn cvtI8x4ToBf16x4(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "cvt_i8x4_to_bf16x4",
        operands,
        operand_types,
        result_type,
    );
}

pub fn cvtF32x2Bf16x2(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "cvt_f32x2_bf16x2", operands, operand_types, result_type);
}

pub fn cvtF32Bf16(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "cvt_f32_bf16", operands, operand_types, result_type);
}

pub fn cvtI8x4ToF32x4(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "cvt_i8x4_to_f32x4",
        operands,
        operand_types,
        result_type,
    );
}

pub fn cvtI8x2ToF32x2(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "cvt_i8x2_to_f32x2",
        operands,
        operand_types,
        result_type,
    );
}

pub fn prmt(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "prmt", operands, operand_types, result_type);
}

pub fn cvtI4Bf16(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "cvt_i4_bf16", operands, operand_types, result_type);
}

pub fn cvtI4ToBf16WithShuffleImpl(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "cvt_i4_to_bf16_with_shuffle_impl",
        operands,
        operand_types,
        result_type,
    );
}

pub fn cvtI4ToBf16Impl(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "cvt_i4_to_bf16_impl",
        operands,
        operand_types,
        result_type,
    );
}

pub fn cvtI4x2ToBf16x2(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "cvt_i4x2_to_bf16x2",
        operands,
        operand_types,
        result_type,
    );
}

pub fn cvtI4x4ToBf16x4(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "cvt_i4x4_to_bf16x4",
        operands,
        operand_types,
        result_type,
    );
}

pub fn cvtI4x8ToBf16x8(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "cvt_i4x8_to_bf16x8",
        operands,
        operand_types,
        result_type,
    );
}

pub fn sextUnpackedI4x4ToI8x4(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "sext_unpacked_i4x4_to_i8x4",
        operands,
        operand_types,
        result_type,
    );
}

pub fn log2OfPow2Int(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "log2_of_pow2_int", operands, operand_types, result_type);
}

pub fn exp(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "exp", operands, operand_types, result_type);
}

pub fn expPackedF32x2(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "exp_packed_f32x2", operands, operand_types, result_type);
}

pub fn griddepcontrolWait(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "griddepcontrol_wait", operands, operand_types);
}

pub fn griddepcontrolLaunchDependents(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(
        builder,
        "griddepcontrol_launch_dependents",
        operands,
        operand_types,
    );
}

pub fn warpReduxSync(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "warp_redux_sync", operands, operand_types, result_type);
}

pub fn atomicAdd(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "atomic_add", operands, operand_types, result_type);
}

pub fn atomicAnd(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "atomic_and", operands, operand_types, result_type);
}

pub fn atomicOr(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "atomic_or", operands, operand_types, result_type);
}

pub fn atomicXor(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "atomic_xor", operands, operand_types, result_type);
}

pub fn atomicMax(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "atomic_max", operands, operand_types, result_type);
}

pub fn atomicMin(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "atomic_min", operands, operand_types, result_type);
}

pub fn atomicExch(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "atomic_exch", operands, operand_types, result_type);
}

pub fn atomicFmax(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "atomic_fmax", operands, operand_types, result_type);
}

pub fn atomicMaxFloat32(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "atomic_max_float32",
        operands,
        operand_types,
        result_type,
    );
}

pub fn atomicCas(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "atomic_cas", operands, operand_types, result_type);
}

pub fn store(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "store", operands, operand_types);
}

pub fn load(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "load", operands, operand_types, result_type);
}

pub fn red(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "red", operands, operand_types);
}

pub fn cvtF4e2m1F16(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "cvt_f4e2m1_f16", operands, operand_types, result_type);
}

pub fn cvtF4e2m1x2ToF16x2(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "cvt_f4e2m1x2_to_f16x2",
        operands,
        operand_types,
        result_type,
    );
}

pub fn cvtF4e2m1x4ToF16x4(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "cvt_f4e2m1x4_to_f16x4",
        operands,
        operand_types,
        result_type,
    );
}

pub fn cvtF4e2m1x8ToF16x8(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(
        builder,
        "cvt_f4e2m1x8_to_f16x8",
        operands,
        operand_types,
        result_type,
    );
}

pub fn smid(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "smid", &.{}, &.{}, mlir.Type.i(32));
}

pub fn nsmid(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "nsmid", &.{}, &.{}, mlir.Type.i(32));
}

pub fn clock(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "clock", &.{}, &.{}, mlir.Type.i(32));
}

pub fn clock64(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "clock64", &.{}, &.{}, mlir.Type.i(64));
}

pub fn matchSync(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "match_sync", operands, operand_types, result_type);
}

pub fn clz(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "clz", operands, operand_types, result_type);
}

pub fn bfind(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "bfind", operands, operand_types, result_type);
}

pub fn brev(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "brev", operands, operand_types, result_type);
}

pub fn bfe(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "bfe", operands, operand_types, result_type);
}

pub fn bfi(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "bfi", operands, operand_types, result_type);
}

pub fn mulHi(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "mul_hi", operands, operand_types, result_type);
}

pub fn mulWide(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "mul_wide", operands, operand_types, result_type);
}

pub fn mul24(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "mul24", operands, operand_types, result_type);
}

pub fn mad24(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "mad24", operands, operand_types, result_type);
}

pub fn addCc(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "add_cc", operands, operand_types, result_type);
}

pub fn addc(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "addc", operands, operand_types, result_type);
}

pub fn subCc(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "sub_cc", operands, operand_types, result_type);
}

pub fn subc(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "subc", operands, operand_types, result_type);
}

pub fn madCc(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "mad_cc", operands, operand_types, result_type);
}

pub fn madc(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "madc", operands, operand_types, result_type);
}

pub fn activemask(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "activemask", &.{}, &.{}, mlir.Type.i(32));
}

pub fn lanemaskLt(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "lanemask_lt", &.{}, &.{}, mlir.Type.i(32));
}

pub fn lanemaskLe(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "lanemask_le", &.{}, &.{}, mlir.Type.i(32));
}

pub fn lanemaskEq(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "lanemask_eq", &.{}, &.{}, mlir.Type.i(32));
}

pub fn lanemaskGe(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "lanemask_ge", &.{}, &.{}, mlir.Type.i(32));
}

pub fn lanemaskGt(builder: anytype) Error!mlir.Value {
    return emitValue(builder, "lanemask_gt", &.{}, &.{}, mlir.Type.i(32));
}

pub fn addSatInt(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "add_sat_int", operands, operand_types, result_type);
}

pub fn subSatInt(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "sub_sat_int", operands, operand_types, result_type);
}

pub fn lop3(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "lop3", operands, operand_types, result_type);
}

pub fn shf(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
    result_type: mlir.Type,
) Error!mlir.Value {
    return emitValue(builder, "shf", operands, operand_types, result_type);
}

pub fn prefetch(
    builder: anytype,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    return emitVoid(builder, "prefetch", operands, operand_types);
}

test "arch_nvvm: source-name wrappers emit deterministic intrinsic calls" {
    var b: mlir.Builder(4096) = .{};
    _ = try laneIdx(&b);
    _ = try shuffleSyncOp(
        &b,
        &.{ .arg(0), .arg(1) },
        &.{ mlir.Type.i(32), mlir.Type.i(32) },
        mlir.Type.i(32),
    );
    try barrier(&b, &.{}, &.{});
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cutlass.nvvm.lane_idx") != null);
    try std.testing.expect(
        std.mem.indexOf(u8, b.slice(), "cutlass.nvvm.shuffle_sync_op") != null,
    );
    try std.testing.expect(std.mem.indexOf(u8, b.slice(), "cutlass.nvvm.barrier") != null);
}
