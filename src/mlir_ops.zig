const std = @import("std");

/// Source-grounded MLIR dialect operation inventory.
///
/// This is intentionally a registry of textual operation spelling, not generated
/// bindings.  The uploaded CuteDSL tree imports generated Python MLIR bindings
/// from `cutlass._mlir`; those bindings are not part of the source archive and
/// would violate the requested zero-dependency direction.  The Zig port therefore
/// records the dialect/op names used by the Python code and emits them as text.
pub const Dialect = enum {
    arith,
    builtin,
    cf,
    cuda,
    cute,
    cute_nvgpu,
    func,
    gpu,
    lir,
    llvm,
    math,
    nvgpu,
    nvvm,
    scf,
    vector,
};

pub const KnownOp = struct {
    dialect: Dialect,
    name: []const u8,

    pub fn fullName(self: KnownOp, out: anytype) !void {
        try out.append(@tagName(self.dialect));
        try out.append(".");
        try out.append(self.name);
    }
};

pub const arith_ops = [_][]const u8{
    "constant", "addi",       "addf",      "subi",   "subf",   "muli",     "mulf",       "divui",        "divsi",
    "divf",     "floordivsi", "ceildivsi", "remui",  "remsi",  "remf",     "andi",       "ori",          "xori",
    "shli",     "shrui",      "shrsi",     "cmpi",   "cmpf",   "select",   "index_cast", "index_castui", "extui",
    "extsi",    "trunci",     "extf",      "truncf", "sitofp", "uitofp",   "fptosi",     "fptoui",       "bitcast",
    "negf",     "minsi",      "minui",     "maxsi",  "maxui",  "minimumf", "maximumf",
};

pub const math_ops = [_][]const u8{
    "absf", "absi",  "acos", "asin",  "atan",  "atan2", "ceil",  "copysign", "cos",  "ctpop",
    "erf",  "exp",   "exp2", "floor", "fpowi", "gcd",   "ipowi", "log",      "log2", "log10",
    "powf", "rsqrt", "sin",  "sqrt",  "tan",   "tanh",
};

pub const builtin_ops = [_][]const u8{
    "module", "unrealized_conversion_cast",
};

pub const func_ops = [_][]const u8{
    "func", "return", "call",
};

pub const gpu_ops = [_][]const u8{
    "module", "container_module", "binary", "global",    "launch",   "launch_func", "printf",
    "return", "sync",             "wait",   "thread_id", "block_id", "grid_dim",    "block_dim",
};

pub const vector_ops = [_][]const u8{
    "broadcast",     "bitcast",        "constant_mask", "extract",       "extractelement",       "extract_strided_slice",
    "from_elements", "gather",         "insert",        "insertelement", "insert_strided_slice", "multi_reduction",
    "reduction",     "scatter",        "shape_cast",    "shuffle",       "splat",                "to_elements",
    "transfer_read", "transfer_write",
};

pub const llvm_ops = [_][]const u8{
    "addrspacecast", "alloca",  "and",  "bitcast",        "br",            "call",              "cond_br",    "extractelement",
    "extractvalue",  "fptrunc", "func", "getelementptr",  "global",        "icmp",              "inline_asm", "insertvalue",
    "inttoptr",      "load",    "mul",  "or",             "ptrtoint",      "return",            "sitofp",     "store",
    "trunc",         "urem",    "xor",  "mlir.addressof", "mlir.constant", "mlir.global_dtors", "mlir.undef", "mlir.zero",
    "mlir.poison",
};

pub const scf_ops = [_][]const u8{
    "for", "if", "while", "condition", "execute_region", "yield",
};

pub const cf_ops = [_][]const u8{
    "assert", "br", "cond_br",
};

pub const nvvm_ops = [_][]const u8{
    "barrier",                  "barrier0",               "barrier_arrive",        "bar_warp_sync",         "cp_async_bulk_commit_group",
    "cp_async_bulk_wait_group", "cp_async_commit_group",  "cp_async_wait_group",   "cluster_arrive",        "cluster_arrive_relaxed",
    "cluster_wait",             "elect_sync",             "fence_acq_rel_cta",     "fence_acq_rel_cluster", "fence_acq_rel_gpu",
    "fence_acq_rel_sys",        "fence_proxy",            "fma_packed_f32x2",      "match_sync",            "mapa",
    "mbarrier_init_shared",     "mbarrier_txn",           "prefetch",              "read.ptx.sreg.clock",   "read.ptx.sreg.clock64",
    "read.ptx.sreg.ctaid.x",    "read.ptx.sreg.ctaid.y",  "read.ptx.sreg.ctaid.z", "read.ptx.sreg.laneid",  "read.ptx.sreg.nctaid.x",
    "read.ptx.sreg.nctaid.y",   "read.ptx.sreg.nctaid.z", "read.ptx.sreg.ntid.x",  "read.ptx.sreg.ntid.y",  "read.ptx.sreg.ntid.z",
    "read.ptx.sreg.smid",       "read.ptx.sreg.tid.x",    "read.ptx.sreg.tid.y",   "read.ptx.sreg.tid.z",   "redux_sync",
    "setmaxregister",           "shfl.sync",              "store",                 "load",                  "tcgen05_commit",
    "tcgen05_wait",             "vote_ballot_sync",       "vote_sync",
};

pub const cuda_ops = [_][]const u8{
    "cast",                   "kernel",                           "launch_cfg_create",      "launch_cfg_programmatic_stream_serialization_allowed",
    "launch_cfg_cluster_dim", "launch_cfg_preferred_cluster_dim", "launch_cfg_cooperative", "launch_ex",
    "return",                 "return_if_error",
};

pub const cute_ops = [_][]const u8{
    "assume",           "blocked_product",       "complement",          "copy",               "cosize",             "deref_arith_tuple_iter",
    "elem_less",        "equal",                 "filter",              "filter_zeros",       "flat_product",       "gemm",
    "get_iter",         "get_layout",            "get_leaves",          "get_shape",          "inttoptr",           "is_static",
    "logical_product",  "make_arith_tuple_iter", "make_atom",           "make_coord",         "make_fragment_like", "make_identity_tensor",
    "make_int_tuple",   "make_layout",           "make_layout_like",    "make_shape",         "make_stride",        "make_tensor",
    "make_tile",        "make_view",             "memref_alloca",       "memref_load",        "memref_load_vec",    "memref_store",
    "memref_store_vec", "mma_make_fragment",     "pack_coord",          "pack_int_tuple",     "pack_shape",         "pack_stride",
    "pack_tile",        "prefetch",              "prepend_to_rank",     "print_view",         "raked_product",      "slice",
    "static",           "tile_to_shape",         "tiled_mma_partition", "tiled_product",      "tuple_add",          "tuple_div",
    "tuple_mod",        "tuple_mul",             "tuple_product",       "tuple_product_each", "tuple_sub",          "zipped_product",
};

pub const cute_nvgpu_ops = [_][]const u8{
    "arch_alloc_smem",                     "arch_get_dyn_smem",                 "arch_get_dyn_smem_size",                  "arch_make_warp_uniform",
    "arch_sm100_alloc_tmem",               "arch_sm100_dealloc_tmem",           "arch_sm100_relinquish_tmem_alloc_permit", "arch_sm100_retrieve_tmem_ptr",
    "atom_get_copy_s2t_smem_desc_view",    "atom_get_value",                    "atom_make_exec_tma",                      "atom_make_non_exec_im2col_tma_load",
    "atom_make_non_exec_im2col_tma_store", "atom_make_non_exec_tiled_tma_load", "atom_make_non_exec_tiled_tma_reduce",     "atom_make_non_exec_tiled_tma_store",
    "atom_make_s2t_copy",                  "atom_make_tmem_copy",               "atom_set_value",                          "atom_tma_partition",
    "copy_tma_desc",                       "get_default_tma_format",            "make_tmem_layout_sfa",                    "make_tmem_layout_sfb",
    "make_umma_smem_desc",                 "prefetch_tma_desc",                 "tile_to_mma_shape",                       "update_tma_desc",
};

pub const lir_ops = [_][]const u8{
    "allocate_buffer",            "copy",                  "create_circular_buffer_pipeline", "create_circular_buffer_pipeline_state",
    "create_pipeline",            "create_pipeline_state", "create_pipeline_with_mask",       "dot",
    "dot_block_scaled",           "func",                  "get_mbarrier",                    "get_pipeline_consume_stage",
    "get_pipeline_produce_stage", "mbarrier_expect_tx",    "partition",                       "pipeline_advance_iterator",
    "producer_acquire",           "producer_commit",       "producer_try_acquire",            "consumer_release",
    "consumer_tail",              "consumer_try_wait",     "consumer_wait",                   "return",
    "simt_auto_vec_copy",         "tma_load",              "tma_load_multicast",              "tma_store",
};

fn table(dialect: Dialect) []const []const u8 {
    return switch (dialect) {
        .arith => &arith_ops,
        .builtin => &builtin_ops,
        .cf => &cf_ops,
        .cuda => &cuda_ops,
        .cute => &cute_ops,
        .cute_nvgpu => &cute_nvgpu_ops,
        .func => &func_ops,
        .gpu => &gpu_ops,
        .lir => &lir_ops,
        .llvm => &llvm_ops,
        .math => &math_ops,
        .nvgpu => &[_][]const u8{},
        .nvvm => &nvvm_ops,
        .scf => &scf_ops,
        .vector => &vector_ops,
    };
}

pub fn isKnown(dialect: Dialect, name: []const u8) bool {
    for (table(dialect)) |candidate| {
        if (std.mem.eql(u8, candidate, name)) return true;
    }
    return false;
}

pub fn isKnownFullName(full_name: []const u8) bool {
    const dot = std.mem.indexOfScalar(u8, full_name, '.') orelse return false;
    const prefix = full_name[0..dot];
    const suffix = full_name[dot + 1 ..];
    inline for (@typeInfo(Dialect).@"enum".fields) |field| {
        if (std.mem.eql(u8, prefix, field.name)) {
            return isKnown(@enumFromInt(field.value), suffix);
        }
    }
    return false;
}

test "mlir_ops: source-grounded registry recognizes major CuteDSL ops" {
    try std.testing.expect(isKnown(.cute, "make_layout"));
    try std.testing.expect(isKnown(.cute, "tile_to_shape"));
    try std.testing.expect(isKnown(.cute_nvgpu, "atom_make_exec_tma"));
    try std.testing.expect(isKnown(.arith, "cmpi"));
    try std.testing.expect(isKnown(.nvvm, "shfl.sync"));
    try std.testing.expect(isKnown(.lir, "tma_load"));
    try std.testing.expect(!isKnown(.cute, "not_a_real_cute_op"));
}
