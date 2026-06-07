const std = @import("std");
const layout = @import("layout.zig");
const tuple = @import("tuple.zig");
const core = @import("core.zig");
const basis = @import("basis.zig");
const typing = @import("typing.zig");
const runtime = @import("runtime.zig");
const mlir = @import("mlir_text.zig");
const tensor_api = @import("tensor_api.zig");
const algorithm = @import("algorithm.zig");
const experimental = @import("experimental.zig");
const testing = @import("testing.zig");
const export_ = @import("export.zig");
const atom = @import("atom.zig");

pub const Error = layout.Error || runtime.Error || mlir.Error || tensor_api.Error || algorithm.Error || experimental.Error || testing.Error || export_.Error || error{ AssertionFailed, InvalidPipelineState, UnsupportedCompatOperation };

pub const ModeOpDecorator = struct { name: []const u8 = "" };
pub const AssertionError = error{AssertionFailed};
pub const CantImplementError = error{CantImplement};
pub const CuptiProfiler = struct { enabled: bool = false };
pub const AllocationRequirement = struct { bytes: usize = 0, alignment: usize = 16 };
pub const QueryDeviceWorkspaceFunc = *const fn () usize;
pub const SymIntId = struct { id: usize = 0 };
pub const ConverterContext = struct { arg_count: usize = 0 };
pub const CuteCHeaderGenerator = export_.CHeader;
pub const CuteSignatureProcessor = export_.CWrapper;
pub const GenericPipelineBase = Pipeline;
pub const GenericPipeline = Pipeline;
pub const TMAToUMMAPipeline = Pipeline;
pub const TMAToUMMACircularPipeline = Pipeline;
pub const TMAToAsyncPipeline = Pipeline;
pub const AsyncToUMMAPipeline = Pipeline;
pub const UMMAtoAsyncPipeline = Pipeline;
pub const TMAStorePipeline = Pipeline;
pub const GroupedGemmSchedulerPipeline = Pipeline;
pub const CLCPipeline = Pipeline;

pub const PipelineState = struct { stage: i32 = 0, phase: i32 = 0, count: i32 = 0 };
pub const CircularBufferPipelineState = PipelineState;
pub const Pipeline = struct { stages: u16 = 1, producer: PipelineState = .{}, consumer: PipelineState = .{} };

pub fn pretty_str(tree: *const layout.Tree, out: anytype) Error!void {
    return writeTree(tree, out);
}
pub fn printf(builder: anytype, fmt: []const u8, operands: []const mlir.Operand, operand_types: []const mlir.Type) Error!void {
    try builder.operationNoResult(.{ .name = "cute.printf", .operands = operands, .attrs = &.{.{ .key = "format", .value = fmt }}, .operand_types = operand_types, .result_types = &.{} });
}
pub fn front(tree: *const layout.Tree) Error!layout.Scalar {
    const flat = try tree.flattenLeaves();
    if (flat.len == 0) return Error.InvalidShape;
    return flat.at(0);
}
pub fn is_major(l: *const layout.Layout, mode: usize) Error!bool {
    const flat = try l.stride.flattenLeaves();
    if (mode >= flat.len) return Error.IndexOutOfBounds;
    return flat.at(mode) == 1;
}
pub fn assume(cond: bool) Error!void {
    if (!cond) return Error.AssertionFailed;
}
pub fn make_swizzle(comptime bits: u16, comptime base: u16, comptime shift: i16) basis.Swizzle {
    return basis.Swizzle.init(bits, base, shift);
}
pub fn recast_ptr(ptr: runtime.Pointer, dtype: typing.Numeric) runtime.Pointer {
    var out = ptr;
    out.dtype = dtype;
    out.assumed_align = @min(out.assumed_align, dtype.bytes());
    return out;
}
pub fn make_ptr(address: usize, dtype: typing.Numeric, memspace: typing.AddressSpace, align_: ?usize) Error!runtime.Pointer {
    return runtime.Pointer.init(address, dtype, memspace, align_);
}
pub fn @"struct"(name: []const u8) ModeOpDecorator {
    return .{ .name = name };
}
pub fn @"union"(name: []const u8) ModeOpDecorator {
    return .{ .name = name };
}

pub fn wrap(value: layout.Scalar) Error!layout.Tree {
    return tuple.wrapScalar(value);
}
pub fn unwrap(value: layout.Tree) Error!layout.Tree {
    return tuple.unwrapSingleton(value);
}
pub fn flatten_to_tuple(value: *const layout.Tree) Error!layout.Flat {
    return tuple.flattenToTuple(value);
}
pub fn product_like(value: *const layout.Tree, profile: *const layout.Tree) Error!layout.Tree {
    return tuple.productLike(value, profile);
}
pub fn product_each(value: *const layout.Tree) Error!layout.Tree {
    return tuple.productEach(value);
}
pub fn find_if(value: *const layout.Tree, needle: layout.Scalar) Error!tuple.FindResult {
    return tuple.findScalar(value, needle, true);
}
pub fn transform_leaf(value: *const layout.Tree, comptime f: fn (layout.Scalar) layout.Scalar) Error!layout.Tree {
    return tuple.mapLeaves(value, f);
}
pub fn tuple_cat(values: []const layout.Tree) Error!layout.Tree {
    return tuple.tupleCat(values);
}
pub fn transform_apply(value: *const layout.Tree, comptime f: fn (layout.Scalar) layout.Scalar) Error!layout.Tree {
    return tuple.mapLeaves(value, f);
}
pub fn filter_tuple(value: *const layout.Tree, comptime pred: fn (layout.Scalar) bool) Error!layout.Tree {
    const flat = try value.flattenLeaves();
    var out: layout.Flat = .{};
    for (flat.slice()) |v| if (pred(v)) try out.append(v);
    return layout.Tree.fromFlat(out.slice());
}

pub fn sym_int(divisibility: u64, symbol: ?[]const u8) Error!typing.SymInt {
    return typing.symInt64(divisibility, symbol);
}
pub fn sym_int32(divisibility: u64, symbol: ?[]const u8) Error!typing.SymInt {
    return typing.symInt32(divisibility, symbol);
}
pub fn sym_int64(divisibility: u64, symbol: ?[]const u8) Error!typing.SymInt {
    return typing.symInt64(divisibility, symbol);
}
pub fn is_integer(dtype: typing.Numeric) bool {
    return dtype.isInteger();
}
pub fn is_int_tuple(tree: *const layout.Tree) bool {
    return tree.leafCount() > 0;
}

pub fn make_fake_compact_tensor(dtype: typing.Numeric, shape: layout.Tree, memspace: typing.AddressSpace, align_: ?usize) Error!runtime.FakeTensor {
    return runtime.makeFakeCompactTensor(dtype, shape, memspace, align_);
}
pub fn make_fake_tensor(dtype: typing.Numeric, shape: layout.Tree, stride: layout.Tree, memspace: typing.AddressSpace, align_: ?usize) Error!runtime.FakeTensor {
    return runtime.makeFakeTensor(dtype, shape, stride, memspace, align_);
}
pub fn make_fake_stream() runtime.Stream {
    return runtime.Stream.default();
}
pub fn from_dlpack(_: usize, dtype: typing.Numeric, shape: layout.Tree, memspace: typing.AddressSpace) Error!runtime.FakeTensor {
    return runtime.makeFakeCompactTensor(dtype, shape, memspace, null);
}
pub fn find_runtime_libraries() []const []const u8 {
    return &.{ "libcute_dsl_runtime.so", "libcuda_dialect_runtime_static.a" };
}
pub fn load_module(path: []const u8) Error!runtime.BinaryModule {
    return runtime.BinaryModule.init(path, .cubin);
}

pub fn basic_copy(builder: anytype, src: anytype, dst: anytype) Error!void {
    return algorithm.basicCopy(builder, src, dst);
}
pub fn basic_copy_if(builder: anytype, src: anytype, dst: anytype, pred: anytype) Error!void {
    return algorithm.conditionalCopy(builder, src, dst, pred);
}
pub fn autovec_copy(builder: anytype, src: anytype, dst: anytype) Error!void {
    return algorithm.autovecCopy(builder, src, dst);
}
pub fn simt_auto_vec_copy(builder: anytype, src: anytype, dst: anytype) Error!void {
    return algorithm.autovecCopy(builder, src, dst);
}
pub fn partition(x: anytype) @TypeOf(x) {
    return x;
}
pub fn partition_and_copy(builder: anytype, src: anytype, dst: anytype) Error!void {
    return algorithm.basicCopy(builder, src, dst);
}
pub fn tma_load(builder: anytype, desc: experimental.TmaDescriptor, dst: mlir.Operand) Error!mlir.Value {
    return experimental.emitTmaLoad(builder, desc, dst);
}
pub fn tma_load_multicast(builder: anytype, desc: experimental.TmaDescriptor, dst: mlir.Operand, mask: mlir.Operand) Error!mlir.Value {
    return experimental.emitTmaLoadMulticast(builder, desc, dst, mask);
}
pub fn tma_store(builder: anytype, desc: experimental.TmaDescriptor, src: mlir.Operand) Error!void {
    return experimental.emitTmaStore(builder, desc, src);
}

pub fn elect_sync(builder: anytype) Error!mlir.Value {
    return builder.genericOp("cute.experimental.elect_sync", &.{}, &.{}, &.{}, &.{mlir.Type.i(1)});
}
pub fn get_mbarrier(builder: anytype, ptr: mlir.Operand) Error!mlir.Value {
    return builder.genericOp("cute.experimental.get_mbarrier", &.{ptr}, &.{}, &.{mlir.Type.raw("!cute.ptr")}, &.{mlir.Type.raw("!cute.mbarrier")});
}
pub fn create_pipeline(stages: u16) Pipeline {
    return .{ .stages = stages };
}
pub fn create_pipeline_with_mask(stages: u16, _: u32) Pipeline {
    return create_pipeline(stages);
}
pub fn pipeline_advance_iterator(state: PipelineState) PipelineState {
    return .{ .stage = state.stage + 1, .phase = state.phase, .count = state.count + 1 };
}
pub fn producer_acquire(p: Pipeline) Pipeline {
    return p;
}
pub fn producer_commit(p: Pipeline) Pipeline {
    return p;
}
pub fn consumer_wait(p: Pipeline) Pipeline {
    return p;
}
pub fn consumer_release(p: Pipeline) Pipeline {
    return p;
}
pub fn consumer_release_elect_one_sync(p: Pipeline) Pipeline {
    return p;
}
pub fn consumer_tail(p: Pipeline) Pipeline {
    return p;
}
pub fn get_pipeline_produce_stage(p: Pipeline) u16 {
    return @intCast(@mod(p.producer.stage, p.stages));
}
pub fn get_pipeline_consume_stage(p: Pipeline) u16 {
    return @intCast(@mod(p.consumer.stage, p.stages));
}
pub fn create_circular_buffer_pipeline(stages: u16) Pipeline {
    return create_pipeline(stages);
}
pub fn circular_buffer_pipeline_consume(p: Pipeline) Pipeline {
    return p;
}
pub fn circular_buffer_pipeline_consumer_release(p: Pipeline) Pipeline {
    return p;
}
pub fn circular_buffer_pipeline_advance_iterator(state: PipelineState) PipelineState {
    return pipeline_advance_iterator(state);
}
pub fn mbarrier_expect_tx(builder: anytype, barrier: mlir.Operand, bytes: mlir.Operand) Error!void {
    try builder.operationNoResult(.{ .name = "cute.experimental.mbarrier_expect_tx", .operands = &.{ barrier, bytes }, .operand_types = &.{ mlir.Type.raw("!cute.mbarrier"), mlir.Type.i(32) }, .result_types = &.{} });
}
pub fn normalize_skip_wait_token(token: ?bool) bool {
    return token orelse false;
}
pub fn producer_try_acquire(p: Pipeline) bool {
    return p.stages != 0;
}
pub fn consumer_try_wait(p: Pipeline) bool {
    return p.stages != 0;
}

pub fn assert_(cond: bool) Error!void {
    if (!cond) return Error.AssertionFailed;
}
pub fn convert(value: anytype) @TypeOf(value) {
    return value;
}
pub fn sample_pytest() void {}
pub fn benchmark(iterations: usize) testing.Benchmark {
    return .{ .iterations = iterations };
}
pub fn get_workspace_count() usize {
    return 0;
}
pub fn autotune_jit() void {}
pub fn tune() void {}
pub fn add_tensor_init_args(config: testing.TensorInitConfig) testing.TensorInitConfig {
    return config;
}
pub fn validate_tensor_init_args(config: testing.TensorInitConfig) Error!void {
    return testing.validateTensorInitConfig(config);
}
pub fn tensor_init_config_from_args(kind: testing.TensorInitKind) testing.TensorInitConfig {
    return .{ .kind = kind };
}
pub fn should_use_normal_init(config: testing.TensorInitConfig) bool {
    return config.kind == .normal;
}

pub fn get_libdir() []const u8 {
    return "nvidia_cutlass_dsl/lib";
}
pub fn get_libs() []const []const u8 {
    return find_runtime_libraries();
}
pub fn get_lib_paths() []const []const u8 {
    return find_runtime_libraries();
}
pub fn get_ldflags() []const []const u8 {
    return &.{"-lcute_dsl_runtime"};
}
pub fn attach_args_spec_converter(ctx: ConverterContext) ConverterContext {
    return ctx;
}
pub fn version_checker(_: []const u8) bool {
    return true;
}
pub fn ffi() void {}

pub fn get_cta_v_map_ab(shape: layout.Tree) layout.Tree {
    return shape;
}
pub fn get_cta_v_map_c(shape: layout.Tree) layout.Tree {
    return shape;
}
pub fn make_tmem_layout_acc(shape: layout.Tree) Error!layout.Layout {
    return layout.Layout.makeCompact(shape);
}
pub fn make_tmem_layout_a(shape: layout.Tree) Error!layout.Layout {
    return layout.Layout.makeCompact(shape);
}
pub fn make_t2r_rmem_layout(shape: layout.Tree) Error!layout.Layout {
    return layout.Layout.makeCompact(shape);
}
pub fn epilogue_tma_store(builder: anytype, desc: experimental.TmaDescriptor, src: mlir.Operand) Error!void {
    return tma_store(builder, desc, src);
}
pub fn mainloop_mma(builder: anytype, tiled: atom.TiledMma, d: mlir.Operand, a: mlir.Operand, b: mlir.Operand, c: mlir.Operand) Error!mlir.Value {
    _ = tiled;
    return builder.genericOp("cute.gemm", &.{ d, a, b, c }, &.{}, &.{ mlir.Type.raw("!cute.tensor"), mlir.Type.raw("!cute.tensor"), mlir.Type.raw("!cute.tensor"), mlir.Type.raw("!cute.tensor") }, &.{mlir.Type.raw("!cute.tensor")});
}
pub fn dot_block_scaled(builder: anytype, a: mlir.Operand, b: mlir.Operand, c: mlir.Operand) Error!mlir.Value {
    return builder.genericOp("cute.experimental.dot_block_scaled", &.{ a, b, c }, &.{}, &.{ mlir.Type.f(32), mlir.Type.f(32), mlir.Type.f(32) }, &.{mlir.Type.f(32)});
}
pub fn dot(builder: anytype, a: mlir.Operand, b: mlir.Operand) Error!mlir.Value {
    return builder.genericOp("cute.experimental.dot", &.{ a, b }, &.{}, &.{ mlir.Type.f(32), mlir.Type.f(32) }, &.{mlir.Type.f(32)});
}

fn writeTree(tree: *const layout.Tree, out: anytype) Error!void {
    const flat = try tree.flattenLeaves();
    try out.append("(");
    for (flat.slice(), 0..) |v, i| {
        if (i != 0) try out.append(",");
        try out.appendSigned(v);
    }
    try out.append(")");
}

test "cute_compat: core tuple and runtime source names are usable" {
    const t = layout.Tree.fromComptime(.{ 2, 3 });
    var out: mlir.TextBuffer(64) = .{};
    try pretty_str(&t, &out);
    try std.testing.expectEqualStrings("(2,3)", out.slice());
    try std.testing.expectEqual(@as(layout.Scalar, 2), try front(&t));
    const fake = try make_fake_compact_tensor(typing.Float32, t, .gmem, null);
    try std.testing.expectEqual(@as(usize, 24), try fake.sizeInBytes());
}
