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

pub const Error = layout.Error || runtime.Error || mlir.Error ||
    tensor_api.Error || algorithm.Error || experimental.Error ||
    testing.Error || export_.Error ||
    error{ AssertionFailed, InvalidPipelineState, UnsupportedCompatOperation };

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
pub const Pipeline = struct {
    stages: u16 = 1,
    producer: PipelineState = .{},
    consumer: PipelineState = .{},
};

pub fn prettyStr(tree: *const layout.Tree, out: anytype) Error!void {
    return writeTree(tree, out);
}
pub fn printf(
    builder: anytype,
    fmt: []const u8,
    operands: []const mlir.Operand,
    operand_types: []const mlir.Type,
) Error!void {
    try builder.operationNoResult(.{
        .name = "cute.printf",
        .operands = operands,
        .attrs = &.{.{ .key = "format", .value = fmt }},
        .operand_types = operand_types,
        .result_types = &.{},
    });
}
pub fn front(tree: *const layout.Tree) Error!layout.Scalar {
    const flat = try tree.flattenLeaves();
    if (flat.len == 0) return Error.InvalidShape;
    return flat.at(0);
}
pub fn isMajor(l: *const layout.Layout, mode: usize) Error!bool {
    const flat = try l.stride.flattenLeaves();
    if (mode >= flat.len) return Error.IndexOutOfBounds;
    return flat.at(mode) == 1;
}
pub fn assume(cond: bool) Error!void {
    if (!cond) return Error.AssertionFailed;
}
pub fn makeSwizzle(
    comptime bits: u16,
    comptime base: u16,
    comptime shift: i16,
) basis.Swizzle {
    return basis.Swizzle.init(bits, base, shift);
}
pub fn recastPtr(ptr: runtime.Pointer, dtype: typing.Numeric) runtime.Pointer {
    var out = ptr;
    out.dtype = dtype;
    out.assumed_align = @min(out.assumed_align, dtype.bytes());
    return out;
}
pub fn makePtr(
    address: usize,
    dtype: typing.Numeric,
    memspace: typing.AddressSpace,
    align_: ?usize,
) Error!runtime.Pointer {
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
pub fn flattenToTuple(value: *const layout.Tree) Error!layout.Flat {
    return tuple.flattenToTuple(value);
}
pub fn productLike(
    value: *const layout.Tree,
    profile: *const layout.Tree,
) Error!layout.Tree {
    return tuple.productLike(value, profile);
}
pub fn productEach(value: *const layout.Tree) Error!layout.Tree {
    return tuple.productEach(value);
}
pub fn findIf(
    value: *const layout.Tree,
    needle: layout.Scalar,
) Error!tuple.FindResult {
    return tuple.findScalar(value, needle, true);
}
pub fn transformLeaf(
    value: *const layout.Tree,
    comptime f: fn (layout.Scalar) layout.Scalar,
) Error!layout.Tree {
    return tuple.mapLeaves(value, f);
}
pub fn tupleCat(values: []const layout.Tree) Error!layout.Tree {
    return tuple.tupleCat(values);
}
pub fn transformApply(
    value: *const layout.Tree,
    comptime f: fn (layout.Scalar) layout.Scalar,
) Error!layout.Tree {
    return tuple.mapLeaves(value, f);
}
pub fn filterTuple(
    value: *const layout.Tree,
    comptime pred: fn (layout.Scalar) bool,
) Error!layout.Tree {
    const flat = try value.flattenLeaves();
    var out: layout.Flat = .{};
    for (flat.slice()) |v| if (pred(v)) try out.append(v);
    return layout.Tree.fromFlat(out.slice());
}

pub fn symInt(divisibility: u64, symbol: ?[]const u8) Error!typing.SymInt {
    return typing.symInt64(divisibility, symbol);
}
pub fn symInt32(divisibility: u64, symbol: ?[]const u8) Error!typing.SymInt {
    return typing.symInt32(divisibility, symbol);
}
pub fn symInt64(divisibility: u64, symbol: ?[]const u8) Error!typing.SymInt {
    return typing.symInt64(divisibility, symbol);
}
pub fn isInteger(dtype: typing.Numeric) bool {
    return dtype.isInteger();
}
pub fn isIntTuple(tree: *const layout.Tree) bool {
    return tree.leafCount() > 0;
}

pub fn makeFakeCompactTensor(
    dtype: typing.Numeric,
    shape: layout.Tree,
    memspace: typing.AddressSpace,
    align_: ?usize,
) Error!runtime.FakeTensor {
    return runtime.makeFakeCompactTensor(dtype, shape, memspace, align_);
}
pub fn makeFakeTensor(
    dtype: typing.Numeric,
    shape: layout.Tree,
    stride: layout.Tree,
    memspace: typing.AddressSpace,
    align_: ?usize,
) Error!runtime.FakeTensor {
    return runtime.makeFakeTensor(dtype, shape, stride, memspace, align_);
}
pub fn makeFakeStream() runtime.Stream {
    return runtime.Stream.default();
}
pub fn fromDlpack(
    _: usize,
    dtype: typing.Numeric,
    shape: layout.Tree,
    memspace: typing.AddressSpace,
) Error!runtime.FakeTensor {
    return runtime.makeFakeCompactTensor(dtype, shape, memspace, null);
}
pub fn findRuntimeLibraries() []const []const u8 {
    return &.{ "libcute_dsl_runtime.so", "libcuda_dialect_runtime_static.a" };
}
pub fn loadModule(path: []const u8) Error!runtime.BinaryModule {
    return runtime.BinaryModule.init(path, .cubin);
}

pub fn basicCopy(builder: anytype, src: anytype, dst: anytype) Error!void {
    return algorithm.basicCopy(builder, src, dst);
}
pub fn basicCopyIf(
    builder: anytype,
    src: anytype,
    dst: anytype,
    pred: anytype,
) Error!void {
    return algorithm.conditionalCopy(builder, src, dst, pred);
}
pub fn autovecCopy(builder: anytype, src: anytype, dst: anytype) Error!void {
    return algorithm.autovecCopy(builder, src, dst);
}
pub fn simtAutoVecCopy(builder: anytype, src: anytype, dst: anytype) Error!void {
    return algorithm.autovecCopy(builder, src, dst);
}
pub fn partition(x: anytype) @TypeOf(x) {
    return x;
}
pub fn partitionAndCopy(builder: anytype, src: anytype, dst: anytype) Error!void {
    return algorithm.basicCopy(builder, src, dst);
}
pub fn tmaLoad(
    builder: anytype,
    desc: experimental.TmaDescriptor,
    dst: mlir.Operand,
) Error!mlir.Value {
    return experimental.emitTmaLoad(builder, desc, dst);
}
pub fn tmaLoadMulticast(
    builder: anytype,
    desc: experimental.TmaDescriptor,
    dst: mlir.Operand,
    mask: mlir.Operand,
) Error!mlir.Value {
    return experimental.emitTmaLoadMulticast(builder, desc, dst, mask);
}
pub fn tmaStore(
    builder: anytype,
    desc: experimental.TmaDescriptor,
    src: mlir.Operand,
) Error!void {
    return experimental.emitTmaStore(builder, desc, src);
}

pub fn electSync(builder: anytype) Error!mlir.Value {
    return builder.genericOp(
        "cute.experimental.elect_sync",
        &.{},
        &.{},
        &.{},
        &.{mlir.Type.i(1)},
    );
}
pub fn getMbarrier(builder: anytype, ptr: mlir.Operand) Error!mlir.Value {
    return builder.genericOp(
        "cute.experimental.get_mbarrier",
        &.{ptr},
        &.{},
        &.{mlir.Type.raw("!cute.ptr")},
        &.{mlir.Type.raw("!cute.mbarrier")},
    );
}
pub fn createPipeline(stages: u16) Pipeline {
    return .{ .stages = stages };
}
pub fn createPipelineWithMask(stages: u16, _: u32) Pipeline {
    return createPipeline(stages);
}
pub fn pipelineAdvanceIterator(state: PipelineState) PipelineState {
    return .{
        .stage = state.stage + 1,
        .phase = state.phase,
        .count = state.count + 1,
    };
}
pub fn producerAcquire(p: Pipeline) Pipeline {
    return p;
}
pub fn producerCommit(p: Pipeline) Pipeline {
    return p;
}
pub fn consumerWait(p: Pipeline) Pipeline {
    return p;
}
pub fn consumerRelease(p: Pipeline) Pipeline {
    return p;
}
pub fn consumerReleaseElectOneSync(p: Pipeline) Pipeline {
    return p;
}
pub fn consumerTail(p: Pipeline) Pipeline {
    return p;
}
pub fn getPipelineProduceStage(p: Pipeline) u16 {
    return @intCast(@mod(p.producer.stage, p.stages));
}
pub fn getPipelineConsumeStage(p: Pipeline) u16 {
    return @intCast(@mod(p.consumer.stage, p.stages));
}
pub fn createCircularBufferPipeline(stages: u16) Pipeline {
    return createPipeline(stages);
}
pub fn circularBufferPipelineConsume(p: Pipeline) Pipeline {
    return p;
}
pub fn circularBufferPipelineConsumerRelease(p: Pipeline) Pipeline {
    return p;
}
pub fn circularBufferPipelineAdvanceIterator(state: PipelineState) PipelineState {
    return pipelineAdvanceIterator(state);
}
pub fn mbarrierExpectTx(
    builder: anytype,
    barrier: mlir.Operand,
    bytes: mlir.Operand,
) Error!void {
    try builder.operationNoResult(.{
        .name = "cute.experimental.mbarrier_expect_tx",
        .operands = &.{ barrier, bytes },
        .operand_types = &.{ mlir.Type.raw("!cute.mbarrier"), mlir.Type.i(32) },
        .result_types = &.{},
    });
}
pub fn normalizeSkipWaitToken(token: ?bool) bool {
    return token orelse false;
}
pub fn producerTryAcquire(p: Pipeline) bool {
    return p.stages != 0;
}
pub fn consumerTryWait(p: Pipeline) bool {
    return p.stages != 0;
}

pub fn assert_(cond: bool) Error!void {
    if (!cond) return Error.AssertionFailed;
}
pub fn convert(value: anytype) @TypeOf(value) {
    return value;
}
pub fn samplePytest() void {}
pub fn benchmark(iterations: usize) testing.Benchmark {
    return .{ .iterations = iterations };
}
pub fn getWorkspaceCount() usize {
    return 0;
}
pub fn autotuneJit() void {}
pub fn tune() void {}
pub fn addTensorInitArgs(config: testing.TensorInitConfig) testing.TensorInitConfig {
    return config;
}
pub fn validateTensorInitArgs(config: testing.TensorInitConfig) Error!void {
    return testing.validateTensorInitConfig(config);
}
pub fn tensorInitConfigFromArgs(kind: testing.TensorInitKind) testing.TensorInitConfig {
    return .{ .kind = kind };
}
pub fn shouldUseNormalInit(config: testing.TensorInitConfig) bool {
    return config.kind == .normal;
}

pub fn getLibdir() []const u8 {
    return "nvidia_cutlass_dsl/lib";
}
pub fn getLibs() []const []const u8 {
    return findRuntimeLibraries();
}
pub fn getLibPaths() []const []const u8 {
    return findRuntimeLibraries();
}
pub fn getLdflags() []const []const u8 {
    return &.{"-lcute_dsl_runtime"};
}
pub fn attachArgsSpecConverter(ctx: ConverterContext) ConverterContext {
    return ctx;
}
pub fn versionChecker(_: []const u8) bool {
    return true;
}
pub fn ffi() void {}

pub fn getCtaVMapAb(shape: layout.Tree) layout.Tree {
    return shape;
}
pub fn getCtaVMapC(shape: layout.Tree) layout.Tree {
    return shape;
}
pub fn makeTmemLayoutAcc(shape: layout.Tree) Error!layout.Layout {
    return layout.Layout.makeCompact(shape);
}
pub fn makeTmemLayoutA(shape: layout.Tree) Error!layout.Layout {
    return layout.Layout.makeCompact(shape);
}
pub fn makeT2rRmemLayout(shape: layout.Tree) Error!layout.Layout {
    return layout.Layout.makeCompact(shape);
}
pub fn epilogueTmaStore(
    builder: anytype,
    desc: experimental.TmaDescriptor,
    src: mlir.Operand,
) Error!void {
    return tmaStore(builder, desc, src);
}
pub fn mainloopMma(
    builder: anytype,
    tiled: atom.TiledMma,
    d: mlir.Operand,
    a: mlir.Operand,
    b: mlir.Operand,
    c: mlir.Operand,
) Error!mlir.Value {
    _ = tiled;
    return builder.genericOp(
        "cute.gemm",
        &.{ d, a, b, c },
        &.{},
        &.{
            mlir.Type.raw("!cute.tensor"),
            mlir.Type.raw("!cute.tensor"),
            mlir.Type.raw("!cute.tensor"),
            mlir.Type.raw("!cute.tensor"),
        },
        &.{mlir.Type.raw("!cute.tensor")},
    );
}
pub fn dotBlockScaled(
    builder: anytype,
    a: mlir.Operand,
    b: mlir.Operand,
    c: mlir.Operand,
) Error!mlir.Value {
    return builder.genericOp(
        "cute.experimental.dot_block_scaled",
        &.{ a, b, c },
        &.{},
        &.{ mlir.Type.f(32), mlir.Type.f(32), mlir.Type.f(32) },
        &.{mlir.Type.f(32)},
    );
}
pub fn dot(builder: anytype, a: mlir.Operand, b: mlir.Operand) Error!mlir.Value {
    return builder.genericOp(
        "cute.experimental.dot",
        &.{ a, b },
        &.{},
        &.{ mlir.Type.f(32), mlir.Type.f(32) },
        &.{mlir.Type.f(32)},
    );
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
    try prettyStr(&t, &out);
    try std.testing.expectEqualStrings("(2,3)", out.slice());
    try std.testing.expectEqual(@as(layout.Scalar, 2), try front(&t));
    const fake = try makeFakeCompactTensor(typing.Float32, t, .gmem, null);
    try std.testing.expectEqual(@as(usize, 24), try fake.sizeInBytes());
}
