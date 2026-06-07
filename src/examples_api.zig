const std = @import("std");
const mlir = @import("mlir_text.zig");

pub const Error = mlir.Error || error{
    InvalidExample,
    GoldenMismatch,
};

pub const ExampleKind = enum {
    layout_demo,
    tensor_demo,
    copy_demo,
    mma_demo,
    gemm_skeleton,

    pub fn stem(self: ExampleKind) []const u8 {
        return switch (self) {
            .layout_demo => "layout_demo",
            .tensor_demo => "tensor_demo",
            .copy_demo => "copy_demo",
            .mma_demo => "mma_demo",
            .gemm_skeleton => "gemm_skeleton",
        };
    }
};

pub const all_examples = [_]ExampleKind{
    .layout_demo,
    .tensor_demo,
    .copy_demo,
    .mma_demo,
    .gemm_skeleton,
};

pub const KernelSignature = struct {
    name: []const u8,
    args: []const mlir.Type,
    return_types: []const mlir.Type = &.{},
    attrs: []const mlir.Attribute = &.{},

    pub fn validate(self: KernelSignature) Error!void {
        if (self.name.len == 0) return Error.InvalidExample;
        try mlir.validateSymbol(self.name);
        for (self.args) |arg| try mlir.validateTypeText(arg.text);
        for (self.return_types) |ret| try mlir.validateTypeText(ret.text);
    }
};

pub const KernelBuilder = struct {
    name: []const u8,
    arch: []const u8 = "sm_90",

    pub fn init(name: []const u8) Error!KernelBuilder {
        try mlir.validateSymbol(name);
        return .{ .name = name };
    }

    pub fn validateSignature(self: KernelBuilder, sig: KernelSignature) Error!void {
        _ = self;
        try sig.validate();
    }
};

pub fn exampleText(kind: ExampleKind) []const u8 {
    return switch (kind) {
        .layout_demo => layout_demo_mlir,
        .tensor_demo => tensor_demo_mlir,
        .copy_demo => copy_demo_mlir,
        .mma_demo => mma_demo_mlir,
        .gemm_skeleton => gemm_skeleton_mlir,
    };
}

pub fn emitExample(kind: ExampleKind, builder: anytype) Error!void {
    try builder.append(exampleText(kind));
}

pub fn renderExample(kind: ExampleKind, out: anytype) Error!void {
    try out.append(exampleText(kind));
    try mlir.validateBalancedText(exampleText(kind));
}

pub fn writeExampleIndex(out: anytype) Error!void {
    try out.append("# Not-CuTe examples\n\n");
    for (all_examples) |kind| {
        try out.append("- ");
        try out.append(kind.stem());
        try out.append("\n");
    }
}

pub fn writeRuntimePlanForExample(kind: ExampleKind, out: anytype) Error!void {
    const stem = kind.stem();
    try out.append("{\n");
    try out.append("  \"module\": \"examples.cubin\",\n");
    try out.append("  \"kernel\": \"");
    try out.append(stem);
    try out.append("\",\n");
    try out.append("  \"grid\": [2, 1, 1],\n");
    try out.append("  \"block\": [128, 1, 1],\n");
    try out.append("  \"dynamic_smem_bytes\": 0,\n");
    try out.append("  \"argument_slots\": 3\n");
    try out.append("}\n");
}

pub fn writeCompilePlanForExample(kind: ExampleKind, out: anytype) Error!void {
    const stem = kind.stem();
    const input_mlir = switch (kind) {
        .layout_demo => "testdata/golden/examples/layout_demo.mlir",
        .tensor_demo => "testdata/golden/examples/tensor_demo.mlir",
        .copy_demo => "testdata/golden/examples/copy_demo.mlir",
        .mma_demo => "testdata/golden/examples/mma_demo.mlir",
        .gemm_skeleton => "testdata/golden/examples/gemm_skeleton.mlir",
    };
    try out.append("cute-opt --pass-pipeline=\"");
    try out.append("builtin.module(cute-to-nvvm{cubin-format=bin enable-cuda-dialect=true cuda-dialect-external-module=true opt-level=3 cubin-chip='sm_90' dump-cubin-path='zig-out/not-cute/");
    try out.append(stem);
    try out.append("' })\" ");
    try out.append(input_mlir);
    try out.append(" -o zig-out/not-cute/examples.cubin");
}

pub fn expectContains(haystack: []const u8, needle: []const u8) Error!void {
    if (std.mem.indexOf(u8, haystack, needle) == null) return Error.GoldenMismatch;
}

pub fn expectExampleContains(kind: ExampleKind, needle: []const u8) Error!void {
    const text = exampleText(kind);
    if (std.mem.indexOf(u8, text, needle) == null) return Error.GoldenMismatch;
}

pub fn expectedExampleHash(kind: ExampleKind) u64 {
    return std.hash.Wyhash.hash(0, exampleText(kind));
}

pub fn sourceFindingsSummary(out: anytype) Error!void {
    try out.append("CuTe DSL loads cutlass._mlir._mlir_libs._cutlass_ir and calls populate(); ");
    try out.append("compile() parses pipelines with cutlass._mlir.passmanager.PassManager; ");
    try out.append("JIT uses cutlass._mlir.execution_engine.ExecutionEngine. ");
    try out.append("The visible Python tree builds cute-to-nvvm/lir-to-cute pipeline strings; ");
    try out.append("the pass definitions are in the packaged native _cutlass_ir extension, not Python source.\n");
}

pub const layout_demo_mlir =
    \\module {
    \\  func.func @layout_demo() {
    \\    %0 = cute.make_shape() : () -> !cute.shape<"(2,3)">
    \\    %1 = cute.make_stride() : () -> !cute.stride<"(3,1)">
    \\    %2 = cute.make_layout(%0, %1) : !cute.layout<"(2,3):(3,1)">
    \\    return
    \\  }
    \\}
    \\
;

pub const tensor_demo_mlir =
    \\module {
    \\  func.func @tensor_case(%arg0: !cute.memref<f32, gmem, align<16>, "(4):(1)">, %arg1: vector<4xf32>) {
    \\    %0 = cute.memref.load_vec(%arg0) : (!cute.memref<f32, gmem, align<16>, "(4):(1)">) -> vector<4xf32>
    \\    %1 = arith.addf %0, %arg1 : vector<4xf32>
    \\    cute.memref.store_vec(%1, %arg0) : (vector<4xf32>, !cute.memref<f32, gmem, align<16>, "(4):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const copy_demo_mlir =
    \\module {
    \\  func.func @copy_case(%arg0: !cute.memref<f32, gmem, align<16>, "(1):(1)">, %arg1: !cute.memref<f32, gmem, align<16>, "(1):(1)">) {
    \\    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>
    \\    cute.copy_atom_call(%atom, %arg0, %arg1) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !cute.memref<f32, gmem, align<16>, "(1):(1)">, !cute.memref<f32, gmem, align<16>, "(1):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const mma_demo_mlir =
    \\module {
    \\  func.func @mma_case(%arg0: !cute.memref<f32, generic, "(1):(1)">, %arg1: !cute.memref<f32, generic, "(1):(1)">, %arg2: !cute.memref<f32, generic, "(1):(1)">, %arg3: !cute.memref<f32, generic, "(1):(1)">) {
    \\    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >
    \\    cute.mma_atom_call(%atom, %arg3, %arg0, %arg1, %arg2) : (!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, !cute.memref<f32, generic, "(1):(1)">, !cute.memref<f32, generic, "(1):(1)">, !cute.memref<f32, generic, "(1):(1)">, !cute.memref<f32, generic, "(1):(1)">) -> ()
    \\    return
    \\  }
    \\}
    \\
;

pub const gemm_skeleton_mlir =
    \\!copy_simt = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 32 b>, layout_copy_tv = <"(1,1):(1,1)">, tiler_mn = <"[1:0;1:0]">>
    \\!mma_f32_1x1x1 = !cute.tiled_mma<!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, atom_layout_MNK = <"(1,1,1):(1,1,1)">>
    \\!memref_gmem_f32_1x1 = !cute.memref<f32, gmem, "(1,1):(1,1)">
    \\!memref_gmem_f32_partition = !cute.memref<f32, gmem, "((1,1),1,1):((0,0),0,0)">
    \\!memref_generic_f32_1x1 = !cute.memref<f32, generic, "(1,1):(1,1)">
    \\!memref_generic_f32_frag = !cute.memref<f32, generic, "(1,1,1):(0,0,0)">
    \\module {
    \\  func.func @gemm_skeleton(%arg0: !memref_gmem_f32_1x1, %arg1: !memref_gmem_f32_1x1, %arg2: !memref_generic_f32_1x1, %arg3: !memref_generic_f32_1x1, %arg4: !memref_generic_f32_1x1, %arg5: !memref_generic_f32_1x1, %arg6: !cute.coord<"0">) {
    \\    %copy_atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>
    \\    %copy_tiled = cute.make_tiled_copy(%copy_atom) : !copy_simt
    \\    %src_partitioned = cute.tiled.copy.partition_S(%copy_tiled, %arg0, %arg6) : (!copy_simt, !memref_gmem_f32_1x1, !cute.coord<"0">) -> !memref_gmem_f32_partition
    \\    %dst_partitioned = cute.tiled.copy.partition_D(%copy_tiled, %arg1, %arg6) : (!copy_simt, !memref_gmem_f32_1x1, !cute.coord<"0">) -> !memref_gmem_f32_partition
    \\    cute.copy(%copy_tiled, %src_partitioned, %dst_partitioned) : (!copy_simt, !memref_gmem_f32_partition, !memref_gmem_f32_partition)
    \\    %mma_atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >
    \\    %mma_tiled = cute.make_tiled_mma(%mma_atom) : !mma_f32_1x1x1
    \\    %a = cute.tiled.mma.partition A(%mma_tiled, %arg2, %arg6) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    \\    %b = cute.tiled.mma.partition B(%mma_tiled, %arg3, %arg6) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    \\    %c = cute.tiled.mma.partition C(%mma_tiled, %arg4, %arg6) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    \\    %d = cute.tiled.mma.partition C(%mma_tiled, %arg5, %arg6) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    \\    cute.gemm(%mma_tiled, %d, %a, %b, %c) : (!mma_f32_1x1x1, !memref_generic_f32_frag, !memref_generic_f32_frag, !memref_generic_f32_frag, !memref_generic_f32_frag)
    \\    return
    \\  }
    \\}
    \\
;

test "examples_api: every example has balanced MLIR text" {
    for (all_examples) |kind| {
        const text = exampleText(kind);
        try mlir.validateBalancedText(text);
        try expectContains(text, "module");
    }
}

test "examples_api: gemm skeleton composes copy and mma hooks" {
    try expectExampleContains(.gemm_skeleton, "cute.tiled.copy.partition_S");
    try expectExampleContains(.gemm_skeleton, "cute.gemm");
    try expectExampleContains(.gemm_skeleton, "cute.copy(%copy_tiled");
}

test "examples_api: runtime plan for example is deterministic" {
    var out: mlir.TextBuffer(2048) = .{};
    try writeRuntimePlanForExample(.gemm_skeleton, &out);
    try expectContains(out.slice(), "\"kernel\": \"gemm_skeleton\"");
    try expectContains(out.slice(), "\"argument_slots\": 3");
}

test "examples_api: compile plan uses cute-to-nvvm pipeline" {
    var out: mlir.TextBuffer(4096) = .{};
    try writeCompilePlanForExample(.gemm_skeleton, &out);
    try expectContains(out.slice(), "cute-to-nvvm");
    try expectContains(out.slice(), "gemm_skeleton");
}

test "examples_api: source findings document native pass container" {
    var out: mlir.TextBuffer(1024) = .{};
    try sourceFindingsSummary(&out);
    try expectContains(out.slice(), "_cutlass_ir");
    try expectContains(out.slice(), "PassManager");
}
