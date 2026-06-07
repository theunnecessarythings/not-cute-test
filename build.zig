const std = @import("std");

const ExampleSpec = struct {
    source: []const u8,
    name: []const u8,
};

fn addExamples(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Step.Options,
    library_module: *std.Build.Module,
    parent_step: *std.Build.Step,
    specs: []const ExampleSpec,
) void {
    for (specs) |spec| {
        const module = b.createModule(.{
            .root_source_file = b.path(spec.source),
            .target = target,
            .optimize = optimize,
        });
        module.addOptions("build_options", build_options);
        module.addImport("not_cute", library_module);

        const executable = b.addExecutable(.{
            .name = spec.name,
            .root_module = module,
        });
        parent_step.dependOn(&b.addInstallArtifact(executable, .{}).step);
    }
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const enable_mlir_tools = b.option(
        bool,
        "mlir-tools",
        "Run external MLIR verifier tests",
    ) orelse false;
    const assume_mlir_tools_present = b.option(
        bool,
        "assume-mlir-tools-present",
        "Assume cute-opt/mlir-opt/FileCheck are available when verifier tests request them",
    ) orelse false;
    const cute_opt_path = b.option(
        []const u8,
        "cute-opt",
        "Path to cute-opt",
    ) orelse "cute-opt";
    const mlir_opt_path = b.option(
        []const u8,
        "mlir-opt",
        "Path to mlir-opt",
    ) orelse "mlir-opt";
    const filecheck_path = b.option(
        []const u8,
        "filecheck",
        "Path to FileCheck",
    ) orelse "FileCheck";
    const cutlass_python_path = b.option(
        []const u8,
        "cutlass-python",
        "Python executable with nvidia-cutlass-dsl/cutlass installed",
    ) orelse "python3";
    const cutlass_bridge_script = b.option(
        []const u8,
        "cutlass-bridge-script",
        "Path to tools/cutlass_mlir_bridge.py",
    ) orelse "tools/cutlass_mlir_bridge.py";
    const cutlass_pipeline = b.option(
        []const u8,
        "cutlass-pipeline",
        "CUTLASS MLIR pass pipeline for parser-aligned verify-cutlass fixtures",
    ) orelse "builtin.module(canonicalize)";

    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_mlir_tools", enable_mlir_tools);
    build_options.addOption(bool, "assume_mlir_tools_present", assume_mlir_tools_present);
    build_options.addOption([]const u8, "cute_opt_path", cute_opt_path);
    build_options.addOption([]const u8, "mlir_opt_path", mlir_opt_path);
    build_options.addOption([]const u8, "filecheck_path", filecheck_path);

    const mod = b.addModule("not_cute", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addOptions("build_options", build_options);

    const lib = b.addLibrary(.{
        .name = "not_cute",
        .root_module = mod,
        .linkage = .static,
    });
    b.installArtifact(lib);

    const examples_step = b.step(
        "examples",
        "Build and install standalone API examples",
    );

    addExamples(b, target, optimize, build_options, mod, examples_step, &.{
        .{ .source = "examples/layout_demo.zig", .name = "layout_demo" },
        .{ .source = "examples/tensor_demo.zig", .name = "tensor_demo" },
        .{ .source = "examples/copy_demo.zig", .name = "copy_demo" },
        .{ .source = "examples/mma_demo.zig", .name = "mma_demo" },
        .{ .source = "examples/kernel_plan.zig", .name = "kernel_plan" },
    });

    const launch_module = b.createModule(.{
        .root_source_file = b.path("tools/launch.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    launch_module.addImport("not_cute", mod);
    const launch_executable = b.addExecutable(.{
        .name = "not-cute-launch",
        .root_module = launch_module,
    });
    const launch_install = b.addInstallArtifact(launch_executable, .{});
    const launch_step = b.step("launch", "Build and install the CUDA launch tool");
    launch_step.dependOn(&launch_install.step);

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_mod.addOptions("build_options", build_options);
    const unit_tests = b.addTest(.{ .root_module = test_mod });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run not-cute unit tests");
    test_step.dependOn(&run_unit_tests.step);
    const verify_mlir_step = b.step(
        "verify-mlir",
        "Run external cute-opt verifier checks over generated golden MLIR",
    );
    const verifier_cases = [_][]const u8{
        "src/testdata/golden/layout_case.mlir",
        "src/testdata/golden/tensor_case.mlir",
        "src/testdata/golden/copy_case.mlir",
        "src/testdata/golden/mma_case.mlir",
    };
    for (verifier_cases) |case_path| {
        const run_verify = b.addSystemCommand(&.{
            cute_opt_path,
            "--verify-diagnostics",
            case_path,
        });
        verify_mlir_step.dependOn(&run_verify.step);
    }

    const verify_cutlass_step = b.step(
        "verify-cutlass",
        "Run CUTLASS DSL bridge verifier over parser-aligned fixtures",
    );
    const cutlass_cases = [_][]const u8{
        "testdata/cutlass/builtin_case.mlir",
        "testdata/cutlass/layout_case.mlir",
        "testdata/cutlass/identity_tensor_case.mlir",
        "testdata/cutlass/memref_load_case.mlir",
        "testdata/cutlass/cutlass_emit_tensor_scalar.mlir",
        "testdata/cutlass/cutlass_emit_tensor_vector.mlir",
        "testdata/cutlass/cutlass_emit_copy_atom.mlir",
        "testdata/cutlass/cutlass_emit_tiled_copy.mlir",
        "testdata/cutlass/cutlass_emit_mma_atom.mlir",
        "testdata/cutlass/cutlass_routed_tensor_vector.mlir",
        "testdata/cutlass/cutlass_routed_copy_atom.mlir",
        "testdata/cutlass/cutlass_routed_mma_atom.mlir",
        "testdata/cutlass/tiled_emit_full_tiled_copy.mlir",
        "testdata/cutlass/tiled_emit_full_tiled_mma.mlir",
    };
    for (cutlass_cases) |case_path| {
        const run_bridge_verify = b.addSystemCommand(&.{
            cutlass_python_path,
            cutlass_bridge_script,
            "verify",
            "--input",
            case_path,
            "--pipeline",
            cutlass_pipeline,
            "--enable-verifier",
        });
        verify_cutlass_step.dependOn(&run_bridge_verify.step);
    }
    const run_bridge_negative = b.addSystemCommand(&.{
        cutlass_python_path,
        cutlass_bridge_script,
        "expect-fail",
        "--input",
        "testdata/cutlass/negative_fake_tensor.mlir",
        "--expected",
        "unknown  type `tensor` in dialect `cute`",
    });
    verify_cutlass_step.dependOn(&run_bridge_negative.step);

    const verify_cutlass_tensor_step = b.step(
        "verify-cutlass-tensor",
        "Run CUTLASS parser checks for tensor/default examples",
    );
    for ([_][]const u8{
        "testdata/cutlass/memref_load_case.mlir",
        "testdata/cutlass/cutlass_routed_tensor_vector.mlir",
        "testdata/golden/tensor_case.mlir",
        "testdata/golden/examples/tensor_demo.mlir",
    }) |case_path| {
        const run = b.addSystemCommand(&.{
            cutlass_python_path,
            cutlass_bridge_script,
            "verify",
            "--input",
            case_path,
            "--pipeline",
            cutlass_pipeline,
            "--enable-verifier",
        });
        verify_cutlass_tensor_step.dependOn(&run.step);
    }

    const verify_cutlass_copy_step = b.step(
        "verify-cutlass-copy",
        "Run CUTLASS parser checks for copy examples",
    );
    for ([_][]const u8{
        "testdata/cutlass/cutlass_routed_copy_atom.mlir",
        "testdata/cutlass/tiled_emit_full_tiled_copy.mlir",
        "testdata/golden/copy_case.mlir",
        "testdata/golden/examples/copy_demo.mlir",
    }) |case_path| {
        const run = b.addSystemCommand(&.{
            cutlass_python_path,
            cutlass_bridge_script,
            "verify",
            "--input",
            case_path,
            "--pipeline",
            cutlass_pipeline,
            "--enable-verifier",
        });
        verify_cutlass_copy_step.dependOn(&run.step);
    }

    const verify_cutlass_mma_step = b.step(
        "verify-cutlass-mma",
        "Run CUTLASS parser checks for MMA/GEMM examples",
    );
    for ([_][]const u8{
        "testdata/cutlass/cutlass_routed_mma_atom.mlir",
        "testdata/cutlass/tiled_emit_full_tiled_mma.mlir",
        "testdata/golden/mma_case.mlir",
        "testdata/golden/examples/mma_demo.mlir",
        "testdata/golden/examples/gemm_skeleton.mlir",
    }) |case_path| {
        const run = b.addSystemCommand(&.{
            cutlass_python_path,
            cutlass_bridge_script,
            "verify",
            "--input",
            case_path,
            "--pipeline",
            cutlass_pipeline,
            "--enable-verifier",
        });
        verify_cutlass_mma_step.dependOn(&run.step);
    }

    const verify_cutlass_negative_step = b.step(
        "verify-cutlass-negative",
        "Run expected-failure CUTLASS parser checks",
    );
    verify_cutlass_negative_step.dependOn(&run_bridge_negative.step);

    const verify_cutlass_parse_step = b.step(
        "verify-cutlass-parse",
        "Run CUTLASS parser-only checks by shard",
    );
    for ([_][]const u8{
        "testdata/cutlass/layout_case.mlir",
        "testdata/cutlass/identity_tensor_case.mlir",
        "testdata/cutlass/memref_load_case.mlir",
        "testdata/cutlass/cutlass_routed_tensor_vector.mlir",
        "testdata/cutlass/cutlass_routed_copy_atom.mlir",
        "testdata/cutlass/cutlass_routed_mma_atom.mlir",
        "testdata/cutlass/tiled_emit_full_tiled_copy.mlir",
        "testdata/cutlass/tiled_emit_full_tiled_mma.mlir",
        "testdata/cutlass/kernel_tiled_copy.mlir",
    }) |case_path| {
        const run = b.addSystemCommand(&.{
            cutlass_python_path,
            cutlass_bridge_script,
            "parse",
            "--input",
            case_path,
        });
        verify_cutlass_parse_step.dependOn(&run.step);
    }

    const verify_cutlass_pipeline_step = b.step(
        "verify-cutlass-pipeline",
        "Run sharded CUTLASS canonicalization/pipeline checks",
    );
    for ([_][]const u8{
        "testdata/cutlass/cutlass_routed_tensor_vector.mlir",
        "testdata/cutlass/cutlass_routed_copy_atom.mlir",
        "testdata/cutlass/cutlass_routed_mma_atom.mlir",
        "testdata/cutlass/tiled_emit_full_tiled_copy.mlir",
        "testdata/cutlass/tiled_emit_full_tiled_mma.mlir",
    }) |case_path| {
        const run = b.addSystemCommand(&.{
            cutlass_python_path,
            cutlass_bridge_script,
            "verify",
            "--input",
            case_path,
            "--pipeline",
            "builtin.module(canonicalize)",
            "--enable-verifier",
        });
        verify_cutlass_pipeline_step.dependOn(&run.step);
    }

    const verify_cutlass_kernel_cubin_step = b.step(
        "verify-cutlass-kernel-cubin",
        "Run cute-to-nvvm over a kernel-shaped CUTLASS fixture and require a dumped CUBIN",
    );
    const run_kernel_cubin = b.addSystemCommand(&.{
        cutlass_python_path,
        cutlass_bridge_script,
        "compile-artifact",
        "--input",
        "testdata/cutlass/kernel_tiled_copy.mlir",
        "--work-dir",
        "zig-cache/not-cute-artifacts/kernel_tiled_copy",
        "--function",
        "tiled_copy_kernel",
        "--pipeline",
        "builtin.module(cute-to-nvvm{cubin-format=bin cubin-chip='sm_90' dump-cubin-path='zig-cache/not-cute-artifacts/kernel_tiled_copy/tiled_copy_kernel' preserve-line-info=true})",
        "--enable-verifier",
        "--expect-cubin",
    });
    verify_cutlass_kernel_cubin_step.dependOn(&run_kernel_cubin.step);

    const verify_kernel_builders_parse_step = b.step(
        "verify-kernel-builders-parse",
        "Run CUTLASS parser checks for generated kernel-builder fixtures",
    );
    for ([_][]const u8{
        "testdata/cutlass/kernel_builders/copy_kernel.mlir",
        "testdata/cutlass/kernel_builders/vector_copy_kernel.mlir",
        "testdata/cutlass/kernel_builders/tiled_copy_kernel.mlir",
        "testdata/cutlass/kernel_builders/mma_microkernel.mlir",
        "testdata/cutlass/kernel_builders/gemm_mainloop.mlir",
        "testdata/cutlass/kernel_builders/epilogue_kernel.mlir",
        "testdata/cutlass/kernel_builders/sm80_gemm_kernel.mlir",
        "testdata/cutlass/kernel_builders/sm90_tma_wgmma_kernel.mlir",
        "testdata/cutlass/kernel_builders/sm100_tcgen05_kernel.mlir",
    }) |case_path| {
        const run = b.addSystemCommand(&.{
            cutlass_python_path,
            cutlass_bridge_script,
            "parse",
            "--input",
            case_path,
        });
        verify_kernel_builders_parse_step.dependOn(&run.step);
    }
}
