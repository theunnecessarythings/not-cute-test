const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const enable_mlir_tools = b.option(bool, "mlir-tools", "Run external MLIR verifier tests") orelse false;
    const assume_mlir_tools_present = b.option(bool, "assume-mlir-tools-present", "Assume cute-opt/mlir-opt/FileCheck are available when verifier tests request them") orelse false;
    const cute_opt_path = b.option([]const u8, "cute-opt", "Path to cute-opt") orelse "cute-opt";
    const mlir_opt_path = b.option([]const u8, "mlir-opt", "Path to mlir-opt") orelse "mlir-opt";
    const filecheck_path = b.option([]const u8, "filecheck", "Path to FileCheck") orelse "FileCheck";
    const cutlass_python_path = b.option([]const u8, "cutlass-python", "Python executable with nvidia-cutlass-dsl/cutlass installed") orelse "python3";
    const cutlass_bridge_script = b.option([]const u8, "cutlass-bridge-script", "Path to tools/cutlass_mlir_bridge.py") orelse "tools/cutlass_mlir_bridge.py";
    const cutlass_pipeline = b.option([]const u8, "cutlass-pipeline", "CUTLASS MLIR pass pipeline for parser-aligned verify-cutlass fixtures") orelse "builtin.module(canonicalize)";

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

    const cli_mod = b.createModule(.{
        .root_source_file = b.path("src/mlir_harness_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    cli_mod.addOptions("build_options", build_options);
    const cli = b.addExecutable(.{
        .name = "not-cute-mlir-harness",
        .root_module = cli_mod,
    });
    const install_cli = b.addInstallArtifact(cli, .{});
    const harness_step = b.step("harness", "Build and install the MLIR harness CLI");
    harness_step.dependOn(&install_cli.step);

    const runtime_plan_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/runtime_plan_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    runtime_plan_cli_mod.addOptions("build_options", build_options);
    const runtime_plan_cli = b.addExecutable(.{
        .name = "not-cute-runtime-plan",
        .root_module = runtime_plan_cli_mod,
    });
    const install_runtime_plan_cli = b.addInstallArtifact(runtime_plan_cli, .{});
    const runtime_plan_step = b.step("runtime-plan", "Build and install the runtime/export planning CLI");
    runtime_plan_step.dependOn(&install_runtime_plan_cli.step);

    const examples_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/examples_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    examples_cli_mod.addOptions("build_options", build_options);
    const examples_cli = b.addExecutable(.{
        .name = "not-cute-examples",
        .root_module = examples_cli_mod,
    });
    const install_examples_cli = b.addInstallArtifact(examples_cli, .{});
    const examples_step = b.step("examples", "Build and install example CLI and standalone examples");
    examples_step.dependOn(&install_examples_cli.step);

    const cutlass_bridge_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/cutlass_bridge_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    cutlass_bridge_cli_mod.addOptions("build_options", build_options);
    const cutlass_bridge_cli = b.addExecutable(.{
        .name = "not-cute-cutlass-bridge",
        .root_module = cutlass_bridge_cli_mod,
    });
    const install_cutlass_bridge_cli = b.addInstallArtifact(cutlass_bridge_cli, .{});
    const cutlass_bridge_step = b.step("cutlass-bridge", "Build and install the CUTLASS MLIR bridge CLI");
    cutlass_bridge_step.dependOn(&install_cutlass_bridge_cli.step);

    const cutlass_fixtures_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/cutlass_fixtures_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    cutlass_fixtures_cli_mod.addOptions("build_options", build_options);
    const cutlass_fixtures_cli = b.addExecutable(.{
        .name = "not-cute-cutlass-fixtures",
        .root_module = cutlass_fixtures_cli_mod,
    });
    const install_cutlass_fixtures_cli = b.addInstallArtifact(cutlass_fixtures_cli, .{});
    const cutlass_fixtures_step = b.step("cutlass-fixtures", "Build and install the CUTLASS parser fixture CLI");
    cutlass_fixtures_step.dependOn(&install_cutlass_fixtures_cli.step);

    const cutlass_emit_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/cutlass_emit_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    cutlass_emit_cli_mod.addOptions("build_options", build_options);
    const cutlass_emit_cli = b.addExecutable(.{
        .name = "not-cute-cutlass-emission",
        .root_module = cutlass_emit_cli_mod,
    });
    const install_cutlass_emit_cli = b.addInstallArtifact(cutlass_emit_cli, .{});
    const cutlass_emission_step = b.step("cutlass-emission", "Build and install the CUTLASS parser-aligned emission CLI");
    cutlass_emission_step.dependOn(&install_cutlass_emit_cli.step);

    const cutlass_routed_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/cutlass_routed_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    cutlass_routed_cli_mod.addOptions("build_options", build_options);
    const cutlass_routed_cli = b.addExecutable(.{
        .name = "not-cute-cutlass-routed",
        .root_module = cutlass_routed_cli_mod,
    });
    const install_cutlass_routed_cli = b.addInstallArtifact(cutlass_routed_cli, .{});
    const cutlass_routed_step = b.step("cutlass-routed", "Build and install the routed CUTLASS emission CLI");
    cutlass_routed_step.dependOn(&install_cutlass_routed_cli.step);

    const tiled_emit_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/tiled_emit_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    tiled_emit_cli_mod.addOptions("build_options", build_options);
    const tiled_emit_cli = b.addExecutable(.{
        .name = "not-cute-cutlass-full-tiled",
        .root_module = tiled_emit_cli_mod,
    });
    const install_tiled_emit_cli = b.addInstallArtifact(tiled_emit_cli, .{});
    const cutlass_full_tiled_step = b.step("cutlass-full-tiled", "Build and install the full tiled copy/MMA CUTLASS fixture CLI");
    cutlass_full_tiled_step.dependOn(&install_tiled_emit_cli.step);

    const api_audit_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/api_audit_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    api_audit_cli_mod.addOptions("build_options", build_options);
    const api_audit_cli = b.addExecutable(.{
        .name = "not-cute-api-audit",
        .root_module = api_audit_cli_mod,
    });
    const install_api_audit_cli = b.addInstallArtifact(api_audit_cli, .{});
    const api_audit_step = b.step("api-audit", "Build and install the API/architecture audit CLI");
    api_audit_step.dependOn(&install_api_audit_cli.step);

    const execution_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/execution_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    execution_cli_mod.addOptions("build_options", build_options);
    const execution_cli = b.addExecutable(.{
        .name = "not-cute-exec",
        .root_module = execution_cli_mod,
    });
    const install_execution_cli = b.addInstallArtifact(execution_cli, .{});
    const exec_step = b.step("exec", "Build and install the CUDA execution wiring CLI");
    exec_step.dependOn(&install_execution_cli.step);

    const compile_pipeline_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/compile_pipeline_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    compile_pipeline_cli_mod.addOptions("build_options", build_options);
    const compile_pipeline_cli = b.addExecutable(.{
        .name = "not-cute-compile-pipeline",
        .root_module = compile_pipeline_cli_mod,
    });
    const install_compile_pipeline_cli = b.addInstallArtifact(compile_pipeline_cli, .{});
    const compile_pipeline_step = b.step("compile-pipeline", "Build and install the CUTLASS compile-pipeline planning CLI");
    compile_pipeline_step.dependOn(&install_compile_pipeline_cli.step);

    const pipeline_verify_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/pipeline_verify_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    pipeline_verify_cli_mod.addOptions("build_options", build_options);
    const pipeline_verify_cli = b.addExecutable(.{
        .name = "not-cute-pipeline-verify",
        .root_module = pipeline_verify_cli_mod,
    });
    const install_pipeline_verify_cli = b.addInstallArtifact(pipeline_verify_cli, .{});
    const pipeline_verify_step = b.step("pipeline-verify", "Build and install the sharded CUTLASS verifier CLI");
    pipeline_verify_step.dependOn(&install_pipeline_verify_cli.step);

    const integration_audit_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/integration_audit_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    integration_audit_cli_mod.addOptions("build_options", build_options);
    const integration_audit_cli = b.addExecutable(.{
        .name = "not-cute-audit",
        .root_module = integration_audit_cli_mod,
    });
    const install_integration_audit_cli = b.addInstallArtifact(integration_audit_cli, .{});
    const audit_step = b.step("audit", "Build and install the integration audit CLI");
    audit_step.dependOn(&install_integration_audit_cli.step);

    const launch_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/launch_cli.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    launch_cli_mod.addOptions("build_options", build_options);
    const launch_cli = b.addExecutable(.{
        .name = "not-cute-launch",
        .root_module = launch_cli_mod,
    });
    const install_launch_cli = b.addInstallArtifact(launch_cli, .{});
    const launch_step = b.step("launch", "Build and install the CUDA launch CLI");
    launch_step.dependOn(&install_launch_cli.step);

    const example_sources = [_][]const u8{
        "examples/layout_demo.zig",
        "examples/tensor_demo.zig",
        "examples/copy_demo.zig",
        "examples/mma_demo.zig",
        "examples/gemm_skeleton.zig",
    };
    const example_names = [_][]const u8{
        "layout_demo",
        "tensor_demo",
        "copy_demo",
        "mma_demo",
        "gemm_skeleton",
    };
    for (example_sources, example_names) |source, name| {
        const ex_mod = b.createModule(.{
            .root_source_file = b.path(source),
            .target = target,
            .optimize = optimize,
        });
        ex_mod.addOptions("build_options", build_options);
        ex_mod.addImport("not_cute", mod);
        const ex = b.addExecutable(.{ .name = name, .root_module = ex_mod });
        examples_step.dependOn(&b.addInstallArtifact(ex, .{}).step);
    }

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
    const verify_mlir_step = b.step("verify-mlir", "Run external cute-opt verifier checks over generated golden MLIR");
    const verifier_cases = [_][]const u8{
        "src/testdata/golden/layout_case.mlir",
        "src/testdata/golden/tensor_case.mlir",
        "src/testdata/golden/copy_case.mlir",
        "src/testdata/golden/mma_case.mlir",
    };
    for (verifier_cases) |case_path| {
        const run_verify = b.addSystemCommand(&.{ cute_opt_path, "--verify-diagnostics", case_path });
        verify_mlir_step.dependOn(&run_verify.step);
    }

    const verify_cutlass_step = b.step("verify-cutlass", "Run CUTLASS DSL bridge verifier over parser-aligned fixtures");
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

    const verify_cutlass_tensor_step = b.step("verify-cutlass-tensor", "Run CUTLASS parser checks for tensor/default examples");
    for ([_][]const u8{
        "testdata/cutlass/memref_load_case.mlir",
        "testdata/cutlass/cutlass_routed_tensor_vector.mlir",
        "testdata/golden/tensor_case.mlir",
        "testdata/golden/examples/tensor_demo.mlir",
    }) |case_path| {
        const run = b.addSystemCommand(&.{ cutlass_python_path, cutlass_bridge_script, "verify", "--input", case_path, "--pipeline", cutlass_pipeline, "--enable-verifier" });
        verify_cutlass_tensor_step.dependOn(&run.step);
    }

    const verify_cutlass_copy_step = b.step("verify-cutlass-copy", "Run CUTLASS parser checks for copy examples");
    for ([_][]const u8{
        "testdata/cutlass/cutlass_routed_copy_atom.mlir",
        "testdata/cutlass/tiled_emit_full_tiled_copy.mlir",
        "testdata/golden/copy_case.mlir",
        "testdata/golden/examples/copy_demo.mlir",
    }) |case_path| {
        const run = b.addSystemCommand(&.{ cutlass_python_path, cutlass_bridge_script, "verify", "--input", case_path, "--pipeline", cutlass_pipeline, "--enable-verifier" });
        verify_cutlass_copy_step.dependOn(&run.step);
    }

    const verify_cutlass_mma_step = b.step("verify-cutlass-mma", "Run CUTLASS parser checks for MMA/GEMM examples");
    for ([_][]const u8{
        "testdata/cutlass/cutlass_routed_mma_atom.mlir",
        "testdata/cutlass/tiled_emit_full_tiled_mma.mlir",
        "testdata/golden/mma_case.mlir",
        "testdata/golden/examples/mma_demo.mlir",
        "testdata/golden/examples/gemm_skeleton.mlir",
    }) |case_path| {
        const run = b.addSystemCommand(&.{ cutlass_python_path, cutlass_bridge_script, "verify", "--input", case_path, "--pipeline", cutlass_pipeline, "--enable-verifier" });
        verify_cutlass_mma_step.dependOn(&run.step);
    }

    const verify_cutlass_negative_step = b.step("verify-cutlass-negative", "Run expected-failure CUTLASS parser checks");
    verify_cutlass_negative_step.dependOn(&run_bridge_negative.step);

    const verify_cutlass_parse_step = b.step("verify-cutlass-parse", "Run CUTLASS parser-only checks by shard");
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
        const run = b.addSystemCommand(&.{ cutlass_python_path, cutlass_bridge_script, "parse", "--input", case_path });
        verify_cutlass_parse_step.dependOn(&run.step);
    }

    const verify_cutlass_pipeline_step = b.step("verify-cutlass-pipeline", "Run sharded CUTLASS canonicalization/pipeline checks");
    for ([_][]const u8{
        "testdata/cutlass/cutlass_routed_tensor_vector.mlir",
        "testdata/cutlass/cutlass_routed_copy_atom.mlir",
        "testdata/cutlass/cutlass_routed_mma_atom.mlir",
        "testdata/cutlass/tiled_emit_full_tiled_copy.mlir",
        "testdata/cutlass/tiled_emit_full_tiled_mma.mlir",
    }) |case_path| {
        const run = b.addSystemCommand(&.{ cutlass_python_path, cutlass_bridge_script, "verify", "--input", case_path, "--pipeline", "builtin.module(canonicalize)", "--enable-verifier" });
        verify_cutlass_pipeline_step.dependOn(&run.step);
    }

    const verify_cutlass_kernel_cubin_step = b.step("verify-cutlass-kernel-cubin", "Run cute-to-nvvm over a kernel-shaped CUTLASS fixture and require a dumped CUBIN");
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

    const compile_artifact_plan_step = b.step("compile-artifact-plan", "Build compile pipeline CLI and print artifact plan command");
    compile_artifact_plan_step.dependOn(&install_compile_pipeline_cli.step);

    const kernel_builders_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/kernel_builders_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    kernel_builders_cli_mod.addOptions("build_options", build_options);
    const kernel_builders_cli = b.addExecutable(.{
        .name = "not-cute-kernel-builders",
        .root_module = kernel_builders_cli_mod,
    });
    const install_kernel_builders_cli = b.addInstallArtifact(kernel_builders_cli, .{});
    const kernel_builders_step = b.step("kernel-builders", "Build and install kernel builder CLI");
    kernel_builders_step.dependOn(&install_kernel_builders_cli.step);

    const memory_model_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/memory_model_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    memory_model_cli_mod.addOptions("build_options", build_options);
    const memory_model_cli = b.addExecutable(.{
        .name = "not-cute-memory-model",
        .root_module = memory_model_cli_mod,
    });
    const install_memory_model_cli = b.addInstallArtifact(memory_model_cli, .{});
    const memory_model_step = b.step("memory-model", "Build and install memory model CLI");
    memory_model_step.dependOn(&install_memory_model_cli.step);

    const verify_kernel_builders_parse_step = b.step("verify-kernel-builders-parse", "Run CUTLASS parser checks for generated kernel-builder fixtures");
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
        const run = b.addSystemCommand(&.{ cutlass_python_path, cutlass_bridge_script, "parse", "--input", case_path });
        verify_kernel_builders_parse_step.dependOn(&run.step);
    }

    const upstream_parity_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/upstream_parity_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    upstream_parity_cli_mod.addOptions("build_options", build_options);
    const upstream_parity_cli = b.addExecutable(.{
        .name = "not-cute-upstream-parity",
        .root_module = upstream_parity_cli_mod,
    });
    const install_upstream_parity_cli = b.addInstallArtifact(upstream_parity_cli, .{});
    const upstream_parity_step = b.step("upstream-parity", "Build and install upstream CuTeDSL example parity CLI");
    upstream_parity_step.dependOn(&install_upstream_parity_cli.step);

    const upstream_example_sources = [_][]const u8{
        "examples/upstream/hello_world.zig",
        "examples/upstream/print_values.zig",
        "examples/upstream/data_types.zig",
        "examples/upstream/layout_algebra.zig",
        "examples/upstream/tensor.zig",
        "examples/upstream/tensorssa.zig",
        "examples/upstream/elementwise_add.zig",
        "examples/upstream/cuda_graphs.zig",
        "examples/upstream/ffi_tensor.zig",
    };
    const upstream_example_names = [_][]const u8{
        "upstream_hello_world",
        "upstream_print_values",
        "upstream_data_types",
        "upstream_layout_algebra",
        "upstream_tensor",
        "upstream_tensorssa",
        "upstream_elementwise_add",
        "upstream_cuda_graphs",
        "upstream_ffi_tensor",
    };
    for (upstream_example_sources, upstream_example_names) |source, name| {
        const ex_mod = b.createModule(.{
            .root_source_file = b.path(source),
            .target = target,
            .optimize = optimize,
        });
        ex_mod.addOptions("build_options", build_options);
        ex_mod.addImport("not_cute", mod);
        const ex = b.addExecutable(.{ .name = name, .root_module = ex_mod });
        upstream_parity_step.dependOn(&b.addInstallArtifact(ex, .{}).step);
    }

    const verify_upstream_parity_parse_step = b.step("verify-upstream-parity-parse", "Run CUTLASS parser checks for upstream parity golden MLIR");
    for ([_][]const u8{
        "testdata/golden/upstream/hello_world.mlir",
        "testdata/golden/upstream/print_values.mlir",
        "testdata/golden/upstream/data_types.mlir",
        "testdata/golden/upstream/layout_algebra.mlir",
        "testdata/golden/upstream/tensor.mlir",
        "testdata/golden/upstream/tensorssa.mlir",
        "testdata/golden/upstream/elementwise_add.mlir",
        "testdata/golden/upstream/cuda_graphs.mlir",
        "testdata/golden/upstream/ffi_tensor.mlir",
    }) |case_path| {
        const run = b.addSystemCommand(&.{ cutlass_python_path, cutlass_bridge_script, "parse", "--input", case_path });
        verify_upstream_parity_parse_step.dependOn(&run.step);
    }
}
