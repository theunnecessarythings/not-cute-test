from __future__ import annotations

import subprocess
import os
from pathlib import Path

import modal

APP_NAME = "not-cute-verify"

# Local paths
LOCAL_DIR = Path(__file__).parent.parent
LOCAL_LAUNCHER_BIN = LOCAL_DIR / "zig-out" / "bin" / "not-cute-launch"
LOCAL_BRIDGE_SCRIPT = LOCAL_DIR / "tools" / "cutlass_mlir_bridge.py"
LOCAL_MLIR_FIXTURE = LOCAL_DIR / "testdata" / "cutlass" / "kernel_tiled_copy.mlir"

# Remote paths
REMOTE_LAUNCHER_BIN = "/workspace/not-cute-launch"
REMOTE_BRIDGE_SCRIPT = "/workspace/cutlass_mlir_bridge.py"
REMOTE_MLIR_FIXTURE = "/workspace/kernel.mlir"
REMOTE_WORK_DIR = "/workspace/work"
CUDA_DRIVER_CANDIDATES = (
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/local/nvidia/lib64/libcuda.so.1",
    "/usr/local/nvidia/lib/libcuda.so.1",
)

# Match the Ubuntu 22.04 runtime instead of linking against the host glibc.
if modal.is_local():
    subprocess.run(
        [
            "zig",
            "build",
            "launch",
            "-Dtarget=x86_64-linux-gnu.2.35",
        ],
        cwd=LOCAL_DIR,
        check=True,
    )

# Ensure the image has CUDA, Python, and the required CUTLASS DSL package
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("nvidia-cutlass-dsl")
)

# Only add files if they exist locally
image = image.add_local_file(LOCAL_LAUNCHER_BIN, remote_path=REMOTE_LAUNCHER_BIN)
if LOCAL_BRIDGE_SCRIPT.exists():
    image = image.add_local_file(LOCAL_BRIDGE_SCRIPT, remote_path=REMOTE_BRIDGE_SCRIPT)
if LOCAL_MLIR_FIXTURE.exists():
    image = image.add_local_file(LOCAL_MLIR_FIXTURE, remote_path=REMOTE_MLIR_FIXTURE)

app = modal.App(APP_NAME, image=image)

@app.function(gpu="T4", timeout=60)
def run_verify(kernel_name: str) -> None:
    print("--- Modal GPU Verification ---")
    
    # Debug: find libcuda and check dependencies
    print("Searching for libcuda.so.1...")
    find_res = subprocess.run(["find", "/usr", "-name", "libcuda.so.1"], capture_output=True, text=True)
    print(find_res.stdout)
    
    print("Launcher dependencies (ldd):")
    subprocess.run(["ldd", REMOTE_LAUNCHER_BIN], check=False)
    
    subprocess.run(["ldconfig"], check=False)
    
    cuda_driver_path = next(
        (path for path in CUDA_DRIVER_CANDIDATES if os.path.exists(path)),
        None,
    )
    if cuda_driver_path is None:
        raise RuntimeError("GPU-mounted libcuda.so.1 was not found")
    print("CUDA driver library:", cuda_driver_path)
    
    # 1. Check dependencies
    if not os.path.exists(REMOTE_LAUNCHER_BIN):
        raise RuntimeError(f"Launcher binary not found on remote. Build locally first with 'zig build launch'")
        
    subprocess.run(["chmod", "+x", REMOTE_LAUNCHER_BIN], check=True)

    # 2. Compile MLIR to CUBIN using the Python bridge
    os.makedirs(REMOTE_WORK_DIR, exist_ok=True)
    
    compile_cmd = [
        "python", REMOTE_BRIDGE_SCRIPT, "compile-artifact",
        "--input", REMOTE_MLIR_FIXTURE,
        "--work-dir", REMOTE_WORK_DIR,
        "--function", kernel_name,
        "--pipeline", f"builtin.module(cute-to-nvvm{{cubin-format=bin cubin-chip='sm_75' dump-cubin-path='{REMOTE_WORK_DIR}/{kernel_name}' preserve-line-info=true}})",
        "--enable-verifier",
        "--expect-cubin"
    ]
    
    print(f"Compiling: {' '.join(compile_cmd)}")
    result = subprocess.run(compile_cmd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, compile_cmd)

    # 3. Launch the generated CUBIN via the Zig execution CLI
    cubin_path = os.path.join(REMOTE_WORK_DIR, kernel_name)
    launch_cmd = [
        REMOTE_LAUNCHER_BIN,
        cubin_path,
        kernel_name,
        "1", "1", "1",
        "128", "1", "1",
        cuda_driver_path,
    ]
    
    print(f"\nLaunching: {' '.join(launch_cmd)}")
    try:
        result = subprocess.run(launch_cmd, text=True, timeout=45)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("CUDA launcher did not finish within 45 seconds") from exc
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, launch_cmd)
        
    print("\nVerification successful!")


@app.local_entrypoint()
def main(kernel: str = "tiled_copy_kernel") -> None:
    """
    Verify the CUTLASS MLIR bridge and Zig CUDA executor on a remote GPU.
    """
    run_verify.remote(kernel)
