#!/usr/bin/env python3
"""Optional bridge from the Zig port to an installed CUTLASS DSL MLIR stack.

This script is intentionally small and stdlib-only.  The Zig code remains
zero-dependency by default; this helper is only used when the developer wants to
validate generated textual MLIR with an installed `cutlass` / `nvidia-cutlass-dsl`
package.  Importing `cutlass` triggers `_cutlass_ir.populate(_cutlass_ir)` in the
upstream package, which registers the CUTLASS/CuTe dialects and passes with the
embedded MLIR bindings.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import importlib.metadata
import json
import os
import pathlib
import sys
import traceback
from typing import Any


def _json(status: str, **fields: Any) -> int:
    payload = {"status": status, **fields}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status in {"ok", "passed"} else 1


def import_cutlass(module_name: str):
    return importlib.import_module(module_name)


def distribution_version() -> str:
    candidates = ["nvidia-cutlass-dsl", "cutlass", "cutlass-dsl"]
    for name in candidates:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def discover(module_name: str) -> dict[str, Any]:
    module = import_cutlass(module_name)
    package_file = pathlib.Path(getattr(module, "__file__", "")).resolve()
    package_root = package_file.parent

    mlir_module = importlib.import_module(f"{module_name}._mlir")
    candidates: list[pathlib.Path] = []
    candidates.append(package_root / "_mlir" / "_mlir_libs")
    for path in getattr(mlir_module, "__path__", []):
        candidates.append(pathlib.Path(path) / "_mlir_libs")

    mlir_libs_dir = None
    for candidate in candidates:
        if candidate.exists():
            mlir_libs_dir = candidate.resolve()
            break
    if mlir_libs_dir is None:
        raise RuntimeError("Could not find cutlass._mlir/_mlir_libs")

    matches = sorted(glob.glob(str(mlir_libs_dir / "_cutlass_ir.cpython*.so")))
    if not matches:
        matches = sorted(glob.glob(str(mlir_libs_dir / "_cutlass_ir*.so")))
    if not matches:
        raise RuntimeError(f"Could not find _cutlass_ir shared library under {mlir_libs_dir}")

    cuda_version = None
    try:
        cuda_version = str(getattr(module, "CUDA_VERSION", "")) or None
    except Exception:
        cuda_version = None

    return {
        "python_exe": sys.executable,
        "package_module": module_name,
        "package_version": getattr(module, "__version__", distribution_version()),
        "distribution_version": distribution_version(),
        "package_file": str(package_file),
        "mlir_libs_dir": str(mlir_libs_dir),
        "cutlass_ir_so": str(pathlib.Path(matches[0]).resolve()),
        "cuda_version": cuda_version,
    }


def parse_module(module_name: str, text: str):
    # Importing top-level cutlass triggers _cutlass_ir.populate(_cutlass_ir).
    import_cutlass(module_name)
    ir = importlib.import_module(f"{module_name}._mlir.ir")
    with ir.Context() as ctx, ir.Location.unknown(ctx):
        return ir.Module.parse(text)


def run_pipeline(module_name: str, mlir_text: str, pipeline: str, enable_verifier: bool):
    import_cutlass(module_name)
    ir = importlib.import_module(f"{module_name}._mlir.ir")
    passmanager = importlib.import_module(f"{module_name}._mlir.passmanager")
    with ir.Context() as ctx, ir.Location.unknown(ctx):
        module = ir.Module.parse(mlir_text)
        pm = passmanager.PassManager.parse(pipeline)
        pm.enable_verifier(enable_verifier)
        pm.run(module.operation)
        return str(module)


def parse_only(module_name: str, mlir_text: str) -> str:
    import_cutlass(module_name)
    ir = importlib.import_module(f"{module_name}._mlir.ir")
    with ir.Context() as ctx, ir.Location.unknown(ctx):
        module = ir.Module.parse(mlir_text)
        return str(module)


def dialect_ops(module_name: str, dialect: str) -> list[str]:
    import_cutlass(module_name)
    if dialect == "cute":
        generated = importlib.import_module(f"{module_name}._mlir.dialects._cute_ops_gen")
    elif dialect == "cute_nvgpu":
        generated = importlib.import_module(f"{module_name}._mlir.dialects._cute_nvgpu_ops_gen")
    else:
        raise RuntimeError(f"unsupported dialect: {dialect}")
    names: list[str] = []
    for value in generated.__dict__.values():
        op_name = getattr(value, "OPERATION_NAME", None)
        if isinstance(op_name, str):
            names.append(op_name)
    return sorted(set(names))


def command_discover(args: argparse.Namespace) -> int:
    try:
        info = discover(args.module)
        if args.json:
            return _json("ok", **info)
        for k, v in info.items():
            print(f"{k}: {v}")
        return 0
    except Exception as exc:
        if args.traceback:
            traceback.print_exc()
        return _json("error", error=str(exc))


def command_metadata(args: argparse.Namespace) -> int:
    try:
        info = discover(args.module)
        return _json("ok", bridge="cutlass_mlir_bridge.py", cwd=os.getcwd(), **info)
    except Exception as exc:
        if args.traceback:
            traceback.print_exc()
        return _json("error", error=str(exc))


def command_parse(args: argparse.Namespace) -> int:
    try:
        mlir_text = pathlib.Path(args.input).read_text()
        parsed = parse_only(args.module, mlir_text)
        return _json("passed", input=args.input, output_size=len(parsed))
    except Exception as exc:
        if args.traceback:
            traceback.print_exc()
        return _json("failed", input=args.input, error=str(exc))


def command_ops(args: argparse.Namespace) -> int:
    try:
        ops = dialect_ops(args.module, args.dialect)
        return _json("ok", module=args.module, dialect=args.dialect, count=len(ops), operations=ops)
    except Exception as exc:
        if args.traceback:
            traceback.print_exc()
        return _json("error", dialect=args.dialect, error=str(exc))


def command_verify(args: argparse.Namespace) -> int:
    try:
        mlir_text = pathlib.Path(args.input).read_text()
        lowered = run_pipeline(args.module, mlir_text, args.pipeline, args.enable_verifier)
        return _json("passed", input=args.input, output_size=len(lowered))
    except Exception as exc:
        if args.traceback:
            traceback.print_exc()
        return _json("failed", input=args.input, error=str(exc))


def command_expect_fail(args: argparse.Namespace) -> int:
    try:
        mlir_text = pathlib.Path(args.input).read_text()
        if args.pipeline:
            run_pipeline(args.module, mlir_text, args.pipeline, args.enable_verifier)
        else:
            parse_only(args.module, mlir_text)
        return _json("failed", input=args.input, error="expected failure but command succeeded")
    except Exception as exc:
        message = str(exc)
        if args.expected and args.expected not in message:
            if args.traceback:
                traceback.print_exc()
            return _json("failed", input=args.input, error=message, expected=args.expected)
        return _json("passed", input=args.input, expected=args.expected or None, error=message)


def command_lower(args: argparse.Namespace) -> int:
    try:
        mlir_text = pathlib.Path(args.input).read_text()
        lowered = run_pipeline(args.module, mlir_text, args.pipeline, args.enable_verifier)
        pathlib.Path(args.output).write_text(lowered)
        return _json("passed", input=args.input, output=args.output, output_size=len(lowered))
    except Exception as exc:
        if args.traceback:
            traceback.print_exc()
        return _json("failed", input=args.input, output=args.output, error=str(exc))



def _artifact_record(path: pathlib.Path, kind: str) -> dict[str, Any]:
    exists = path.exists()
    return {
        "kind": kind,
        "path": str(path),
        "exists": exists,
        "size": path.stat().st_size if exists else 0,
    }


def _classify_extensionless_artifact(path: pathlib.Path) -> str | None:
    """Classify CUTLASS pass-manager dump paths that have no extension.

    Upstream CompileOptions pass dump options use a base dump path such as
    ``dump-cubin-path='/tmp/foo/kernel'``.  The pass writes the CUBIN exactly at
    that path, not necessarily at ``kernel.cubin``.  This classifier lets the
    bridge report those real files as CUBIN artifacts rather than missing files.
    """
    if not path.is_file():
        return None
    try:
        prefix = path.read_bytes()[:16]
    except OSError:
        return None
    if prefix.startswith(b"\x7fELF"):
        return "cubin"
    # PTX dumps are textual and typically start with .version/.target.
    try:
        text = prefix.decode("ascii", errors="ignore")
    except Exception:
        text = ""
    if text.startswith(".version") or text.startswith("//"):
        return "ptx"
    return None


def _scan_artifacts(work_dir: pathlib.Path, function: str) -> list[dict[str, Any]]:
    base = work_dir / function
    candidates = [
        (work_dir / f"{function}.lowered.mlir", "lowered_mlir"),
        # Extensionless base path is the canonical CUTLASS pass dump path for
        # cubin-format=bin.  Keep it before extension-based fallbacks so runtime
        # loaders get the file that was actually produced.
        (base, _classify_extensionless_artifact(base) or "cubin"),
        (work_dir / f"{function}.cubin", "cubin"),
        (work_dir / f"{function}.ptx", "ptx"),
        (work_dir / f"{function}.o", "object"),
        (work_dir / f"{function}.json", "json"),
        (work_dir / f"{function}.diag.txt", "diagnostics"),
    ]
    # CUTLASS pass pipelines historically dump several filename variants.  Keep
    # deterministic primary records above, then include discovered extras.
    seen = {str(p) for p, _ in candidates}
    records = [_artifact_record(p, k) for p, k in candidates]
    for pattern, kind in [("*.cubin", "cubin"), ("*.ptx", "ptx"), ("*.o", "object"), ("*.mlir", "lowered_mlir")]:
        for path in sorted(work_dir.glob(pattern)):
            if str(path) in seen:
                continue
            seen.add(str(path))
            records.append(_artifact_record(path, kind))
    # Finally include any other extensionless dump files and classify them by
    # content.  This catches user-provided dump prefixes that do not match the
    # function name exactly.
    for path in sorted(work_dir.iterdir() if work_dir.exists() else []):
        if str(path) in seen or path.suffix:
            continue
        kind = _classify_extensionless_artifact(path)
        if kind is None:
            continue
        seen.add(str(path))
        records.append(_artifact_record(path, kind))
    return records


def command_compile_artifact(args: argparse.Namespace) -> int:
    """Run a CUTLASS MLIR pipeline and report expected/produced artifacts.

    This deliberately works at the Python/PassManager boundary used by CuteDSL.
    The bridge writes a lowered textual MLIR snapshot even when cubin/PTX dumping
    is disabled or unsupported by the selected pipeline.  CUBIN/PTX/object files
    are discovered from the requested work directory after the pass run.
    """
    work_dir = pathlib.Path(args.work_dir)
    diagnostics_path = work_dir / f"{args.function}.diag.txt"
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        mlir_text = pathlib.Path(args.input).read_text()
        if args.pipeline == "parse-only":
            lowered = parse_only(args.module, mlir_text)
        else:
            lowered = run_pipeline(args.module, mlir_text, args.pipeline, args.enable_verifier)
        lowered_path = work_dir / f"{args.function}.lowered.mlir"
        lowered_path.write_text(lowered)
        artifacts = _scan_artifacts(work_dir, args.function)
        missing: list[str] = []
        if args.expect_cubin and not any(r["kind"] == "cubin" and r["exists"] for r in artifacts):
            missing.append("cubin")
        if args.expect_ptx and not any(r["kind"] == "ptx" and r["exists"] for r in artifacts):
            missing.append("ptx")
        if args.expect_object and not any(r["kind"] == "object" and r["exists"] for r in artifacts):
            missing.append("object")
        status = "passed" if not missing else "missing-artifacts"
        payload_status = "passed" if status == "passed" else "failed"
        return _json(
            payload_status,
            phase="compile-artifact",
            status_detail=status,
            input=args.input,
            work_dir=str(work_dir),
            function=args.function,
            pipeline=args.pipeline,
            lowered_mlir=str(lowered_path),
            artifacts=artifacts,
            missing=missing,
        )
    except Exception as exc:
        work_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(str(exc))
        if args.traceback:
            traceback.print_exc()
        return _json(
            "failed",
            phase="compile-artifact",
            input=args.input,
            work_dir=str(work_dir),
            function=args.function,
            pipeline=args.pipeline,
            diagnostics=str(diagnostics_path),
            error=str(exc),
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--module", default="cutlass", help="Python package module to import")
        sp.add_argument("--traceback", action="store_true", help="print Python traceback on failure")

    d = sub.add_parser("discover")
    common(d)
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=command_discover)

    m = sub.add_parser("metadata")
    common(m)
    m.set_defaults(func=command_metadata)

    pr = sub.add_parser("parse")
    common(pr)
    pr.add_argument("--input", required=True)
    pr.set_defaults(func=command_parse)

    o = sub.add_parser("ops")
    common(o)
    o.add_argument("--dialect", choices=["cute", "cute_nvgpu"], default="cute")
    o.set_defaults(func=command_ops)

    ef = sub.add_parser("expect-fail")
    common(ef)
    ef.add_argument("--input", required=True)
    ef.add_argument("--pipeline", default=None)
    ef.add_argument("--enable-verifier", action="store_true")
    ef.add_argument("--expected", default=None, help="substring expected in the diagnostic")
    ef.set_defaults(func=command_expect_fail)

    v = sub.add_parser("verify")
    common(v)
    v.add_argument("--input", required=True)
    v.add_argument("--pipeline", required=True)
    v.add_argument("--enable-verifier", action="store_true")
    v.set_defaults(func=command_verify)

    l = sub.add_parser("lower")
    common(l)
    l.add_argument("--input", required=True)
    l.add_argument("--output", required=True)
    l.add_argument("--pipeline", required=True)
    l.add_argument("--enable-verifier", action="store_true")
    l.set_defaults(func=command_lower)

    ca = sub.add_parser("compile-artifact")
    common(ca)
    ca.add_argument("--input", required=True)
    ca.add_argument("--work-dir", required=True)
    ca.add_argument("--function", required=True)
    ca.add_argument("--pipeline", required=True)
    ca.add_argument("--enable-verifier", action="store_true")
    ca.add_argument("--expect-cubin", action="store_true")
    ca.add_argument("--expect-ptx", action="store_true")
    ca.add_argument("--expect-object", action="store_true")
    ca.set_defaults(func=command_compile_artifact)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
