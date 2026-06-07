module {
  func.func @mma_atom_case(%arg0: !cute.memref<f32, rmem, "(1):(1)">, %arg1: !cute.memref<f32, rmem, "(1):(1)">, %arg2: !cute.memref<f32, rmem, "(1):(1)">, %arg3: !cute.memref<f32, rmem, "(1):(1)">) {
    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >
    cute.mma_atom_call(%atom, %arg0, %arg1, %arg2, %arg3) : (!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, !cute.memref<f32, rmem, "(1):(1)">, !cute.memref<f32, rmem, "(1):(1)">, !cute.memref<f32, rmem, "(1):(1)">, !cute.memref<f32, rmem, "(1):(1)">) -> ()
    return
  }
}

