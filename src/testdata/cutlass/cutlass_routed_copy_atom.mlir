module {
  func.func @routed_copy_atom(%arg0: !cute.memref<f32, gmem, "(1):(1)">, %arg1: !cute.memref<f32, gmem, "(1):(1)">) {
    %0 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>
    cute.copy_atom_call(%0, %arg0, %arg1) : (!cute_nvgpu.atom.universal_copy<f32, 32 b>, !cute.memref<f32, gmem, "(1):(1)">, !cute.memref<f32, gmem, "(1):(1)">) -> ()
    return
  }
}
