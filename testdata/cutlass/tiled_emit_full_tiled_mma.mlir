!mma_f32_1x1x1 = !cute.tiled_mma<!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, atom_layout_MNK = <"(1,1,1):(1,1,1)">>
!memref_generic_f32_1x1 = !cute.memref<f32, generic, "(1,1):(1,1)">
!memref_generic_f32_frag = !cute.memref<f32, generic, "(1,1,1):(0,0,0)">
module {
  func.func @tiled_emit_full_tiled_mma(%arg0: !memref_generic_f32_1x1, %arg1: !memref_generic_f32_1x1, %arg2: !memref_generic_f32_1x1, %arg3: !memref_generic_f32_1x1, %arg4: !cute.coord<"0">) {
    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >
    %tiled = cute.make_tiled_mma(%atom) : !mma_f32_1x1x1
    %a = cute.tiled.mma.partition A(%tiled, %arg0, %arg4) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    %b = cute.tiled.mma.partition B(%tiled, %arg1, %arg4) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    %c = cute.tiled.mma.partition C(%tiled, %arg2, %arg4) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    %d = cute.tiled.mma.partition C(%tiled, %arg3, %arg4) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    cute.gemm(%tiled, %d, %a, %b, %c) : (!mma_f32_1x1x1, !memref_generic_f32_frag, !memref_generic_f32_frag, !memref_generic_f32_frag, !memref_generic_f32_frag)
    return
  }
}

