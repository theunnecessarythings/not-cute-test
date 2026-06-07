!copy_simt = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 32 b>, layout_copy_tv = <"(1,1):(1,1)">, tiler_mn = <"[1:0;1:0]">>
!mma_f32_1x1x1 = !cute.tiled_mma<!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, atom_layout_MNK = <"(1,1,1):(1,1,1)">>
!memref_gmem_f32_1x1 = !cute.memref<f32, gmem, "(1,1):(1,1)">
!memref_gmem_f32_partition = !cute.memref<f32, gmem, "((1,1),1,1):((0,0),0,0)">
!memref_generic_f32_1x1 = !cute.memref<f32, generic, "(1,1):(1,1)">
!memref_generic_f32_frag = !cute.memref<f32, generic, "(1,1,1):(0,0,0)">
module {
  func.func @gemm_skeleton(%arg0: !memref_gmem_f32_1x1, %arg1: !memref_gmem_f32_1x1, %arg2: !memref_generic_f32_1x1, %arg3: !memref_generic_f32_1x1, %arg4: !memref_generic_f32_1x1, %arg5: !memref_generic_f32_1x1, %arg6: !cute.coord<"0">) {
    %copy_atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>
    %copy_tiled = cute.make_tiled_copy(%copy_atom) : !copy_simt
    %src_partitioned = cute.tiled.copy.partition_S(%copy_tiled, %arg0, %arg6) : (!copy_simt, !memref_gmem_f32_1x1, !cute.coord<"0">) -> !memref_gmem_f32_partition
    %dst_partitioned = cute.tiled.copy.partition_D(%copy_tiled, %arg1, %arg6) : (!copy_simt, !memref_gmem_f32_1x1, !cute.coord<"0">) -> !memref_gmem_f32_partition
    cute.copy(%copy_tiled, %src_partitioned, %dst_partitioned) : (!copy_simt, !memref_gmem_f32_partition, !memref_gmem_f32_partition)
    %mma_atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >
    %mma_tiled = cute.make_tiled_mma(%mma_atom) : !mma_f32_1x1x1
    %a = cute.tiled.mma.partition A(%mma_tiled, %arg2, %arg6) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    %b = cute.tiled.mma.partition B(%mma_tiled, %arg3, %arg6) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    %c = cute.tiled.mma.partition C(%mma_tiled, %arg4, %arg6) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    %d = cute.tiled.mma.partition C(%mma_tiled, %arg5, %arg6) : (!mma_f32_1x1x1, !memref_generic_f32_1x1, !cute.coord<"0">) -> !memref_generic_f32_frag
    cute.gemm(%mma_tiled, %d, %a, %b, %c) : (!mma_f32_1x1x1, !memref_generic_f32_frag, !memref_generic_f32_frag, !memref_generic_f32_frag, !memref_generic_f32_frag)
    return
  }
}
