!copy_simt = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 32 b>, layout_copy_tv = <"(1,1):(1,1)">, tiler_mn = <"[1:0;1:0]">>
!memref_gmem_f32_1x1 = !cute.memref<f32, gmem, "(1,1):(1,1)">
!memref_gmem_f32_partition = !cute.memref<f32, gmem, "((1,1),1,1):((0,0),0,0)">
module {
  func.func @tiled_emit_full_tiled_copy(%arg0: !memref_gmem_f32_1x1, %arg1: !memref_gmem_f32_1x1, %arg2: !cute.coord<"0">) {
    %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>
    %tiled = cute.make_tiled_copy(%atom) : !copy_simt
    %src_partitioned = cute.tiled.copy.partition_S(%tiled, %arg0, %arg2) : (!copy_simt, !memref_gmem_f32_1x1, !cute.coord<"0">) -> !memref_gmem_f32_partition
    %dst_partitioned = cute.tiled.copy.partition_D(%tiled, %arg1, %arg2) : (!copy_simt, !memref_gmem_f32_1x1, !cute.coord<"0">) -> !memref_gmem_f32_partition
    cute.copy(%tiled, %src_partitioned, %dst_partitioned) : (!copy_simt, !memref_gmem_f32_partition, !memref_gmem_f32_partition)
    return
  }
}

