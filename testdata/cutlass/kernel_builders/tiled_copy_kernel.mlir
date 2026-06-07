// not-cute kernel builder: tiled_copy arch=generic
!coord_zero = !cute.coord<"0">
!memref_scalar = !cute.memref<f32, gmem, align<16>, "(1):(1)">
!memref_vec = !cute.memref<f32, gmem, align<16>, "(4):(1)">
!memref_tile = !cute.memref<f32, gmem, "(1,1):(1,1)">
!copy_simt = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 32 b>, layout_copy_tv = <"(1,1):(1,1)">, tiler_mn = <"[1:0;1:0]">>
!memref_partition = !cute.memref<f32, gmem, "((1,1),1,1):((0,0),0,0)">
module {
  gpu.module @notcute_kernels {
    gpu.func @tiled_copy_kernel(%src: !memref_tile, %dst: !memref_tile, %coord: !coord_zero) kernel {
      %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32, 32 b>
      %tiled = cute.make_tiled_copy(%atom) : !copy_simt
      %s = cute.tiled.copy.partition_S(%tiled, %src, %coord) : (!copy_simt, !memref_tile, !coord_zero) -> !memref_partition
      %d = cute.tiled.copy.partition_D(%tiled, %dst, %coord) : (!copy_simt, !memref_tile, !coord_zero) -> !memref_partition
      cute.copy(%tiled, %s, %d) : (!copy_simt, !memref_partition, !memref_partition)
      gpu.return
    }
  }
}
