// not-cute kernel builder: mma_microkernel arch=generic
!coord_zero = !cute.coord<"0">
!memref_scalar = !cute.memref<f32, gmem, align<16>, "(1):(1)">
!memref_vec = !cute.memref<f32, gmem, align<16>, "(4):(1)">
!memref_tile = !cute.memref<f32, gmem, "(1,1):(1,1)">
// MMA microkernel
// flavor: universal_fma
!mma_f32 = !cute.tiled_mma<!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, atom_layout_MNK = <"(1,1,1):(1,1,1)">>
!frag = !cute.memref<f32, gmem, "(1,1,1):(0,0,0)">
module {
  gpu.module @notcute_kernels {
    gpu.func @mma_microkernel(%a_in: !memref_tile, %b_in: !memref_tile, %c_in: !memref_tile, %d_out: !memref_tile, %coord: !coord_zero) kernel {
      %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >
      %tiled = cute.make_tiled_mma(%atom) : !mma_f32
      %a = cute.tiled.mma.partition A(%tiled, %a_in, %coord) : (!mma_f32, !memref_tile, !coord_zero) -> !frag
      %b = cute.tiled.mma.partition B(%tiled, %b_in, %coord) : (!mma_f32, !memref_tile, !coord_zero) -> !frag
      %c = cute.tiled.mma.partition C(%tiled, %c_in, %coord) : (!mma_f32, !memref_tile, !coord_zero) -> !frag
      %d = cute.tiled.mma.partition C(%tiled, %d_out, %coord) : (!mma_f32, !memref_tile, !coord_zero) -> !frag
      cute.gemm(%tiled, %d, %a, %b, %c) : (!mma_f32, !frag, !frag, !frag, !frag)
      gpu.return
    }
  }
}
