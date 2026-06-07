// not-cute kernel builder: copy arch=generic
!coord_zero = !cute.coord<"0">
!memref_scalar = !cute.memref<f32, gmem, align<16>, "(1):(1)">
!memref_vec = !cute.memref<f32, gmem, align<16>, "(4):(1)">
!memref_tile = !cute.memref<f32, gmem, "(1,1):(1,1)">
module {
  gpu.module @notcute_kernels {
    gpu.func @copy_kernel(%src: !memref_scalar, %dst: !memref_scalar, %coord: !coord_zero) kernel {
      %v = cute.memref.load(%src, %coord) : (!memref_scalar, !coord_zero) -> f32
      cute.memref.store(%dst, %coord, %v) : (!memref_scalar, !coord_zero, f32) -> ()
      gpu.return
    }
  }
}
