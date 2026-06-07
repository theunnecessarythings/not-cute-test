// not-cute kernel builder: vector_copy arch=generic
!coord_zero = !cute.coord<"0">
!memref_scalar = !cute.memref<f32, gmem, align<16>, "(1):(1)">
!memref_vec = !cute.memref<f32, gmem, align<16>, "(4):(1)">
!memref_tile = !cute.memref<f32, gmem, "(1,1):(1,1)">
module {
  gpu.module @notcute_kernels {
    gpu.func @vector_copy_kernel(%src: !memref_vec, %dst: !memref_vec) kernel {
      %v = cute.memref.load_vec(%src) : (!memref_vec) -> vector<4xf32>
      cute.memref.store_vec(%v, %dst) : (vector<4xf32>, !memref_vec) -> ()
      gpu.return
    }
  }
}
