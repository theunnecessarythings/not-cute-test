// not-cute kernel builder: epilogue arch=generic
!coord_zero = !cute.coord<"0">
!memref_scalar = !cute.memref<f32, gmem, align<16>, "(1):(1)">
!memref_vec = !cute.memref<f32, gmem, align<16>, "(4):(1)">
!memref_tile = !cute.memref<f32, gmem, "(1,1):(1,1)">
module {
  gpu.module @notcute_kernels {
    gpu.func @epilogue_kernel(%acc: !memref_vec, %bias: !memref_vec, %dst: !memref_vec) kernel {
      %a = cute.memref.load_vec(%acc) : (!memref_vec) -> vector<4xf32>
      %b = cute.memref.load_vec(%bias) : (!memref_vec) -> vector<4xf32>
      %r = arith.addf %a, %b : vector<4xf32>
      cute.memref.store_vec(%r, %dst) : (vector<4xf32>, !memref_vec) -> ()
      gpu.return
    }
  }
}
