!copy_simt = !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32, 32 b>, layout_copy_tv = <"(1,1):(1,1)">, tiler_mn = <"[1:0;1:0]">>
module {
  func.func @tiled_copy_case(%arg0: !copy_simt, %arg1: !cute.memref<f32, gmem, align<16>, "(1,1):(1,1)">, %arg2: !cute.coord<"0">) {
    %0 = cute.tiled.copy.partition_S(%arg0, %arg1, %arg2) : (!copy_simt, !cute.memref<f32, gmem, align<16>, "(1,1):(1,1)">, !cute.coord<"0">) -> !cute.memref<f32, gmem, align<16>, "((1,1),1,1):((0,0),0,0)">
    return
  }
}
