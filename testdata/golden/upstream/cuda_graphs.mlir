// upstream parity: cuda_graphs.ipynb dry-run launch body
module {
  gpu.module @notcute_upstream {
    gpu.func @cuda_graphs() kernel {
      gpu.return
    }
  }
}
