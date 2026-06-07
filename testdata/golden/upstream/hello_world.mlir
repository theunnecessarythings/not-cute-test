// upstream parity: hello_world.ipynb
module {
  gpu.module @notcute_upstream {
    gpu.func @hello_world() kernel {
      gpu.return
    }
  }
}
