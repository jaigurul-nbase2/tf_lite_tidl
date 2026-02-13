#!/bin/bash
set -e

aarch64-linux-gnu-g++ infer_model.cpp -o infer_dsp \
  --sysroot=/home/nbase2/Downloads/am62a-rootfs \
  -I/home/nbase2/edgeai/am62a/include/tensorflow \
  -I/home/nbase2/edgeai/am62a/include/tensorflow/tensorflow/lite \
  -I/home/nbase2/edgeai/am62a/flatbuffers-2.0.8/flatbuffers-2.0.8/include \
  -L/home/nbase2/edgeai/am62a/lib \
  -L/home/nbase2/edgeai/am62a/lib/tflite_2.12/ruy-build \
  -L/home/nbase2/edgeai/am62a/lib/tflite_2.12/pthreadpool \
  -L/home/nbase2/edgeai/am62a/lib/tflite_2.12/cpuinfo-build \
  -L/home/nbase2/edgeai/am62a/lib/tflite_2.12/xnnpack-build \
  -L/home/nbase2/edgeai/am62a/lib/tflite_2.12/fft2d-build \
  -L/home/nbase2/edgeai/am62a/lib/tflite_2.12/abseil-cpp-build \
  -L/home/nbase2/edgeai/am62a/lib/tflite_2.12/farmhash-build \
  -L/home/nbase2/fresh_try/compile_model/edgeai-tidl-tools/tools/AM62A/tidl_tools \
  -Wl,--start-group \
    -ltensorflow-lite -lvx_tidl_rt -lXNNPACK -lfft2d_fftsg2d -lfft2d_fftsg \
    -lruy_kernel_arm -lruy_pack_arm -lruy_apply_multiplier \
    -lruy_frontend -lruy_trmul -lruy_ctx -lruy_context \
    -lruy_context_get_ctx -lruy_allocator -lruy_block_map \
    -lruy_blocking_counter -lruy_prepacked_cache \
    -lruy_system_aligned_alloc -lruy_tune -lruy_wait \
    -lruy_thread_pool -lruy_cpuinfo -lpthreadpool -lcpuinfo \
    -labsl_hash -labsl_city -labsl_low_level_hash \
    -lruy_denormal -lruy_prepare_packed_matrices -lfarmhash \
  -Wl,--end-group \
  -Wl,--whole-archive \
  /home/nbase2/edgeai/am62a/lib/libflatbuffers.a \
  -Wl,--no-whole-archive \
  -lpthread -ldl -lm \
  -Xlinker -Map=output_host.map
