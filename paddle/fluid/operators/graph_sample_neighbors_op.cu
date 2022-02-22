/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

This file is inspired by

    https://github.com/quiver-team/torch-quiver/blob/main/srcs/cpp/src/quiver/cuda/quiver_sample.cu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/transform.h>

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>
#else
#include <cuda_runtime.h>
#include <curand_kernel.h>
#endif

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/graph_sample_neighbors_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/place.h"

constexpr int WARP_SIZE = 32;

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
struct MaxFunctor {
  T cap;
  HOSTDEVICE explicit inline MaxFunctor(T cap) { this->cap = cap; }
  HOSTDEVICE inline T operator()(T x) const {
    if (x > cap) {
      return cap;
    }
    return x;
  }
};

template <typename T>
struct DegreeFunctor {
  const T* dst_count;
  HOSTDEVICE explicit inline DegreeFunctor(const T* x) { this->dst_count = x; }
  HOSTDEVICE inline T operator()(T i) const {
    return dst_count[i + 1] - dst_count[i];
  }
};

template <typename T, int BLOCK_WARPS, int TILE_SIZE>
__global__ void SampleKernel(const uint64_t rand_seed, int k,
                             const int64_t num_rows, const T* in_rows,
                             const T* src, const T* dst_count, T* outputs,
                             T* output_ptr, T* output_idxs) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);
#ifdef PADDLE_WITH_HIP
  hiprandState rng;
  hiprand_init(rand_seed * gridDim.x + blockIdx.x,
               threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);
#else
  curandState rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x,
              threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);
#endif

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = dst_count[row];
    const int64_t deg = dst_count[row + 1] - in_row_start;
    const int64_t out_row_start = output_ptr[out_row];

    if (deg <= k) {
      for (int idx = threadIdx.x; idx < deg; idx += WARP_SIZE) {
        const T in_idx = in_row_start + idx;
        outputs[out_row_start + idx] = src[in_idx];
      }
    } else {
      for (int idx = threadIdx.x; idx < k; idx += WARP_SIZE) {
        output_idxs[out_row_start + idx] = idx;
      }
#ifdef PADDLE_WITH_CUDA
      __syncwarp();
#endif

      for (int idx = k + threadIdx.x; idx < deg; idx += WARP_SIZE) {
#ifdef PADDLE_WITH_HIP
        const int num = hiprand(&rng) % (idx + 1);
#else
        const int num = curand(&rng) % (idx + 1);
#endif
        if (num < k) {
          atomicMax(reinterpret_cast<unsigned long long int*>(  // NOLINT
                        output_idxs + out_row_start + num),
                    static_cast<unsigned long long int>(idx));  // NOLINT
        }
      }
#ifdef PADDLE_WITH_CUDA
      __syncwarp();
#endif

      for (int idx = threadIdx.x; idx < k; idx += WARP_SIZE) {
        const T perm_idx = output_idxs[out_row_start + idx] + in_row_start;
        outputs[out_row_start + idx] = src[perm_idx];
      }
    }

    out_row += BLOCK_WARPS;
  }
}

template <typename T>
void SampleNeighbors(const framework::ExecutionContext& ctx, const T* src,
                     const T* dst_count, thrust::device_vector<T>* inputs,
                     thrust::device_vector<T>* outputs,
                     thrust::device_vector<T>* output_counts, int k) {
  const size_t bs = inputs->size();
  output_counts->resize(bs);

  thrust::transform(inputs->begin(), inputs->end(), output_counts->begin(),
                    DegreeFunctor<T>(dst_count));

  if (k >= 0) {
    thrust::transform(output_counts->begin(), output_counts->end(),
                      output_counts->begin(), MaxFunctor<T>(k));
  }

  T total_sample_num =
      thrust::reduce(output_counts->begin(), output_counts->end());

  outputs->resize(total_sample_num);

  thrust::device_vector<T> output_ptr;
  thrust::device_vector<T> output_idxs;
  output_ptr.resize(bs);
  output_idxs.resize(total_sample_num);
  thrust::exclusive_scan(output_counts->begin(), output_counts->end(),
                         output_ptr.begin(), 0);

  constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
  constexpr int TILE_SIZE = BLOCK_WARPS * 16;
  const dim3 block(WARP_SIZE, BLOCK_WARPS);
  const dim3 grid((bs + TILE_SIZE - 1) / TILE_SIZE);
  SampleKernel<T, BLOCK_WARPS, TILE_SIZE><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx.device_context())
          .stream()>>>(0, k, bs, thrust::raw_pointer_cast(inputs->data()), src,
                       dst_count, thrust::raw_pointer_cast(outputs->data()),
                       thrust::raw_pointer_cast(output_ptr.data()),
                       thrust::raw_pointer_cast(output_idxs.data()));
}

template <typename DeviceContext, typename T>
class GraphSamplerNeighborsOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* src = ctx.Input<Tensor>("Row");
    auto* dst_count = ctx.Input<Tensor>("Col_Ptr");
    auto* vertices = ctx.Input<Tensor>("X");
    int sample_size = ctx.Attr<int>("sample_size");

    const T* src_data = src->data<T>();
    const T* dst_count_data = dst_count->data<T>();
    const T* p_vertices = vertices->data<T>();
    const int bs = vertices->dims()[0];

    thrust::device_vector<T> inputs(bs);
    thrust::copy(p_vertices, p_vertices + bs, inputs.begin());
    thrust::device_vector<T> outputs;
    thrust::device_vector<T> output_counts;

    SampleNeighbors<T>(ctx, src_data, dst_count_data, &inputs, &outputs,
                       &output_counts, sample_size);

    auto* out = ctx.Output<Tensor>("Out");
    out->Resize({static_cast<int>(outputs.size())});
    T* p_out = out->mutable_data<T>(ctx.GetPlace());
    thrust::copy(outputs.begin(), outputs.end(), p_out);
    auto* out_count = ctx.Output<Tensor>("Out_Count");
    out_count->Resize({static_cast<int>(output_counts.size())});
    T* p_out_count = out_count->mutable_data<T>(ctx.GetPlace());
    thrust::copy(output_counts.begin(), output_counts.end(), p_out_count);
  }
};

}  // namespace operators
}  // namespace paddle

using CUDA = paddle::platform::CUDADeviceContext;
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(graph_sample_neighbors,
                        ops::GraphSamplerNeighborsOpCUDAKernel<CUDA, int32_t>,
                        ops::GraphSamplerNeighborsOpCUDAKernel<CUDA, int64_t>);
