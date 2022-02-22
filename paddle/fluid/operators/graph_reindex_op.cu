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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/graph_khop_sampler_imp.h"
#include "paddle/fluid/operators/graph_reindex_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/place.h"

constexpr int WARP_SIZE = 32;

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
void FillHashTable(const framework::ExecutionContext& ctx, const T* input,
                   int num_input, int64_t len_hashtable,
                   thrust::device_vector<T>* unique_items, T* keys, int* values,
                   int* key_index) {
#ifdef PADDLE_WITH_HIP
  int block = 256;
#else
  int block = 1024;
#endif
  const auto& dev_ctx = ctx.cuda_device_context();
  int max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int grid_tmp = (num_input + block - 1) / block;
  int grid = grid_tmp < max_grid_dimx ? grid_tmp : max_grid_dimx;
  // TODO(daisiming): Diff 2 - number of grid and block.
  // Insert data into keys and values.
  BuildHashTable<
      T><<<grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                               ctx.device_context())
                               .stream()>>>(input, num_input, len_hashtable,
                                            keys, key_index);

  // TODO(daisiming): Diff 3 - cuda kernel and thrust::for_each.
  // Get item index count.
  thrust::device_vector<int> item_count(num_input + 1, 0);
  GetItemIndexCount<
      T><<<grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                               ctx.device_context())
                               .stream()>>>(
      input, thrust::raw_pointer_cast(item_count.data()), num_input,
      len_hashtable, keys, key_index);

  thrust::exclusive_scan(item_count.begin(), item_count.end(),
                         item_count.begin());
  size_t total_unique_items = item_count[num_input];
  unique_items->resize(total_unique_items);

  // Get unique items
  FillUniqueItems<
      T><<<grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                               ctx.device_context())
                               .stream()>>>(
      input, num_input, len_hashtable,
      thrust::raw_pointer_cast(unique_items->data()),
      thrust::raw_pointer_cast(item_count.data()), keys, values, key_index);
}

template <typename T>
void Reindex(const framework::ExecutionContext& ctx,
             thrust::device_vector<T>* inputs,
             thrust::device_vector<T>* src_outputs,
             thrust::device_vector<T>* out_nodes) {
  out_nodes->resize(inputs->size() + src_outputs->size());
  thrust::copy(inputs->begin(), inputs->end(), out_nodes->begin());
  thrust::copy(src_outputs->begin(), src_outputs->end(),
               out_nodes->begin() + inputs->size());
  thrust::device_vector<T> unique_nodes;
  unique_nodes.clear();

  // Fill hash table.
  int64_t num = out_nodes->size();
  int64_t log_num = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  int64_t table_size = log_num << 1;
  // TODO(daisiming): Diff 1, hash table.
  // 备选：改用  framework::Tensor 形式来申请显存
  T* keys;
  int *values, *key_index;
  cudaMalloc(&keys, table_size * sizeof(T));
  cudaMalloc(&values, table_size * sizeof(int));
  cudaMalloc(&key_index, table_size * sizeof(int));
  cudaMemset(keys, -1, table_size * sizeof(T));
  cudaMemset(values, -1, table_size * sizeof(int));
  cudaMemset(key_index, -1, table_size * sizeof(int));
  FillHashTable<T>(ctx, thrust::raw_pointer_cast(out_nodes->data()),
                   out_nodes->size(), table_size, &unique_nodes, keys, values,
                   key_index);
  out_nodes->resize(unique_nodes.size());
  thrust::copy(unique_nodes.begin(), unique_nodes.end(), out_nodes->begin());

// Fill outputs with reindex result.
#ifdef PADDLE_WITH_HIP
  int block = 256;
#else
  int block = 1024;
#endif
  const auto& dev_ctx = ctx.cuda_device_context();
  int max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int grid_tmp = (src_outputs->size() + block - 1) / block;
  int grid = grid_tmp < max_grid_dimx ? grid_tmp : max_grid_dimx;
  ReindexSrcOutput<
      T><<<grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                               ctx.device_context())
                               .stream()>>>(
      thrust::raw_pointer_cast(src_outputs->data()), src_outputs->size(),
      table_size, keys, values);
  cudaFree(keys);
  cudaFree(values);
  cudaFree(key_index);
}

template <typename T, int BLOCK_WARPS, int TILE_SIZE>
__global__ void GetDstEdgeCUDAKernel(const int64_t num_rows, const T* in_rows,
                                     const T* dst_counts, const T* dst_ptr,
                                     T* dst_outputs) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t dst_sample_size = dst_counts[out_row];
    const int64_t out_row_start = dst_ptr[out_row];
    for (int idx = threadIdx.x; idx < dst_sample_size; idx += WARP_SIZE) {
      dst_outputs[out_row_start + idx] = row;
    }
#ifdef PADDLE_WITH_CUDA
    __syncwarp();
#endif

    out_row += BLOCK_WARPS;
  }
}

template <typename DeviceContext, typename T>
class GraphReindexOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* src = ctx.Input<Tensor>("Src");
    auto* count = ctx.Input<Tensor>("Count");
    auto* vertices = ctx.Input<Tensor>("X");

    const T* src_data = src->data<T>();
    const T* count_data = count->data<T>();
    const T* vertices_data = vertices->data<T>();
    const int bs = vertices->dims()[0];
    const int num_edges = src->dims()[0];

    thrust::device_vector<T> inputs(bs);
    thrust::device_vector<T> src_outputs(num_edges);
    thrust::device_vector<T> dst_outputs(num_edges);
    thrust::device_vector<T> out_nodes;
    thrust::copy(vertices_data, vertices_data + bs, inputs.begin());
    thrust::copy(src_data, src_data + num_edges, src_outputs.begin());
    Reindex<T>(ctx, &inputs, &src_outputs, &out_nodes);

    // Get reindex dst edge.
    int64_t unique_dst_size = inputs.size();
    thrust::device_vector<T> unique_dst_reindex(unique_dst_size);
    thrust::sequence(unique_dst_reindex.begin(), unique_dst_reindex.end());
    thrust::device_vector<T> dst_ptr(unique_dst_size);
    thrust::exclusive_scan(count_data, count_data + bs, dst_ptr.begin());
    constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
    constexpr int TILE_SIZE = BLOCK_WARPS * 16;
    const dim3 block(WARP_SIZE, BLOCK_WARPS);
    const dim3 grid((unique_dst_size + TILE_SIZE - 1) / TILE_SIZE);
    GetDstEdgeCUDAKernel<T, BLOCK_WARPS, TILE_SIZE><<<
        grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                            ctx.device_context())
                            .stream()>>>(
        unique_dst_size, thrust::raw_pointer_cast(unique_dst_reindex.data()),
        count_data, thrust::raw_pointer_cast(dst_ptr.data()),
        thrust::raw_pointer_cast(dst_outputs.data()));

    auto* reindex_src = ctx.Output<Tensor>("Reindex_Src");
    reindex_src->Resize({static_cast<int>(src_outputs.size())});
    auto* reindex_dst = ctx.Output<Tensor>("Reindex_Dst");
    reindex_dst->Resize({static_cast<int>(dst_outputs.size())});
    auto* out_nodes_ = ctx.Output<Tensor>("Out_Nodes");
    out_nodes_->Resize({static_cast<int>(out_nodes.size())});
    T* p_src = reindex_src->mutable_data<T>(ctx.GetPlace());
    T* p_dst = reindex_dst->mutable_data<T>(ctx.GetPlace());
    T* p_nodes = out_nodes_->mutable_data<T>(ctx.GetPlace());
    thrust::copy(src_outputs.begin(), src_outputs.end(), p_src);
    thrust::copy(dst_outputs.begin(), dst_outputs.end(), p_dst);
    thrust::copy(out_nodes.begin(), out_nodes.end(), p_nodes);
  }
};

}  // namespace operators
}  // namespace paddle

using CUDA = paddle::platform::CUDADeviceContext;
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(graph_reindex,
                        ops::GraphReindexOpCUDAKernel<CUDA, int32_t>,
                        ops::GraphReindexOpCUDAKernel<CUDA, int64_t>);
