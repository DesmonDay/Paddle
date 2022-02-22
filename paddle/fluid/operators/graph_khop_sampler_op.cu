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
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <ostream>

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>
#else
#include <cuda_runtime.h>
#include <curand_kernel.h>
#endif

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/graph_khop_sampler_imp.h"
#include "paddle/fluid/operators/graph_khop_sampler_op.h"
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
__global__ void GraphSampleNeighborsCUDAKernel(
    const uint64_t rand_seed, int k, const int64_t num_rows, const T* in_rows,
    const T* src, const T* dst_count, const T* src_eids, T* outputs,
    T* outputs_eids, T* output_ptr, T* output_idxs, bool return_eids) {
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
        if (return_eids) {
          outputs_eids[out_row_start + idx] = src_eids[in_idx];
        }
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
          paddle::platform::CudaAtomicMax(output_idxs + out_row_start + num,
                                          idx);
        }
      }
#ifdef PADDLE_WITH_CUDA
      __syncwarp();
#endif

      for (int idx = threadIdx.x; idx < k; idx += WARP_SIZE) {
        const T perm_idx = output_idxs[out_row_start + idx] + in_row_start;
        outputs[out_row_start + idx] = src[perm_idx];
        if (return_eids) {
          outputs_eids[out_row_start + idx] = src_eids[perm_idx];
        }
      }
    }

    out_row += BLOCK_WARPS;
  }
}

template <typename T, int BLOCK_WARPS, int TILE_SIZE>
__global__ void GetDstEdgeCUDAKernel(const int64_t num_rows, const T* in_rows,
                                     const T* dst_sample_counts,
                                     const T* dst_ptr, T* outputs) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t dst_sample_size = dst_sample_counts[out_row];
    const int64_t out_row_start = dst_ptr[out_row];
    for (int idx = threadIdx.x; idx < dst_sample_size; idx += WARP_SIZE) {
      outputs[out_row_start + idx] = row;
    }
#ifdef PADDLE_WITH_CUDA
    __syncwarp();
#endif

    out_row += BLOCK_WARPS;
  }
}

template <typename T>
bool SampleNeighbors(const framework::ExecutionContext& ctx, const T* src,
                     const T* dst_count, const T* src_eids,
                     thrust::device_vector<T>* inputs,
                     thrust::device_vector<T>* outputs,
                     thrust::device_vector<T>* output_counts,
                     thrust::device_vector<T>* outputs_eids, int k,
                     bool return_eids) {
  const size_t bs = inputs->size();
  output_counts->resize(bs);

  // 1. Get input nodes' degree.
  thrust::transform(inputs->begin(), inputs->end(), output_counts->begin(),
                    DegreeFunctor<T>(dst_count));

  // 2. Apply sample size k to get final sample size.
  // If k == -1, the degree value will not change.
  if (k >= 0) {
    thrust::transform(output_counts->begin(), output_counts->end(),
                      output_counts->begin(), MaxFunctor<T>(k));
  }

  // 3. Get the number of total sample neighbors and some necessary datas.
  T total_sample_num =
      thrust::reduce(output_counts->begin(), output_counts->end());

  outputs->resize(total_sample_num);
  if (return_eids) {
    outputs_eids->resize(total_sample_num);
  }
  if (total_sample_num == 0) {
    return false;
  }

  thrust::device_vector<T> output_ptr;
  thrust::device_vector<T> output_idxs;
  output_ptr.resize(bs);
  output_idxs.resize(total_sample_num);
  thrust::exclusive_scan(output_counts->begin(), output_counts->end(),
                         output_ptr.begin(), 0);

  // 4. Run graph sample kernel.
  constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
  constexpr int TILE_SIZE = BLOCK_WARPS * 16;
  const dim3 block(WARP_SIZE, BLOCK_WARPS);
  const dim3 grid((bs + TILE_SIZE - 1) / TILE_SIZE);
  // TODO(daisiming): 是否将return_eids 相关的新增一个Kernel，减少 if 判断
  GraphSampleNeighborsCUDAKernel<T, BLOCK_WARPS, TILE_SIZE><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx.device_context())
          .stream()>>>(
      0, k, bs, thrust::raw_pointer_cast(inputs->data()), src, dst_count,
      src_eids, thrust::raw_pointer_cast(outputs->data()),
      thrust::raw_pointer_cast(outputs_eids->data()),
      thrust::raw_pointer_cast(output_ptr.data()),
      thrust::raw_pointer_cast(output_idxs.data()), return_eids);

  return true;
}

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
  // 1. Insert data into keys and values.
  BuildHashTable<
      T><<<grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                               ctx.device_context())
                               .stream()>>>(input, num_input, len_hashtable,
                                            keys, key_index);

  // 2. Get item index count.
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

  // 3. Get unique items.
  FillUniqueItems<
      T><<<grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                               ctx.device_context())
                               .stream()>>>(
      input, num_input, len_hashtable,
      thrust::raw_pointer_cast(unique_items->data()),
      thrust::raw_pointer_cast(item_count.data()), keys, values, key_index);
}

template <typename T>
void ReindexFunc(const framework::ExecutionContext& ctx,
                 thrust::device_vector<T>* inputs,
                 thrust::device_vector<T>* outputs,
                 thrust::device_vector<T>* subset,
                 thrust::device_vector<T>* orig_nodes,
                 thrust::device_vector<T>* reindex_nodes, int bs) {
  subset->resize(inputs->size() + outputs->size());
  thrust::copy(inputs->begin(), inputs->end(), subset->begin());
  thrust::copy(outputs->begin(), outputs->end(),
               subset->begin() + inputs->size());
  thrust::device_vector<T> unique_items;
  unique_items.clear();

  // Fill hash table.
  int64_t num = subset->size();
  int64_t log_num = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  int64_t size = log_num << 1;
  // cudaMalloc?
  // thrust::device_vector<T> keys(size, -1);
  // thrust::device_vector<int> values(size, -1);
  // thrust::device_vector<int> key_index(size, -1);
  T* keys;
  int *values, *key_index;
  cudaMalloc(&keys, size * sizeof(T));
  cudaMalloc(&values, size * sizeof(int));
  cudaMalloc(&key_index, size * sizeof(int));
  cudaMemset(keys, -1, size * sizeof(T));
  cudaMemset(values, -1, size * sizeof(int));
  cudaMemset(key_index, -1, size * sizeof(int));
  FillHashTable<T>(ctx, thrust::raw_pointer_cast(subset->data()),
                   subset->size(), size, &unique_items, keys, values,
                   key_index);

  subset->resize(unique_items.size());
  thrust::copy(unique_items.begin(), unique_items.end(), subset->begin());

// Fill outputs with reindex result.
#ifdef PADDLE_WITH_HIP
  int block = 256;
#else
  int block = 1024;
#endif
  const auto& dev_ctx = ctx.cuda_device_context();
  int max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int grid_tmp = (outputs->size() + block - 1) / block;
  int grid = grid_tmp < max_grid_dimx ? grid_tmp : max_grid_dimx;
  ReindexSrcOutput<
      T><<<grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                               ctx.device_context())
                               .stream()>>>(
      thrust::raw_pointer_cast(outputs->data()), outputs->size(), size, keys,
      values);

  int grid_ = (bs + block - 1) / block;
  ReindexInputNodes<T><<<grid_, block, 0,
                         reinterpret_cast<const platform::CUDADeviceContext&>(
                             ctx.device_context())
                             .stream()>>>(
      thrust::raw_pointer_cast(orig_nodes->data()), bs,
      thrust::raw_pointer_cast(reindex_nodes->data()), size, keys, values);
  cudaFree(keys);
  cudaFree(values);
  cudaFree(key_index);
}

template <typename DeviceContext, typename T>
class GraphKhopSamplerOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // 1. Get sample neighbors operators' inputs.
    auto* src = ctx.Input<Tensor>("Row");
    auto* dst_count = ctx.Input<Tensor>("Col_Ptr");
    auto* vertices = ctx.Input<Tensor>("X");
    int sample_size = ctx.Attr<int>("sample_size");
    bool return_eids = ctx.Attr<bool>("return_eids");
    bool set_reindex = ctx.Attr<bool>("set_reindex");
    bool set_unique = ctx.Attr<bool>("set_unique");

    const T* src_data = src->data<T>();
    const T* dst_count_data = dst_count->data<T>();
    const T* p_vertices = vertices->data<T>();
    const int bs = vertices->dims()[0];

    // 2. Get unique input nodes(X).
    thrust::device_vector<T> inputs(bs);
    thrust::copy(p_vertices, p_vertices + bs, inputs.begin());
    if (set_unique) {
      // TODO(daisiming): thrust::unique：只能去掉连续重复的数字，是否要支持
      // set_unique?
      // thrust::sort(inputs.begin(), inputs.end());
      auto unique_inputs_end = thrust::unique(inputs.begin(), inputs.end());
      inputs.resize(thrust::distance(inputs.begin(), unique_inputs_end));
    }

    // 3. Sample neighbors. We should distinguish w/o "Src_Eids".
    // Update: We only have one sample_size value.
    thrust::device_vector<T> outputs;
    thrust::device_vector<T> output_counts;
    thrust::device_vector<T> outputs_eids;
    bool have_neighbors = false;
    if (return_eids) {
      auto* src_eids = ctx.Input<Tensor>("Eids");
      const T* src_eids_data = src_eids->data<T>();
      have_neighbors = SampleNeighbors<T>(
          ctx, src_data, dst_count_data, src_eids_data, &inputs, &outputs,
          &output_counts, &outputs_eids, sample_size, return_eids);
      auto* out_eids = ctx.Output<Tensor>("Out_Eids");
      out_eids->Resize({static_cast<int>(outputs_eids.size())});
      T* p_out_eids = out_eids->mutable_data<T>(ctx.GetPlace());
      thrust::copy(outputs_eids.begin(), outputs_eids.end(), p_out_eids);
    } else {
      have_neighbors = SampleNeighbors<T>(
          ctx, src_data, dst_count_data, nullptr, &inputs, &outputs,
          &output_counts, &outputs_eids, sample_size, return_eids);
    }

    int64_t num_sample_edges =
        thrust::reduce(output_counts.begin(), output_counts.end());

    PADDLE_ENFORCE_EQ(
        outputs.size(), num_sample_edges,
        platform::errors::PreconditionNotMet(
            "Number of sample edges dismatch, the sample kernel has error."));

    if (!have_neighbors) {
      auto* out_src = ctx.Output<Tensor>("Out_Src");
      auto* out_dst = ctx.Output<Tensor>("Out_Dst");
      auto* sample_count = ctx.Output<Tensor>("Sample_Count");
      auto* sample_index = ctx.Output<Tensor>("Sample_Index");
      out_src->Resize({static_cast<int>(0)});
      out_dst->Resize({static_cast<int>(0)});
      T* p_out_src = out_src->mutable_data<T>(ctx.GetPlace());
      T* p_out_dst = out_dst->mutable_data<T>(ctx.GetPlace());
      thrust::copy(outputs.begin(), outputs.end(), p_out_src);
      thrust::copy(outputs.begin(), outputs.end(), p_out_dst);

      sample_count->Resize({static_cast<int>(output_counts.size())});
      T* p_sample_count = sample_count->mutable_data<T>(ctx.GetPlace());
      thrust::copy(output_counts.begin(), output_counts.end(), p_sample_count);

      if (set_unique) {
        auto new_unique_input_end =
            thrust::unique(inputs.begin(), inputs.end());
        sample_index->Resize({static_cast<int>(
            thrust::distance(inputs.begin(), new_unique_input_end))});
        T* p_sample_index = sample_index->mutable_data<T>(ctx.GetPlace());
        thrust::copy(inputs.begin(), new_unique_input_end, p_sample_index);
      } else {
        sample_index->Resize({static_cast<int>(inputs.size())});
        T* p_sample_index = sample_index->mutable_data<T>(ctx.GetPlace());
        thrust::copy(inputs.begin(), inputs.end(), p_sample_index);
      }

      if (set_reindex) {
        // Get Reindex_X.
        auto* reindex_x = ctx.Output<Tensor>("Reindex_X");
        reindex_x->Resize({static_cast<int>(inputs.size())});
        T* p_reindex_x = reindex_x->mutable_data<T>(ctx.GetPlace());
        thrust::device_vector<T> unique_dst_reindex(inputs.size());
        thrust::sequence(unique_dst_reindex.begin(), unique_dst_reindex.end());
        thrust::copy(unique_dst_reindex.begin(), unique_dst_reindex.end(),
                     p_reindex_x);
      }
    } else {
      // 5. If set_reindex, Get hashtable according to inputs and outputs.
      // We can get unique items(subset) and reindex src nodes of sample edges.
      // We also get Reindex_X for input nodes here.
      thrust::device_vector<T> subset;
      thrust::device_vector<T> orig_nodes(bs);
      thrust::device_vector<T> dst_output(outputs.size());
      thrust::copy(p_vertices, p_vertices + bs, orig_nodes.begin());
      if (set_reindex) {
        thrust::device_vector<T> reindex_nodes(bs);
        // Reindex src and orig_nodes here.
        ReindexFunc<T>(ctx, &inputs, &outputs, &subset, &orig_nodes,
                       &reindex_nodes, bs);
        auto* reindex_x = ctx.Output<Tensor>("Reindex_X");
        reindex_x->Resize({static_cast<int>(reindex_nodes.size())});
        T* p_reindex_x = reindex_x->mutable_data<T>(ctx.GetPlace());
        thrust::copy(reindex_nodes.begin(), reindex_nodes.end(), p_reindex_x);
        // Reindex dst.
        int64_t unique_dst_size = inputs.size();
        thrust::device_vector<T> unique_dst_reindex(unique_dst_size);
        thrust::sequence(unique_dst_reindex.begin(), unique_dst_reindex.end());
        thrust::device_vector<T> dst_ptr(unique_dst_size);
        thrust::exclusive_scan(output_counts.begin(), output_counts.end(),
                               dst_ptr.begin());
        constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
        constexpr int TILE_SIZE = BLOCK_WARPS * 16;
        const dim3 block(WARP_SIZE, BLOCK_WARPS);
        const dim3 grid((unique_dst_size + TILE_SIZE - 1) / TILE_SIZE);
        GetDstEdgeCUDAKernel<
            T, BLOCK_WARPS,
            TILE_SIZE><<<grid, block, 0,
                         reinterpret_cast<const platform::CUDADeviceContext&>(
                             ctx.device_context())
                             .stream()>>>(
            unique_dst_size,
            thrust::raw_pointer_cast(unique_dst_reindex.data()),
            thrust::raw_pointer_cast(output_counts.data()),
            thrust::raw_pointer_cast(dst_ptr.data()),
            thrust::raw_pointer_cast(dst_output.data()));
      } else {
        // Get dst_output.
        int64_t unique_dst_size = inputs.size();
        thrust::device_vector<T> dst_ptr(unique_dst_size);
        thrust::exclusive_scan(output_counts.begin(), output_counts.end(),
                               dst_ptr.begin());
        constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
        constexpr int TILE_SIZE = BLOCK_WARPS * 16;
        const dim3 block(WARP_SIZE, BLOCK_WARPS);
        const dim3 grid((unique_dst_size + TILE_SIZE - 1) / TILE_SIZE);
        GetDstEdgeCUDAKernel<
            T, BLOCK_WARPS,
            TILE_SIZE><<<grid, block, 0,
                         reinterpret_cast<const platform::CUDADeviceContext&>(
                             ctx.device_context())
                             .stream()>>>(
            unique_dst_size, thrust::raw_pointer_cast(inputs.data()),
            thrust::raw_pointer_cast(output_counts.data()),
            thrust::raw_pointer_cast(dst_ptr.data()),
            thrust::raw_pointer_cast(dst_output.data()));

        // Get subset.
        thrust::device_vector<T> inputs_sort(inputs.size());
        thrust::copy(inputs.begin(), inputs.end(), inputs_sort.begin());
        thrust::sort(inputs_sort.begin(), inputs_sort.end());
        thrust::device_vector<T> outputs_sort(outputs.size());
        thrust::copy(outputs.begin(), outputs.end(), outputs_sort.begin());
        thrust::sort(outputs_sort.begin(), outputs_sort.end());
        auto outputs_sort_end =
            thrust::unique(outputs_sort.begin(), outputs_sort.end());
        outputs_sort.resize(
            thrust::distance(outputs_sort.begin(), outputs_sort_end));
        thrust::device_vector<T> unique_outputs(outputs_sort.size());
        auto unique_outputs_end = thrust::set_difference(
            outputs_sort.begin(), outputs_sort.end(), inputs_sort.begin(),
            inputs_sort.end(), unique_outputs.begin());
        unique_outputs.resize(
            thrust::distance(unique_outputs.begin(), unique_outputs_end));
        subset.resize(inputs.size() + unique_outputs.size());
        thrust::copy(inputs.begin(), inputs.end(), subset.begin());
        thrust::copy(unique_outputs.begin(), unique_outputs.end(),
                     subset.begin() + inputs.size());
      }

      // 6. Get Sample_Count.
      auto* sample_count = ctx.Output<Tensor>("Sample_Count");
      sample_count->Resize({static_cast<int>(output_counts.size())});
      T* p_sample_count = sample_count->mutable_data<T>(ctx.GetPlace());
      thrust::copy(output_counts.begin(), output_counts.end(), p_sample_count);

      // 7. Give Out_Src and Out_Dst results.
      auto* out_src = ctx.Output<Tensor>("Out_Src");
      auto* out_dst = ctx.Output<Tensor>("Out_Dst");
      out_src->Resize({static_cast<int>(dst_output.size())});
      out_dst->Resize({static_cast<int>(dst_output.size())});
      T* p_out_src = out_src->mutable_data<T>(ctx.GetPlace());
      T* p_out_dst = out_dst->mutable_data<T>(ctx.GetPlace());
      thrust::copy(outputs.begin(), outputs.end(), p_out_src);
      thrust::copy(dst_output.begin(), dst_output.end(), p_out_dst);

      // 8. Get Sample_Index.
      auto* sample_index = ctx.Output<Tensor>("Sample_Index");
      sample_index->Resize({static_cast<int>(subset.size())});
      T* p_sample_index = sample_index->mutable_data<T>(ctx.GetPlace());
      thrust::copy(subset.begin(), subset.end(), p_sample_index);
    }
  }
};

}  // namespace operators
}  // namespace paddle

using CUDA = paddle::platform::CUDADeviceContext;
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(graph_khop_sampler,
                        ops::GraphKhopSamplerOpCUDAKernel<CUDA, int32_t>,
                        ops::GraphKhopSamplerOpCUDAKernel<CUDA, int64_t>);
