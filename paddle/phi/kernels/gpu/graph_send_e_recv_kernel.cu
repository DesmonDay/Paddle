// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/gpu/graph_send_e_recv_funcs.h"
#include "paddle/phi/kernels/gpu/graph_send_recv_funcs.h"
#include "paddle/phi/kernels/graph_send_e_recv_kernel.h"
#include "paddle/phi/kernels/impl/graph_send_e_recv_kernel_impl.h"

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <algorithm>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"

namespace phi {

template <typename Context, typename T, typename IndexT>
void GraphSendERecvOpCUDAKernelLaunchHelper(const Context& ctx,
                                            const DenseTensor& x,
                                            const DenseTensor& e,
                                            const DenseTensor& src_index,
                                            const DenseTensor& dst_index,
                                            const std::string& compute_type,
                                            const std::string& pool_type,
                                            int64_t out_size,
                                            DenseTensor* out,
                                            DenseTensor* dst_count = nullptr) {
  const int& index_size = src_index.dims()[0];
  ctx.template Alloc<T>(out);
  T* out_data = out->data<T>();
  auto out_dims = out->dims();
  int64_t memset_size = 1;
  for (int i = 0; i < out_dims.size(); i++) {
    memset_size *= out_dims[i];
  }
  const size_t& memset_bytes = memset_size * sizeof(T);
  if (pool_type == "SUM" || pool_type == "MEAN") {
#ifdef PADDLE_WITH_HIP
    hipMemset(out_data, 0, memset_bytes);
#else
    cudaMemset(out_data, 0, memset_bytes);
#endif
  } else if (pool_type == "MAX") {
    thrust::device_ptr<T> out_data_ptr(out_data);
    thrust::fill(thrust::device,
                 out_data_ptr,
                 out_data_ptr + memset_size,
                 std::numeric_limits<T>::min());
  } else if (pool_type == "MIN") {
    thrust::device_ptr<T> out_data_ptr(out_data);
    thrust::fill(thrust::device,
                 out_data_ptr,
                 out_data_ptr + memset_size,
                 std::numeric_limits<T>::max());
  }

  if (index_size == 0) return;

  const auto& bcast_info = CalcBCastInfo(x.dims(), e.dims(), compute_type);
  const T* x_data = x.data<T>();
  const T* e_data = e.data<T>();
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();

  thrust::device_vector<int64_t> x_bcastoff, e_bcastoff;
  if (bcast_info.use_bcast) {
    CopyBCastOff(bcast_info, x_bcastoff, e_bcastoff);
  }

  int64_t out_len = bcast_info.out_len;
  const int ntx = FindNumThreads(out_len);  // 一个block包含的Thread数
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (out_len + ntx - 1) / ntx;
  const int nby = (index_size + nty - 1) / nty;
  const dim3 grid(nbx, nby);
  const dim3 block(ntx, nty);
  int64_t input_size = x.dims()[0];
#ifdef PADDLE_WITH_HIP
  int block_ = 256;
#else
  int block_ = 1024;
#endif
  if (pool_type == "SUM" || pool_type == "MEAN") {
    GraphSendERecvSumCUDAFunctor<T> sum_functor;
    if (compute_type == "ADD") {
      AddFunctor<T> add_funtor;
      GraphSendERecvCUDAKernel<T,
                               IndexT,
                               GraphSendERecvSumCUDAFunctor<T>,
                               AddFunctor<T>><<<grid, block, 0, ctx.stream()>>>(
          x_data,
          e_data,
          s_index,
          d_index,
          thrust::raw_pointer_cast(x_bcastoff.data());
          thrust::raw_pointer_cast(e_bcastoff.data());
          out_data,
          index_size,
          bcast_info.l_len,
          bcast_info.r_len,
          out_len,
          bcast_info.use_bcast,
          add_funtor,
          sum_functor);
    } else if (compute_type == "MUL") {
      MultiplyFunctor<T> mul_functor;
      GraphSendERecvCUDAKernel<
          T,
          IndexT,
          GraphSendERecvSumCUDAFunctor<T>,
          MultiplyFunctor<T>><<<grid, block, 0, ctx.stream()>>>(
          x_data,
          e_data,
          s_index,
          d_index,
          thrust::raw_pointer_cast(x_bcastoff.data());
          thrust::raw_pointer_cast(e_bcastoff.data());
          out_data,
          index_size,
          bcast_info.l_len,
          bcast_info.r_len,
          out_len,
          bcast_info.use_bcast,
          mul_functor,
          sum_functor);
    }
    if (pool_type == "MEAN") {
      ctx.template Alloc(dst_count);
      int32_t* dst_count_data = dst_count->data<int32_t>();
      if (out_size > 0) {
        input_size = out_size;
      }
#ifdef PADDLE_WITH_HIP
      hipMemset(dst_count_data, 0, input_size * sizeof(int));
#else
      cudaMemset(dst_count_data, 0, input_size * sizeof(int));
#endif
      int64_t grid_count = (index_size + block_ - 1) / block_;
      ComputeCountCUDAKernel<T,
                             IndexT><<<grid_count, block_, 0, ctx.stream()>>>(
          dst_count_data, d_index, index_size);

      int64_t grid_mean = (input_size * out_len + block_ - 1) / block_;
      int64_t grid_mean_ =
          grid_mean < max_grid_dimx ? grid_mean : max_grid_dimx;
      ManipulateMeanCUDAKernel<T><<<grid_mean_, block_, 0, ctx.stream()>>>(
          out_data, dst_count_data, input_size, out_len);
    }
  } else if (pool_type == "MAX") {
    GraphSendERecvMaxCUDAFunctor<T> max_functor;
    if (compute_type == "ADD") {
      AddFunctor<T> add_funtor;
      GraphSendERecvCUDAKernel<T,
                               IndexT,
                               GraphSendERecvMaxCUDAFunctor<T>,
                               AddFunctor<T>><<<grid, block, 0, ctx.stream()>>>(
          x_data,
          e_data,
          s_index,
          d_index,
          thrust::raw_pointer_cast(x_bcastoff.data());
          thrust::raw_pointer_cast(e_bcastoff.data());
          out_data,
          index_size,
          bcast_info.l_len,
          bcast_info.r_len,
          out_len,
          bcast_info.use_bcast,
          add_funtor,
          max_functor);
    } else if (compute_type == "MUL") {
      MultiplyFunctor<T> mul_functor;
      GraphSendERecvCUDAKernel<
          T,
          IndexT,
          GraphSendERecvMaxCUDAFunctor<T>,
          MultiplyFunctor<T>><<<grid, block, 0, ctx.stream()>>>(
          x_data,
          e_data,
          s_index,
          d_index,
          thrust::raw_pointer_cast(x_bcastoff.data());
          thrust::raw_pointer_cast(e_bcastoff.data());
          out_data,
          index_size,
          bcast_info.l_len,
          bcast_info.r_len,
          out_len,
          bcast_info.use_bcast,
          mul_functor,
          max_functor);
    }
    if (out_size > 0) {
      input_size = out_size;
    }
    int64_t grid_max = (input_size * out_len + block_ - 1) / block_;
    int64_t grid_max_ = grid_max < max_grid_dimx ? grid_max : max_grid_dimx;
    InputResetMaxCUDAKernel<T><<<grid_max_, block_, 0, ctx.stream()>>>(
        out_data, input_size, out_len);
  } else if (pool_type == "MIN") {
    GraphSendERecvMinCUDAFunctor<T> min_functor;
    if (compute_type == "ADD") {
      AddFunctor<T> add_funtor;
      GraphSendERecvCUDAKernel<T,
                               IndexT,
                               GraphSendERecvMinCUDAFunctor<T>,
                               AddFunctor<T>><<<grid, block, 0, ctx.stream()>>>(
          x_data,
          e_data,
          s_index,
          d_index,
          thrust::raw_pointer_cast(x_bcastoff.data());
          thrust::raw_pointer_cast(e_bcastoff.data());
          out_data,
          index_size,
          bcast_info.l_len,
          bcast_info.r_len,
          out_len,
          bcast_info.use_bcast,
          add_funtor,
          min_functor);
    } else if (compute_type == "MUL") {
      MultiplyFunctor<T> mul_functor;
      GraphSendERecvCUDAKernel<
          T,
          IndexT,
          GraphSendERecvMinCUDAFunctor<T>,
          MultiplyFunctor<T>><<<grid, block, 0, ctx.stream()>>>(
          x_data,
          e_data,
          s_index,
          d_index,
          thrust::raw_pointer_cast(x_bcastoff.data());
          thrust::raw_pointer_cast(e_bcastoff.data());
          out_data,
          index_size,
          bcast_info.l_len,
          bcast_info.r_len,
          out_len,
          bcast_info.use_bcast,
          mul_functor,
          min_functor);
    }
    if (out_size > 0) {
      input_size = out_size;
    }
    int64_t grid_min = (input_size * out_len + block_ - 1) / block_;
    int64_t grid_min_ = grid_min < max_grid_dimx ? grid_min : max_grid_dimx;
    InputResetMinCUDAKernel<T><<<grid_min_, block_, 0, ctx.stream()>>>(
        out_data, input_size, out_len);
  }
}

template <typename T, typename Context>
void GraphSendERecvKernel(const Context& ctx,
                          const DenseTensor& x,
                          const DenseTensor& e,
                          const DenseTensor& src_index,
                          const DenseTensor& dst_index,
                          const std::string& compute_type,
                          const std::string& pool_type,
                          int64_t out_size,
                          DenseTensor* out,
                          DenseTensor* dst_count) {
  auto index_type = src_index.dtype();
  if (index_type == phi::DataType::INT32) {
    GraphSendERecvOpCUDAKernelLaunchHelper<Context, T, int32_t>(ctx,
                                                                x,
                                                                e,
                                                                src_index,
                                                                dst_index,
                                                                compute_type,
                                                                pool_type,
                                                                out_size,
                                                                out,
                                                                dst_count);
  } else if (index_type == phi::DataType::INT64) {
    GraphSendERecvOpCUDAKernelLaunchHelper<Context, T, int64_t>(ctx,
                                                                x,
                                                                e,
                                                                src_index,
                                                                dst_index,
                                                                compute_type,
                                                                pool_type,
                                                                out_size,
                                                                out,
                                                                dst_count);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(graph_send_e_recv,
                   GPU,
                   ALL_LAYOUT,
                   phi::GraphSendERecvKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
