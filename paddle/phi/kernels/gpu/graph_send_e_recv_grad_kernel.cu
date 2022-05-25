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
#include "paddle/phi/kernels/graph_send_e_recv_grad_kernel.h"
#include "paddle/phi/kernels/impl/graph_send_e_recv_funcs.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename Context, typename T, typename IndexT>
void CalculateXEGradForMinMax(const Context& ctx,
                              const T* out_grad,
                              const T* x_data,
                              const T* e_data,
                              const phi::DDim& out_grad_dims,
                              const phi::DDim& x_dims,
                              const phi::DDim& e_dims,
                              const IndexT* s_index,
                              const IndexT* d_index,
                              const std::string& compute_type,
                              const std::string& pool_type,
                              int64_t index_size,
                              int64_t slice_size,
                              T* x_grad,
                              T* e_grad,
                              const DenseTensor* out = nullptr) {
  const T* out_data = out->data<T>();
  const auto& bcast_info = CalcBCastInfo(x_dims, e_dims);
  thrust::device_vector<int64_t> l_bcastoff, r_bcastoff;
  if (bcast_info.use_bcast) {
    CopyBCastOff(bcast_info, l_bcastoff, r_bcastoff);
  }

  int64_t out_len = bcast_info.out_len;
  const int ntx = FindNumThreads(out_len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (out_len + ntx - 1) / ntx;
  const int nby = (index_size + nty - 1) / nty;
  const dim3 grid(nbx, nby);
  const dim3 block(ntx, nty);

  if (compute_type == "ADD") {
    ManipulateMinMaxGradCUDAKernelForAdd<
        T,
        IndexT><<<grid, block, 0, ctx.stream()>>>(
        x_data,
        e_data,
        out_data,
        out_grad,
        d_index,
        s_index,
        thrust::raw_pointer_cast(l_bcastoff.data()),
        thrust::raw_pointer_cast(r_bcastoff.data()),
        x_grad,
        e_grad,
        index_size,
        bcast_info.l_len,
        bcast_info.r_len,
        out_len,
        bcast_info.use_bcast);
  } else if (compute_type == "MUL") {
    ManipulateMinMaxGradCUDAKernelForMul<
        T,
        IndexT><<<grid, block, 0, ctx.stream()>>>(
        x_data,
        e_data,
        out_data,
        out_grad,
        d_index,
        s_index,
        thrust::raw_pointer_cast(l_bcastoff.data()),
        thrust::raw_pointer_cast(r_bcastoff.data()),
        x_grad,
        e_grad,
        index_size,
        bcast_info.l_len,
        bcast_info.r_len,
        out_len,
        bcast_info.use_bcast);
  }
}

template <typename Context, typename T, typename IndexT>
void CalculateXGrad(const Context& ctx,
                    const T* out_grad,
                    const T* x_data,
                    const T* e_data,
                    const phi::DDim& out_grad_dims,
                    const phi::DDim& x_dims,
                    const phi::DDim& e_dims,
                    const IndexT* s_index,
                    const IndexT* d_index,
                    const std::string& compute_type,
                    const std::string& pool_type,
                    int64_t index_size,
                    int64_t slice_size,
                    T* x_grad,
                    const DenseTensor* dst_count = nullptr,
                    const DenseTensor* out = nullptr) {
#ifdef PADDLE_WITH_HIP
  int block = 256;
#else
  int block = 1024;
#endif
  int64_t n = slice_size * index_size;
  int max_grid_dimx = ctx.GetCUDAMaxGridDimSize()[0];
  int64_t grid_tmp = (n + block - 1) / block;
  int64_t grid = grid_tmp < max_grid_dimx ? grid_tmp : max_grid_dim;
  if (pool_type == "SUM") {
    if (compute_type == "ADD") {
      GraphSendRecvSumCUDAFunctor<T, IndexT> functor;
      GraphSendRecvCUDAKernel<T,
                              IndexT,
                              GraphSendRecvSumCUDAFunctor<
                                  T,
                                  IndexT>><<<grid, block, 0, ctx.stream()>>>(
          out_grad, d_index, s_index, x_grad, index_size, slice_size, functor);
    } else if (compute_type == "MUL") {
      const auto& bcast_info = CalcBCastInfo(out_grad_dims, e_dims);
      thrust::device_vector<int64_t> l_bcastoff, r_bcastoff;
      if (bcast_info.use_bcast) {
        CopyBCastOff(bcast_info, l_bcastoff, r_bcastoff);
      }
      int64_t out_len = bcast_info.out_len;
      const int ntx = FindNumThreads(out_len);
      const int nty = CUDA_MAX_NUM_THREADS / ntx;
      const int nbx = (out_len + ntx - 1) / ntx;
      const int nby = (index_size + nty - 1) / nty;
      const dim3 grid_(nbx, nby);
      const dim3 block_(ntx, nty);
      MultiplyFunctor<T> mul_functor;
      GraphSendERecvCUDAKernel<
          T,
          IndexT,
          GraphSendERecvSumCUDAFunctor<T>,
          MultiplyFunctor<T>><<<grid_, block_, 0, ctx.stream()>>>(
          out_grad,
          e_data,
          d_index,
          s_index,
          thrust::raw_pointer_cast(l_bcastoff.data()),
          thrust::raw_pointer_cast(r_bcastoff.data()),
          x_grad,
          index_size,
          bcast_info.l_len,
          bcast_info.r_len,
          out_len,
          bcast_info.use_bcast,
          mul_functor,
          sum_functor);
    }
  } else if (pool_type == "MEAN") {
    const int* s_count = dst_count->data<int>();
    if (compute_type == "ADD") {
      ManipulateMeanGradCUDAKernel<T, IndexT><<<grid, block, 0, ctx.stream()>>>(
          out_grad, d_index, s_index, x_grad, index_size, slice_size, s_count);
    } else if (compute_type == "MUL") {
      const auto& bcast_info = CalcBCastInfo(out_grad_dims, e_dims);
      thrust::device_vector<int64_t> l_bcastoff, r_bcastoff;
      if (bcast_info.use_bcast) {
        CopyBCastOff(bcast_info, l_bcastoff, r_bcastoff);
      }
      int64_t out_len = bcast_info.out_len;
      const int ntx = FindNumThreads(out_len);
      const int nty = CUDA_MAX_NUM_THREADS / ntx;
      const int nbx = (out_len + ntx - 1) / ntx;
      const int nby = (index_size + nty - 1) / nty;
      const dim3 grid_(nbx, nby);
      const dim3 block_(ntx, nty);
      ManipulateMeanGradCUDAKernelV2<
          T,
          IndexT><<<grid_, block_, 0, ctx.stream()>>>(
          out_grad,
          e_data,
          d_index,
          s_index,
          s_count,
          thrust::raw_pointer_cast(l_bcastoff.data()),
          thrust::raw_pointer_cast(r_bcastoff.data()),
          x_grad,
          index_size,
          bcast_info.l_len,
          bcast_info.r_len,
          out_len,
          bcast_info.use_bcast);
    }
  }
}

template <typename Context, typename T, typename IndexT>
void GraphSendERecvGradOpCUDAKernelLaunchHelper(
    const Context& ctx,
    const DenseTensor& out_grad,
    const DenseTensor& x,
    const DenseTensor& e,
    const DenseTensor& src_index,
    const DenseTensor& dst_index,
    const std::string& compute_type,
    const std::string& pool_type,
    DenseTensor* x_grad,
    DenseTensor* e_grad,
    const DenseTensor* dst_count = nullptr,
    const DenseTensor* out = nullptr) {
  const int& index_size = dst_index.dims()[0];

  ctx.template Alloc<T>(x_grad);
  T* x_grad_data = x_grad->data<T>();
  ctx.template Alloc<T>(e_grad);
  T* e_grad_data = e_grad->data<T>();
  const auto& x_dims = x.dims();
  const auto& e_dims = e.dims();
  int64_t memset_size_x = 1, memset_size_e = 1;
  int64_t slice_size = 1;
  for (int i = 0; i < x_dims.size(); i++) {
    memset_size_x *= x_dims[i];
    if (i > 0) slice_size *= x_dims[i];
  }
  for (int i = 0; i < e_dims.size(); i++) {
    memset_size_e *= e_dims[i];
  }
  const size_t& memset_bytes_x = memset_size_x * sizeof(T);
  const size_t& memset_bytes_e = memset_size_e * sizeof(T);
#ifdef PADDLE_WITH_HIP
  hipMemset(x_grad_data, 0, memset_bytes_x);
  hipMemset(e_grad_data, 0, memset_bytes_e);
#else
  cudaMemset(x_grad_data, 0, memset_bytes_x);
  cudaMemset(e_grad_data, 0, memset_bytes_e);
#endif

  if (index_size == 0) return;

  const T* out_grad_data = out_grad.data<T>();
  const T* x_data = x.data<T>();
  const T* e_data = e.data<T>();
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();

  if (pool_type == "SUM" || pool_type == "MEAN") {
    CalculateXGrad<Context, T, IndexT>(ctx,
                                       out_grad_data,
                                       x_data,
                                       e_data,
                                       out_grad.dims(),
                                       x_dims,
                                       e_dims,
                                       s_index,
                                       d_index,
                                       compute_type,
                                       pool_type,
                                       index_size,
                                       slice_size,
                                       x_grad_data,
                                       dst_count,
                                       out);
    CalculateEGrad<Context, T, IndexT>();
  } else if (pool_type == "MIN" || pool_type == "MAX") {
    CalculateXEGradForMinMax<Context, T, IndexT>(ctx,
                                                 out_grad_data,
                                                 x_data,
                                                 e_data,
                                                 out_grad.dims(),
                                                 x_dims,
                                                 e_dims,
                                                 s_index,
                                                 d_index,
                                                 compute_type,
                                                 pool_type,
                                                 index_size,
                                                 slice_size,
                                                 x_grad_data,
                                                 e_grad_data,
                                                 out);
  }
}

template <typename T, typename Context>
void GraphSendERecvGradKernel(const Context& ctx,
                              const DenseTensor& x,
                              const DenseTensor& e,
                              const DenseTensor& src_index,
                              const DenseTensor& dst_index,
                              paddle::optional<const DenseTensor&> out,
                              paddle::optional<const DenseTensor&> dst_count,
                              const DenseTensor& out_grad,
                              const std::string& compute_type,
                              const std::string& pool_type,
                              DenseTensor* x_grad,
                              DenseTensor* e_grad) {
  auto index_type = src_index.dtype();
  if (index_type == phi::DataType::INT32) {
    GraphSendERecvGradOpCUDAKernelLaunchHelper<Context, T, int32_t>(
        ctx,
        out_grad,
        x,
        e,
        src_index,
        dst_index,
        compute_type,
        pool_type,
        x_grad,
        e_grad,
        dst_count.get_ptr(),
        out.get_ptr());
  } else if (index_type == phi::DataType::INT64) {
    GraphSendERecvGradOpCUDAKernelLaunchHelper<Context, T, int64_t>(
        ctx,
        out_grad,
        x,
        e,
        src_index,
        dst_index,
        compute_type,
        pool_type,
        x_grad,
        e_grad,
        dst_count.get_ptr(),
        out.get_ptr());
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(graph_send_e_recv_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::GraphSendERecvGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
