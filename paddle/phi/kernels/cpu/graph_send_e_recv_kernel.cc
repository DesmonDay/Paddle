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

#include "paddle/phi/kernels/graph_send_e_recv_kernel.h"

#include <algorithm>
#include <set>
#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/graph_send_e_recv_funcs.h"
#include "paddle/phi/kernels/impl/graph_send_e_recv_kernel_impl.h"

namespace phi {

template <typename T, typename IndexT, typename ComputeFunctor>
void GraphSendERecvSumCpuKernel(const BroadCastInfo& bcast,
                                const T* x_data,
                                const T* e_data,
                                const IndexT* src_indices,
                                const IndexT* dst_indices,
                                T* output,
                                int64_t index_size,
                                ComputeFunctor cfunctor) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int i = 0; i < index_size; i++) {
    IndexT src = src_indices[i];
    IndexT dst = dst_indices[i];
    T* out_off = output + dst * bcast.out_len;
    const T* x_off = x_data + src * bcast.l_len;
    const T* e_off = e_data + i * bcast.r_len;
    for (int64_t j = 0; j < bcast.out_len; j++) {
      int64_t x_add = bcast.use_bcast ? bcast.l_offset[j] : j;
      int64_t e_add = bcast.use_bcast ? bcast.r_offset[j] : j;
      T val = cfunctor(x_off[x_add], e_off[e_add]);
      if (val != 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
        out_off[j] += val;
      }
    }
  }
}

// template <typename T, typename IndexT, typename ComputeFunctor>
// void GraphSendERecvMaxCpuKernel()

template <typename Context, typename T, typename IndexT>
void GraphSendERecvOpKernelLaunchHelper(const Context& ctx,
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
  memset(out_data, 0, memset_bytes);

  if (index_size == 0) return;
  const auto& bcast_info = phi::CalcBCastInfo(x.dims(), e.dims());
  const T* x_data = x.data<T>();
  const T* e_data = e.data<T>();
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();
  if (pool_type == "SUM" || pool_type == "MEAN") {
    if (compute_type == "ADD") {
      GraphAddFunctor<T> add_functor;
      GraphSendERecvSumCpuKernel<T, IndexT, GraphAddFunctor<T>>(bcast_info,
                                                                x_data,
                                                                e_data,
                                                                s_index,
                                                                d_index,
                                                                out_data,
                                                                index_size,
                                                                add_functor);
    } else if (compute_type == "MUL") {
      GraphMulFunctor<T> mul_functor;
      GraphSendERecvSumCpuKernel<T, IndexT, GraphMulFunctor<T>>(bcast_info,
                                                                x_data,
                                                                e_data,
                                                                s_index,
                                                                d_index,
                                                                out_data,
                                                                index_size,
                                                                mul_functor);
    }
    if (pool_type == "MEAN") {
      int* dst_count_data = ctx.template Alloc<int>(dst_count);
      memset(dst_count_data, 0, dst_count->dims()[0] * sizeof(int));
      for (int i = 0; i < index_size; i++) {
        IndexT dst_idx = d_index[i];
        dst_count_data[dst_idx] += 1;
      }
      for (int i = 0; i < out_dims[0]; i++) {
        if (dst_count_data[i] == 0) continue;
        auto out_slice = out->Slice(i, i + 1);
        auto eigen_out = phi::EigenVector<T>::Flatten(out_slice);
        eigen_out = eigen_out / static_cast<T>(dst_count_data[i]);
      }
    }
  } else if (pool_type == "MIN" || pool_type == "MAX") {
    /*if (compute_type == "ADD") {
      GraphAddFunctor<T> add_funtor;
    } else if (compute_type == "MUL") {
      GraphMulFunctor<T> mul_functor;
    }*/
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
    GraphSendERecvOpKernelLaunchHelper<Context, T, int32_t>(ctx,
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
    GraphSendERecvOpKernelLaunchHelper<Context, T, int64_t>(ctx,
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
                   CPU,
                   ALL_LAYOUT,
                   phi::GraphSendERecvKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
