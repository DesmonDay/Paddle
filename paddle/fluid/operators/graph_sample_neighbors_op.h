/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <class bidiiter>
void SampleUniqueNeighbors(bidiiter begin, bidiiter end, int num_samples) {
  int left_num = std::distance(begin, end);
  std::random_device rd;
  std::mt19937 rng{rd()};
  std::uniform_int_distribution<int> dice_distribution(
      0, std::numeric_limits<int>::max());
  for (int i = 0; i < num_samples; i++) {
    bidiiter r = begin;
    int random_step = dice_distribution(rng) % left_num;
    std::advance(r, random_step);
    std::swap(*begin, *r);
    ++begin;
    --left_num;
  }
}

template <typename T>
void SampleNeighbors(const T* src, const T* dst_count, std::vector<T>* inputs,
                     std::vector<T>* outputs, std::vector<T>* output_counts,
                     int k) {
  const size_t bs = inputs->size();
  // Allocate the memory of outputs
  // Collect the neighbors size
  std::vector<std::vector<T>> out_src_vec;
  // `sample_cumsum_sizes` record the start position and end position after
  // sample.
  std::vector<size_t> sample_cumsum_sizes(bs + 1);
  size_t total_neighbors = 0;
  // `total_neighbors` the size of output after sample
  sample_cumsum_sizes[0] = total_neighbors;
  for (size_t i = 0; i < bs; i++) {
    T node = inputs->data()[i];
    int cap = dst_count[node + 1] - dst_count[node];
    int sample_size = cap > k ? k : cap;
    total_neighbors += sample_size;
    sample_cumsum_sizes[i + 1] = total_neighbors;
    std::vector<T> out_src;
    out_src.resize(cap);
    out_src_vec.emplace_back(out_src);
  }

  output_counts->resize(bs);
  outputs->resize(total_neighbors);

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  // Sample the neighbors parallelism
  for (size_t i = 0; i < bs; i++) {
    T node = inputs->data()[i];
    T begin = dst_count[node], end = dst_count[node + 1];
    int cap = end - begin;
    if (k < cap) {
      std::copy(src + begin, src + end, out_src_vec[i].begin());
      SampleUniqueNeighbors(out_src_vec[i].begin(), out_src_vec[i].end(), k);
      *(output_counts->data() + i) = k;
    } else {
      std::copy(src + begin, src + end, out_src_vec[i].begin());
      *(output_counts->data() + i) = cap;
    }
  }

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  // Copy the results parallelism
  for (size_t i = 0; i < bs; i++) {
    int sample_size = sample_cumsum_sizes[i + 1] - sample_cumsum_sizes[i];
    std::copy(out_src_vec[i].begin(), out_src_vec[i].begin() + sample_size,
              outputs->data() + sample_cumsum_sizes[i]);
  }
}

template <typename DeviceContext, typename T>
class GraphSampleNeighborsOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* src = ctx.Input<Tensor>("Row");
    auto* dst_count = ctx.Input<Tensor>("Col_Ptr");
    auto* vertices = ctx.Input<Tensor>("X");
    int sample_size = ctx.Attr<int>("sample_size");

    const T* src_data = src->data<T>();
    const T* dst_count_data = dst_count->data<T>();
    const T* p_vertices = vertices->data<T>();
    const size_t bs = vertices->dims()[0];

    std::vector<T> inputs(bs);
    std::copy(p_vertices, p_vertices + bs, inputs.begin());
    std::vector<T> outputs;
    std::vector<T> output_counts;

    SampleNeighbors<T>(src_data, dst_count_data, &inputs, &outputs,
                       &output_counts, sample_size);

    auto* out = ctx.Output<Tensor>("Out");
    out->Resize({static_cast<int>(outputs.size())});
    T* p_out = out->mutable_data<T>(ctx.GetPlace());
    std::copy(outputs.begin(), outputs.end(), p_out);
    auto* out_count = ctx.Output<Tensor>("Out_Count");
    out_count->Resize({static_cast<int>(output_counts.size())});
    T* p_out_count = out_count->mutable_data<T>(ctx.GetPlace());
    std::copy(output_counts.begin(), output_counts.end(), p_out_count);
  }
};

}  // namespace operators
}  // namespace paddle
