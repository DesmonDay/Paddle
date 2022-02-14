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

#include <stdlib.h>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>
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

template <class bidiiter>
void SampleUniqueNeighborsWithEids(bidiiter src_begin, bidiiter src_end,
                                   bidiiter eid_begin, bidiiter eid_end,
                                   int num_samples) {
  int left_num = std::distance(src_begin, src_end);
  std::random_device rd;
  std::mt19937 rng{rd()};
  std::uniform_int_distribution<int> dice_distribution(
      0, std::numeric_limits<int>::max());
  for (int i = 0; i < num_samples; i++) {
    bidiiter r1 = src_begin, r2 = eid_begin;
    int random_step = dice_distribution(rng) % left_num;
    std::advance(r1, random_step);
    std::advance(r2, random_step);
    std::swap(*src_begin, *r1);
    std::swap(*eid_begin, *r2);
    ++src_begin;
    ++eid_begin;
    --left_num;
  }
}

template <typename T>
void SampleNeighbors(const T* src, const T* dst_count, const T* src_eids,
                     std::vector<T>* inputs, std::vector<T>* outputs,
                     std::vector<T>* output_counts,
                     std::vector<T>* outputs_eids, int k, bool return_eids) {
  const size_t bs = inputs->size();
  // Allocate the memory of outputs
  // Collect the neighbors size
  std::vector<std::vector<T>> out_src_vec;
  std::vector<std::vector<T>> out_eids_vec;
  // `sample_cumsum_sizes` record the start position and end position after the
  //  sample.
  std::vector<size_t> sample_cumsum_sizes(bs + 1);
  size_t total_neighbors = 0;
  // `total_neighbors` the size of output after the sample
  sample_cumsum_sizes[0] = total_neighbors;
  for (size_t i = 0; i < bs; i++) {
    T node = inputs->data()[i];
    T begin = dst_count[node];
    T end = dst_count[node + 1];
    int cap = end - begin;
    int sample_size = cap > k ? k : cap;
    total_neighbors += sample_size;
    sample_cumsum_sizes[i + 1] = total_neighbors;
    std::vector<T> out_src;
    out_src.resize(cap);
    out_src_vec.emplace_back(out_src);
    if (return_eids) {
      std::vector<T> out_eids;
      out_eids.resize(cap);
      out_eids_vec.emplace_back(out_eids);
    }
  }
  PADDLE_ENFORCE_GT(
      total_neighbors, 0,
      platform::errors::InvalidArgument("The input nodes `X` should have at "
                                        "least one neighbors, but none of the "
                                        "input nodes have neighbors."));
  output_counts->resize(bs);
  outputs->resize(total_neighbors);
  if (return_eids) {
    outputs_eids->resize(total_neighbors);
  }

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  // Sample the neighbour parallelism
  for (size_t i = 0; i < bs; i++) {
    T node = inputs->data()[i];
    T begin = dst_count[node];
    T end = dst_count[node + 1];
    int cap = end - begin;
    if (k < cap) {
      std::copy(src + begin, src + end, out_src_vec[i].begin());
      if (return_eids) {
        std::copy(src_eids + begin, src_eids + end, out_eids_vec[i].begin());
        SampleUniqueNeighborsWithEids(
            out_src_vec[i].begin(), out_src_vec[i].end(),
            out_eids_vec[i].begin(), out_eids_vec[i].end(), k);
      } else {
        SampleUniqueNeighbors(out_src_vec[i].begin(), out_src_vec[i].end(), k);
      }
      *(output_counts->data() + i) = k;
    } else {
      std::copy(src + begin, src + end, out_src_vec[i].begin());
      if (return_eids) {
        std::copy(src_eids + begin, src_eids + end, out_eids_vec[i].begin());
      }
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
    if (return_eids) {
      std::copy(out_eids_vec[i].begin(), out_eids_vec[i].begin() + sample_size,
                outputs_eids->data() + sample_cumsum_sizes[i]);
    }
  }
}

template <typename DeviceContext, typename T>
class GraphKhopSamplerOpKernel : public framework::OpKernel<T> {
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
    const size_t bs = vertices->dims()[0];

    // 2. Get unique input nodes(X).
    std::vector<T> inputs(bs);
    std::copy(p_vertices, p_vertices + bs, inputs.begin());
    if (set_unique) {
      auto unique_inputs_end = std::unique(inputs.begin(), inputs.end());
      inputs.resize(std::distance(inputs.begin(), unique_inputs_end));
    }

    // 3. Sample neighbors. We should distinguish w/o "Eids".
    // Update: We only have one sample_size value.
    std::vector<T> outputs;
    std::vector<T> output_counts;
    std::vector<T> outputs_eids;

    if (return_eids) {
      auto* src_eids = ctx.Input<Tensor>("Eids");
      const T* src_eids_data = src_eids->data<T>();
      SampleNeighbors<T>(src_data, dst_count_data, src_eids_data, &inputs,
                         &outputs, &output_counts, &outputs_eids, sample_size,
                         return_eids);
      auto* out_eids = ctx.Output<Tensor>("Out_Eids");
      out_eids->Resize({static_cast<int>(outputs_eids.size())});
      T* p_out_eids = out_eids->mutable_data<T>(ctx.GetPlace());
      std::copy(outputs_eids.begin(), outputs_eids.end(), p_out_eids);
    } else {
      SampleNeighbors<T>(src_data, dst_count_data, nullptr, &inputs, &outputs,
                         &output_counts, &outputs_eids, sample_size,
                         return_eids);
    }

    int64_t num_sample_edges =
        std::accumulate(output_counts.begin(), output_counts.end(), 0);
    PADDLE_ENFORCE_EQ(
        outputs.size(), num_sample_edges,
        platform::errors::PreconditionNotMet(
            "Number of sample edges dismatch, the sample kernel has error."));

    // 5. If set_reindex, Get hashtable according to inputs and outputs.
    // We can get unique items(subset) and reindex src nodes of sample edges.
    // We also get Reindex_X for input nodes here.
    std::vector<T> unique_nodes;
    std::vector<T> dst_output(outputs.size());
    if (set_reindex) {
      std::unordered_map<T, T> node_map;
      size_t reindex_id = 0;
      for (size_t i = 0; i < inputs.size(); i++) {
        T node = inputs[i];
        unique_nodes.emplace_back(node);
        node_map[node] = reindex_id++;
      }
      // Reindex src.
      for (size_t i = 0; i < outputs.size(); i++) {
        T node = outputs[i];
        if (node_map.find(node) == node_map.end()) {
          unique_nodes.emplace_back(node);
          node_map[node] = reindex_id++;
        }
        outputs[i] = node_map[node];
      }
      // Reindex dst.
      size_t cnt = 0;
      for (size_t i = 0; i < inputs.size(); i++) {
        for (T j = 0; j < output_counts[i]; j++) {
          T node = inputs[i];
          dst_output[cnt++] = node_map[node];
        }
      }
      // Get Reindex_X.
      auto* reindex_x = ctx.Output<Tensor>("Reindex_X");
      T* p_reindex_x = reindex_x->mutable_data<T>(ctx.GetPlace());
      for (size_t i = 0; i < bs; i++) {
        p_reindex_x[i] = node_map[p_vertices[i]];
      }
    } else {
      // Get dst_output.
      size_t cnt = 0;
      for (size_t i = 0; i < inputs.size(); i++) {
        for (T j = 0; j < output_counts[i]; j++) {
          dst_output[cnt++] = inputs[i];
        }
      }
      // Get unique_nodes
      unique_nodes.resize(inputs.size() + outputs.size());
      std::copy(inputs.begin(), inputs.end(), unique_nodes.begin());
      std::copy(outputs.begin(), outputs.end(),
                unique_nodes.begin() + inputs.size());
      auto unique_nodes_end =
          std::unique(unique_nodes.begin(), unique_nodes.end());
      unique_nodes.resize(
          std::distance(unique_nodes.begin(), unique_nodes_end));
    }

    // 6. Get Sample_Count.
    auto* sample_count = ctx.Output<Tensor>("Sample_Count");
    sample_count->Resize({static_cast<int>(output_counts.size())});
    T* p_sample_count = sample_count->mutable_data<T>(ctx.GetPlace());
    std::copy(output_counts.begin(), output_counts.end(), p_sample_count);

    // 7. Give Out_Src and Out_Dst results.
    auto* out_src = ctx.Output<Tensor>("Out_Src");
    auto* out_dst = ctx.Output<Tensor>("Out_Dst");
    out_src->Resize({static_cast<int>(dst_output.size())});
    out_dst->Resize({static_cast<int>(dst_output.size())});
    T* p_out_src = out_src->mutable_data<T>(ctx.GetPlace());
    T* p_out_dst = out_dst->mutable_data<T>(ctx.GetPlace());
    std::copy(outputs.begin(), outputs.end(), p_out_src);
    std::copy(dst_output.begin(), dst_output.end(), p_out_dst);

    // 8. Get Sample_Index.
    auto* sample_index = ctx.Output<Tensor>("Sample_Index");
    sample_index->Resize({static_cast<int>(unique_nodes.size())});
    T* p_sample_index = sample_index->mutable_data<T>(ctx.GetPlace());
    std::copy(unique_nodes.begin(), unique_nodes.end(), p_sample_index);
  }
};

}  // namespace operators
}  // namespace paddle
