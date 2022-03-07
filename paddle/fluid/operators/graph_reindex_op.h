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

#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class GraphReindexOpKernel : public framework::OpKernel<T> {
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

    std::vector<T> src_outputs(num_edges);
    std::vector<T> dst_outputs(num_edges);

    std::unordered_map<T, T> node_map;
    std::vector<T> unique_nodes;
    size_t reindex_id = 0;
    for (int i = 0; i < bs; i++) {
      T node = vertices_data[i];
      unique_nodes.emplace_back(node);
      node_map[node] = reindex_id++;
    }
    // Reindex src.
    for (int i = 0; i < num_edges; i++) {
      T node = src_data[i];
      if (node_map.find(node) == node_map.end()) {
        unique_nodes.emplace_back(node);
        node_map[node] = reindex_id++;
      }
      src_outputs[i] = node_map[node];
    }
    // Reindex dst.
    size_t cnt = 0;
    for (int i = 0; i < bs; i++) {
      for (T j = 0; j < count_data[i]; j++) {
        T node = vertices_data[i];
        dst_outputs[cnt++] = node_map[node];
      }
    }

    auto* reindex_src = ctx.Output<Tensor>("Reindex_Src");
    reindex_src->Resize({num_edges});
    auto* reindex_dst = ctx.Output<Tensor>("Reindex_Dst");
    reindex_dst->Resize({num_edges});
    auto* out_nodes = ctx.Output<Tensor>("Out_Nodes");
    out_nodes->Resize({static_cast<int>(unique_nodes.size())});
    T* p_src = reindex_src->mutable_data<T>(ctx.GetPlace());
    T* p_dst = reindex_dst->mutable_data<T>(ctx.GetPlace());
    T* p_nodes = out_nodes->mutable_data<T>(ctx.GetPlace());
    std::copy(src_outputs.begin(), src_outputs.end(), p_src);
    std::copy(dst_outputs.begin(), dst_outputs.end(), p_dst);
    std::copy(unique_nodes.begin(), unique_nodes.end(), p_nodes);
  }
};

}  // namespace operators
}  // namespace paddle
