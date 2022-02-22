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

#include "paddle/fluid/operators/graph_reindex_op.h"

namespace paddle {
namespace operators {

void InputShapeCheck3(const framework::DDim& dims, std::string tensor_name) {
  if (dims.size() == 2) {
    PADDLE_ENFORCE_EQ(dims[1], 1, platform::errors::InvalidArgument(
                                      "The last dim of %s should be 1 when it "
                                      "is 2D, but we get %d",
                                      tensor_name, dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        dims.size(), 1,
        platform::errors::InvalidArgument(
            "The %s should be 1D, when it is not 2D, but we get %d",
            tensor_name, dims.size()));
  }
}

class GraphReindexOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Src"), "Input", "Src", "GraphReindex");
    OP_INOUT_CHECK(ctx->HasInput("Count"), "Input", "Count", "GraphReindex");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "GraphReindex");
    OP_INOUT_CHECK(ctx->HasOutput("Reindex_Src"), "Output", "Reindex_Src",
                   "GraphReindex");
    OP_INOUT_CHECK(ctx->HasOutput("Reindex_Dst"), "Output", "Reindex_Dst",
                   "GraphReindex");
    OP_INOUT_CHECK(ctx->HasOutput("Out_Nodes"), "Output", "Out_Nodes",
                   "GraphReindex");

    // Restrict all the inputs as 1-dim tensor, or 2-dim tensor with the second
    // dim as 1.
    InputShapeCheck3(ctx->GetInputDim("Src"), "Src");
    InputShapeCheck3(ctx->GetInputDim("Count"), "Count");
    InputShapeCheck3(ctx->GetInputDim("X"), "X");

    ctx->SetOutputDim("Reindex_Src", {-1});
    ctx->SetOutputDim("Reindex_Dst", {-1});
    ctx->SetOutputDim("Out_Nodes", {-1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Src"),
        ctx.device_context());
  }
};

class GraphReindexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Src", "The src index of graph edges before reindex.");
    AddInput("Count", "The number of neighbors of dst node index of graph.");
    AddInput("X", "The input dst node index of graph.");

    AddOutput("Reindex_Src", "The src index of graph edges after reindex.");
    AddOutput("Reindex_Dst", "The dst index of graph edges after reindex.");
    AddOutput("Out_Nodes",
              "The unique output index of graph nodes before reindex");

    AddComment(R"DOC(
Graph Learning Sampling Neighbors operator, for graphsage sampling method.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(graph_reindex, ops::GraphReindexOP, ops::GraphReindexOpMaker);
REGISTER_OP_CPU_KERNEL(graph_reindex, ops::GraphReindexOpKernel<CPU, int32_t>,
                       ops::GraphReindexOpKernel<CPU, int64_t>);
