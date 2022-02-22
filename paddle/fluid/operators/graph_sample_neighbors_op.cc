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

#include "paddle/fluid/operators/graph_sample_neighbors_op.h"

namespace paddle {
namespace operators {

void InputShapeCheck2(const framework::DDim& dims, std::string tensor_name) {
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

class GraphSampleNeighborsOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Row"), "Input", "Row",
                   "GraphSampleNeighbors");
    OP_INOUT_CHECK(ctx->HasInput("Col_Ptr"), "Input", "Col_Ptr",
                   "GraphSampleNeighbors");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "GraphSampleNeighbors");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                   "GraphSampleNeighbors");
    OP_INOUT_CHECK(ctx->HasOutput("Out_Count"), "Output", "Out_Count",
                   "GraphSampleNeighbors");

    // Restrict all the inputs as 1-dim tensor, or 2-dim tensor with the second
    // dim as 1.
    InputShapeCheck2(ctx->GetInputDim("Row"), "Row");
    InputShapeCheck2(ctx->GetInputDim("Col_Ptr"), "Col_Ptr");
    InputShapeCheck2(ctx->GetInputDim("X"), "X");

    ctx->SetOutputDim("Out", {-1});
    ctx->SetOutputDim("Out_Count", {-1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Row"),
        ctx.device_context());
  }
};

class GraphSampleNeighborsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Row",
             "One of the components of the CSC format of the input graph");
    AddInput("Col_Ptr",
             "One of the components of the CSC format of the input graph");
    AddInput("X", "The input center nodes index tensor.");

    AddOutput("Out", "The neighbors of input nodes X after sampling.");
    AddOutput("Out_Count", "The number of sample neighbors of input nodes.");

    AddAttr<int>(
        "sample_size",
        "The sample size of graph sample neighbors method. "
        "Set default value as -1, means return all neighbors of nodes.")
        .SetDefault(-1);

    AddComment(R"DOC(
Graph Learning Sampling Neighbors operator, for graphsage sampling method.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(graph_sample_neighbors, ops::GraphSampleNeighborsOP,
                  ops::GraphSampleNeighborsOpMaker);
REGISTER_OP_CPU_KERNEL(graph_sample_neighbors,
                       ops::GraphSampleNeighborsOpKernel<CPU, int32_t>,
                       ops::GraphSampleNeighborsOpKernel<CPU, int64_t>);
