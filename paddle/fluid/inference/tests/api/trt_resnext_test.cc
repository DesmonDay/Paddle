/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "gflags/gflags.h"
#include "paddle/fluid/inference/tests/api/trt_test_helper.h"

namespace paddle {
namespace inference {

TEST(TensorRT_resnext50, compare) {
  std::string model_dir = FLAGS_infer_model + "/resnext50";
  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir);
  config.DisableGlogInfo();
  auto predictor = CreatePaddlePredictor(config);
  compare(model_dir, /* use_tensorrt */ true);
}

}  // namespace inference
}  // namespace paddle
