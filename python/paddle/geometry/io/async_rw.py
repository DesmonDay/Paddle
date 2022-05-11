# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021, rusty1s(github).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle.fluid import core


def async_read(src, dst, index, cpu_buffer, offset, count, name=None):
    """
    This api provides a way to read from pieces of source tensor to destination tensor 
    asynchronously. In which, we use `index`, `offset` and `count` to determine where 
    to read. `index` means the non-continuous index position of src tensor we want to read. 
    `offset` and `count` means the begin points and length of pieces of src tensor we want to read. 
    To be noted, the copy process will run asynchronously from pin memory to cuda place. 
    
    We can simply remember this as "cuda async_read from pin_memory". We should use this 
    api under GPU version PaddlePaddle.

    Args:
        src (Tensor): The source tensor, and the data type should be `float32` currently. 
                      Besides, `src` should be placed on CUDAPinnedPlace.
        dst (Tensor): The destination tensor, and the data type should be `float32` currently. 
                      Besides, `dst` should be placed on CUDAPlace. The shape of `dst` should 
                      be the same with `src` except for the first dimension.
        index (Tensor): The index tensor, and the data type should be `int64` currently. 
                      Besides, `index` should be on CPUPlace. The shape of `index` should 
                      be one-dimensional.
        cpu_buffer (Tensor): The cpu_buffer tensor, used to buffer index copy tensor temporarily.
                      The data type should be `float32` currently, and should be placed 
                      on CUDAPinnedPlace. The shape of `cpu_buffer` should be the same with 
                      `src` except for the first dimension.
        offset (Tensor): The offset tensor, and the data type should be `int64` currently. 
                      Besides, `offset` should be placed on CPUPlace. The shape of `offset` 
                      should be one-dimensional.
        count (Tensor): The count tensor, and the data type should be `int64` currently. 
                      Besides, `count` should be placed on CPUPlace. The shape of `count` 
                      should be one-dimensinal.
        name (str, optional): Name for the operation (optional, default is None).
                              For more information, please refer to :ref:`api_guide_Name`.

    Examples:

        .. code-block:: python

        import numpy as np
        import paddle
        from paddle.geometry.io import async_read

        src = paddle.rand(shape=[100, 50, 50], dtype="float32").pin_memory()
        dst = paddle.empty(shape=[100, 50, 50], dtype="float32")
        offset = paddle.to_tensor(
            np.array([0, 60], dtype="int64"), place=paddle.CPUPlace())
        count = paddle.to_tensor(
            np.array([40, 10], dtype="int64"), place=paddle.CPUPlace())
        cpu_buffer = paddle.empty(shape=[100, 50, 50], dtype="float32").pin_memory()
        index = paddle.to_tensor(
            np.array([1, 3, 5, 7, 9], dtype="int64")).cpu()
        async_read(src, dst, index, cpu_buffer, offset, count)

    """

    core.async_read(src, dst, index, cpu_buffer, offset, count)


def async_write(src, dst, offset, count, name=None):
    """
    This api provides a way to write pieces of source tensor to destination tensor 
    asynchronously. In which, we use `offset` and `count` to determine copy to where. 
    `offset` means the begin points of the copy destination of `dst`, and `count` 
    means the lengths of the copy destination of `dst`. To be noted, the copy process 
    will run asynchronously from cuda to pin memory.
    
    We can simply remember this as "gpu async_write to pin_memory". We should run this
    api under GPU version PaddlePaddle.
   
    Args:
  
        src (Tensor): The source tensor, and the data type should be `float32` currently. 
                      Besides, `src` should be placed on CUDAPlace.
        dst (Tensor): The destination tensor, and the data type should be `float32` currently. 
                      Besides, `dst` should be placed on CUDAPinnedPlace. The shape of 
                      `dst` should be the same with `src` except for the first dimension. 
        offset (Tensor): The offset tensor, and the data type should be `int64` currently. 
                      Besides, `offset` should be placed on CPUPlace. The shape of `offset` 
                      should be one-dimensional. 
        count (Tensor): The count tensor, and the data type should be `int64` currently. 
                      Besides, `count` should be placed on CPUPlace. The shape of `count` 
                      should be one-dimensinal.
        name (str, optional): Name for the operation (optional, default is None).
                              For more information, please refer to :ref:`api_guide_Name`.
  
    Examples:

        .. code-block:: python

        import numpy as np
        import paddle
        from pgl.utils.stream_pool import async_write
   
        src = paddle.rand(shape=[100, 50, 50])
        dst = paddle.empty(shape=[200, 50, 50]).pin_memory()
        offset = paddle.to_tensor(
             np.array([0, 60], dtype="int64"), place=paddle.CPUPlace())
        count = paddle.to_tensor(
             np.array([40, 60], dtype="int64"), place=paddle.CPUPlace())
        async_write(src, dst, offset, count)

    """

    core.async_write(src, dst, offset, count)
