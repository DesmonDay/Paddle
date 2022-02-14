#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid import core
from paddle import _C_ops


def graph_khop_sampler(row,
                       colptr,
                       input_nodes,
                       sample_size=-1,
                       sorted_eids=None,
                       return_eids=False,
                       set_unique=True,
                       set_reindex=False,
                       name=None):
    """
    Graph Khop Sampler API.

    This API is mainly used in Graph Learning domain, and the main purpose is to 
    provide high performance graph khop sampling method with subgraph reindex step.
    For example, we get the CSC(Compressed Sparse Column) format of the input graph
    edges as `row` and `colptr`, so as to covert graph data into a suitable format 
    for sampling. And the `input_nodes` means the nodes we need to sample neighbors for,
    and `sample_size` means the number of neighbors we want to sample. 

    Args:
        row (Tensor): One of the components of the CSC format of the input graph, and 
                      the shape should be [num_edges, 1] or [num_edges]. The available
                      data type is int32, int64.
        colptr (Tensor): One of the components of the CSC format of the input graph,
                         and the shape should be [num_nodes + 1, 1] or [num_nodes]. 
                         The data type should be the same with `row`.
        input_nodes (Tensor): The input nodes we need to sample neighbors for, and the 
                              data type should be the same with `row`.
        sample_size (int): The number of neighbors to be sampled for each node. The
                           default value is -1, which means sampling all the neighbors 
                           of the input nodes.
        sorted_eids (Tensor): The sorted edge ids, should not be None when `return_eids`
                              is True. The shape should be [num_edges, 1], and the data
                              type should be the same with `row`.
        return_eids (bool): Whether to return the id of the sample edges. Default is False.
                            If set True, the return value `edge_eids` will be initialized.
        set_unique (bool): Whether to unique the input sample nodes.
        set_reindex (bool): Whether to reindex the sample edges and nodes. Default is False.
                            If set True, the return values `edge_src` and `edge_dst` will be 
                            reindexed, and `reindex_nodes` will be initialized. If set True, 
                            `set_unique` must be set True, too.
        name (str, optional): Name for the operation (optional, default is None).
                              For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        edge_src (Tensor): The src index of the output edges, also means the first column of 
                           the edges. The shape is [num_sample_edges] currently.
        edge_dst (Tensor): The dst index of the output edges, also means the second column
                           of the edges. The shape is [num_sample_edges] currently.
        sample_index (Tensor): The original id of the input nodes and sampled neighbor nodes.
        sample_count (Tensor): Return the number of sample neighbors of input nodes or after uniquement.
        reindex_nodes (Tensor): Return the reindex id of the input nodes if `set_reindex` is True,
                                otherwise `reindex_nodes`will not be initialized.
        edge_eids (Tensor): Return the id of the sample edges if `return_eids` is True, otherwise
                            `edge_eids` will not be initialized.

    Examples:
        
        .. code-block:: python

        import paddle

        row = [3, 7, 0, 9, 1, 4, 2, 9, 3, 9, 1, 9, 7]
        colptr = [0, 2, 4, 5, 6, 7, 9, 11, 11, 13, 13]
        nodes = [0, 8, 1, 2]
        sample_size = 2
        row = paddle.to_tensor(row, dtype="int64")
        colptr = paddle.to_tensor(colptr, dtype="int64")
        nodes = paddle.to_tensor(nodes, dtype="int64")
        
        edge_src, edge_dst, sample_index, sample_count, reindex_nodes, edge_eids = \
            paddle.incubate.graph_khop_sampler(row, colptr, nodes, sample_size)

    """

    if in_dygraph_mode():
        if return_eids and sorted_eids is None:
            raise ValueError(f"`sorted_eid` should not be None "
                             f"if `return_eids` is True.")
        if set_reindex is True and set_unique is False:
            raise ValueError(f"`set_unique` should be set True "
                             f"if `set_reindex` is True.")
        edge_src, edge_dst, sample_index, sample_count, reindex_nodes, edge_eids = \
            _C_ops.graph_khop_sampler(row, colptr, sorted_eids, input_nodes,
                                      "sample_size", sample_size,
                                      "return_eids", return_eids,
                                      "set_unique", set_unique,
                                      "set_reindex", set_reindex)
        return edge_src, edge_dst, sample_index, sample_count, reindex_nodes, edge_eids

    check_variable_and_dtype(row, "Row", ("int32", "int64"),
                             "graph_khop_sampler")

    if return_eids and sorted_eids is None:
        raise ValueError(f"`sorted_eid` should not be None "
                         f"if return_eids is True.")
    check_variable_and_dtype(colptr, "Col_Ptr", ("int32", "int64"),
                             "graph_khop_sampler")
    check_variable_and_dtype(input_nodes, "X", ("int32", "int64"),
                             "graph_khop_sampler")
    check_variable_and_dtype(sorted_eids, "Eids", ("int32", "int64"),
                             "graph_khop_sampler")

    helper = LayerHelper("graph_khop_sampler", **locals())
    edge_src = helper.create_variable_for_type_inference(dtype=row.dtype)
    edge_dst = helper.create_variable_for_type_inference(dtype=row.dtype)
    sample_index = helper.create_variable_for_type_inference(dtype=row.dtype)
    sample_count = helper.create_variable_for_type_inference(dtype=row.dtype)
    reindex_nodes = helper.create_variable_for_type_inference(dtype=row.dtype)
    edge_eids = helper.create_variable_for_type_inference(dtype=row.dtype)
    helper.append_op(
        type="graph_khop_sampler",
        inputs={
            "Row": row,
            "Col_Ptr": colptr,
            "Eids": sorted_eids,
            "X": input_nodes
        },
        outputs={
            "Out_Src": edge_src,
            "Out_Dst": edge_dst,
            "Sample_Index": sample_index,
            "Sample_Count": sample_count,
            "Reindex_X": reindex_nodes,
            "Out_Eids": edge_eids
        },
        attrs={
            "sample_sizes": sample_sizes,
            "return_eids": return_eids,
            "set_unique": set_unique,
            "set_reindex": set_reindex
        })
    return edge_src, edge_dst, sample_index, sample_count, reindex_nodes, edge_eids
