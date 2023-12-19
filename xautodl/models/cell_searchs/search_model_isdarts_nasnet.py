####################
# DARTS, ICLR 2019 #
####################
import math
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Text, Dict
from .search_cells import NASNetSearchCell as SearchCell


# The macro structure is based on NASNet
class NASNetworkISDarts(nn.Module):
    def __init__(self, C: int, N: int, steps: int, multiplier: int, stem_multiplier: int, num_classes: int,
                 search_space: List[Text], affine: bool, track_running_stats: bool):
        super(NASNetworkISDarts, self).__init__()
        self._C = C
        self._layerN = N
        self._steps = steps
        self._multiplier = multiplier
        self.stem = nn.Sequential(
            nn.Conv2d(3, C * stem_multiplier, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C * stem_multiplier),
        )

        # config for each layer
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        num_edge, edge2index = None, None
        C_prev_prev, C_prev, C_curr, reduction_prev = C * stem_multiplier, C * stem_multiplier, C, False

        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            cell = SearchCell(search_space, steps, multiplier, C_prev_prev, C_prev, C_curr,
                              reduction, reduction_prev, affine, track_running_stats)
            if num_edge is None:
                num_edge, edge2index = cell.num_edges, cell.edge2index
            else:
                assert (
                    num_edge == cell.num_edges and edge2index == cell.edge2index
                ), "invalid {:} vs. {:}.".format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev_prev, C_prev, reduction_prev = C_prev, multiplier * C_curr, reduction
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        # self.arch_normal_parameters = nn.Parameter(1e-3 * torch.randn(num_edge, len(search_space)))
        # self.arch_reduce_parameters = nn.Parameter(1e-3 * torch.randn(num_edge, len(search_space)))
        self.normal_masks = torch.ones((num_edge, len(search_space)), device='cuda')
        self.reduce_masks = torch.ones((num_edge, len(search_space)), device='cuda')

    def get_weights(self) -> List[torch.nn.Parameter]:
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist

    # def get_alphas(self) -> List[torch.nn.Parameter]:
    #     return [self.arch_normal_parameters, self.arch_reduce_parameters]

    def show_alphas(self) -> Text:
        with torch.no_grad():
            A = "normal-masks :\n{:}".format(self.normal_masks.cpu())
            B = "reduce-masks :\n{:}".format(self.reduce_masks.cpu())
        return "{:}\n{:}".format(A, B)

    def get_message(self) -> Text:
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self) -> Text:
        return "{name}(C={_C}, N={_layerN}, steps={_steps}, multiplier={_multiplier}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def genotype(self) -> Dict[Text, List]:
        def _parse(masks):
            gene = []
            for i in range(self._steps):
                ops = []
                for j in range(2 + i):
                    node_str = "{:}<-{:}".format(i, j)
                    for k, op_name in enumerate(self.op_names):
                        ops.append((op_name, j, masks[self.edge2index[node_str]][k]))
                ops = sorted(ops, key=lambda x: -x[-1])
                selected_ops = ops[:2]
                gene.append(tuple(selected_ops))
            return gene

        with torch.no_grad():
            gene_normal = _parse(self.normal_masks.cpu().numpy())
            gene_reduce = _parse(self.reduce_masks.cpu().numpy())
        return {
            "normal": gene_normal,
            "normal_concat": list(range(2 + self._steps - self._multiplier, self._steps + 2)),
            "reduce": gene_reduce,
            "reduce_concat": list(range(2 + self._steps - self._multiplier, self._steps + 2))}

    # prune by number
    # def update_mask(self, normal_edges, reduce_edges, num_pruned):
    #     # inf = torch.full_like(self.normal_masks, float('inf'), device='cuda')
    #     zeros = torch.full_like(self.normal_masks, 0., device='cuda')
    #     masked_normal_edges = torch.where(torch.eq(self.normal_masks, 0.0), zeros, normal_edges)
    #     print(masked_normal_edges)
    #     masked_reduce_edges = torch.where(torch.eq(self.reduce_masks, 0.0), zeros, reduce_edges)
    #     print(masked_reduce_edges)
    #
    #     with torch.no_grad():
    #         gene_normal = []
    #         for i in range(self._steps):
    #             ops = []
    #             for j in range(2 + i):
    #                 node_str = "{:}<-{:}".format(i, j)
    #                 for k, op_name in enumerate(self.op_names):
    #                     if self.normal_masks[self.edge2index[node_str]][k] == 1.:
    #                         ops.append((op_name, i, j, k, normal_edges[self.edge2index[node_str]][k]))
    #             print(ops)
    #             if len(ops) > 2:
    #                 ops = sorted(ops, key=lambda x: -x[-1])   # 加了负号，去掉最大的num_pruned_i个
    #                 num_pruned_i = min(num_pruned, len(ops) - 2)
    #                 print(num_pruned_i)
    #                 selected_ops = ops[:num_pruned_i]   # select num_pruned_i/-num_pruned_i
    #                 print(selected_ops)
    #                 gene_normal.append(tuple(selected_ops))
    #                 for op in selected_ops:
    #                     node_str = "{:}<-{:}".format(op[1], op[2])
    #                     self.normal_masks[self.edge2index[node_str]][op[3]] = 0.
    #
    #         gene_reduce = []
    #         for i in range(self._steps):
    #             ops = []
    #             for j in range(2 + i):
    #                 node_str = "{:}<-{:}".format(i, j)
    #                 for k, op_name in enumerate(self.op_names):
    #                     if self.reduce_masks[self.edge2index[node_str]][k] == 1.:
    #                         ops.append((op_name, i, j, k, reduce_edges[self.edge2index[node_str]][k]))
    #             if len(ops) > 2:
    #                 ops = sorted(ops, key=lambda x: -x[-1])
    #                 num_pruned_i = min(num_pruned, len(ops) - 2)
    #                 selected_ops = ops[:num_pruned_i]   # select num_pruned_i/-num_pruned_i
    #                 gene_reduce.append(tuple(selected_ops))
    #                 for op in selected_ops:
    #                     node_str = "{:}<-{:}".format(op[1], op[2])
    #                     self.reduce_masks[self.edge2index[node_str]][op[3]] = 0.
    #
    #     return {
    #         "normal": gene_normal,
    #         "normal_concat": list(range(2 + self._steps - self._multiplier, self._steps + 2)),
    #         "reduce": gene_reduce,
    #         "reduce_concat": list(range(2 + self._steps - self._multiplier, self._steps + 2))}

    # prune by rate
    def update_mask(self, normal_edges, reduce_edges, step_total, step_count):
        # inf = torch.full_like(self.normal_masks, float('inf'), device='cuda')
        zeros = torch.full_like(self.normal_masks, 0., device='cuda')
        masked_normal_edges = torch.where(torch.eq(self.normal_masks, 0.0), zeros, normal_edges)
        print(masked_normal_edges)
        masked_reduce_edges = torch.where(torch.eq(self.reduce_masks, 0.0), zeros, reduce_edges)
        print(masked_reduce_edges)

        with torch.no_grad():
            gene_normal = []
            for i in range(self._steps):
                ops = []
                for j in range(2 + i):
                    node_str = "{:}<-{:}".format(i, j)
                    for k, op_name in enumerate(self.op_names):
                        if self.normal_masks[self.edge2index[node_str]][k] == 1.:
                            ops.append((op_name, i, j, k, normal_edges[self.edge2index[node_str]][k]))
                print(ops)
                # if len(ops) > 2:
                ops = sorted(ops, key=lambda x: -x[-1])      # 去掉最大的num_pruned个
                num_pruned = int(math.ceil((len(ops) - 2) / (step_total - step_count)))
                # 需要剪掉的数量均分到step_total步里
                # if num_pruned * (step_total - step_count) < len(ops) - 2:
                #     num_pruned += 1
                print(num_pruned)
                selected_ops = ops[:num_pruned]
                print(selected_ops)
                gene_normal.append(tuple(selected_ops))
                for op in selected_ops:
                    node_str = "{:}<-{:}".format(op[1], op[2])
                    self.normal_masks[self.edge2index[node_str]][op[3]] = 0.

            gene_reduce = []
            for i in range(self._steps):
                ops = []
                for j in range(2 + i):
                    node_str = "{:}<-{:}".format(i, j)
                    for k, op_name in enumerate(self.op_names):
                        if self.reduce_masks[self.edge2index[node_str]][k] == 1.:
                            ops.append((op_name, i, j, k, reduce_edges[self.edge2index[node_str]][k]))
                # if len(ops) > 2:
                ops = sorted(ops, key=lambda x: -x[-1])
                num_pruned = int(math.ceil((len(ops) - 2) / (step_total - step_count)))
                # if step_count < len(ops) - 2 - num_pruned * step_total:
                #     num_pruned += 1
                selected_ops = ops[:num_pruned]   # select num_pruned_i/-num_pruned_i
                gene_reduce.append(tuple(selected_ops))
                for op in selected_ops:
                    node_str = "{:}<-{:}".format(op[1], op[2])
                    self.reduce_masks[self.edge2index[node_str]][op[3]] = 0.

        return {
            "normal": gene_normal,
            "normal_concat": list(range(2 + self._steps - self._multiplier, self._steps + 2)),
            "reduce": gene_reduce,
            "reduce_concat": list(range(2 + self._steps - self._multiplier, self._steps + 2))}

    def forward(self, inputs):
        s0 = s1 = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                masks = self.reduce_masks
            else:
                masks = self.normal_masks
            s0, s1 = s1, cell.forward_isdarts(s0, s1, masks)
        out = self.lastact(s1)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits
