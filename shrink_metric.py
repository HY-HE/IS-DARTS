import torch
from xautodl.models.cell_operations import ResNetBasicblock


def add_iim_methods(model, search_space):
    for cell in model.module.cells:
        if not isinstance(cell, ResNetBasicblock):
            for node_str, edge in cell.edges.items():
                ops = edge if search_space == "nas-bench-201" else edge.ops
                for op in ops:
                    handle = op.register_forward_hook(iim_hook)
                    op.__iim_handle__ = handle
                    op.__iim__ = torch.tensor(0.0, device='cuda', dtype=torch.float)
                    op.__samples__ = torch.tensor(0.0, device='cuda', dtype=torch.float)


def get_iim(model):
    iim = []
    for cell in model.cells:
        if not isinstance(cell, ResNetBasicblock):
            cell_iim = []
            for node_str, node_edge in cell.edges.items():
                node_iim = []
                for op in node_edge:
                    node_iim.append(op.__iim__)
                cell_iim.append(node_iim)
            iim.append(cell_iim)
    torch.cuda.empty_cache()
    iim = torch.tensor(iim, device='cuda')
    return torch.mean(iim, dim=0)


def get_iim_nasnet(model):
    normal_iim = []
    reduce_iim = []
    for cell in model.cells:
        cell_iim = []
        for node_str, node_edge in cell.edges.items():
            node_iim = []
            for op in node_edge.ops:
                node_iim.append(op.__iim__)
            cell_iim.append(node_iim)
        if cell.reduction:
            reduce_iim.append(cell_iim)
        else:
            normal_iim.append(cell_iim)
    torch.cuda.empty_cache()
    normal_iim = torch.tensor(normal_iim, device='cuda')
    normal_iim = torch.mean(normal_iim, dim=0)
    reduce_iim = torch.tensor(reduce_iim, device='cuda')
    reduce_iim = torch.mean(reduce_iim, dim=0)
    return normal_iim, reduce_iim


def remove_iim_methods(model, search_space):
    for cell in model.module.cells:
        if not isinstance(cell, ResNetBasicblock):
            for node_str, edge in cell.edges.items():
                ops = edge if search_space == "nas-bench-201" else edge.ops
                for op in ops:
                    handle = getattr(op, "__iim_handle__")
                    handle.remove()
                    delattr(op, "__iim_handle__")
                    delattr(op, "__iim__")
                    delattr(op, "__samples__")


def iim_hook(module, inputs, output):
    B, _, _, _ = output.size()
    mean = torch.mean(output, dim=1, keepdim=True)
    std = torch.std(output, dim=1, keepdim=True)
    fisher = (3 * torch.pow(output - mean, 2)) / torch.pow(std, 6)
    inf = torch.full_like(fisher, float('inf'), device='cuda')
    fisher = torch.where(torch.isnan(fisher), inf, fisher)
    fisher = torch.mean(fisher, dim=[1, 2, 3])

    module.__iim__ = (module.__iim__ * module.__samples__ + torch.sum(fisher)) / (module.__samples__ + B)
    module.__samples__ += B
