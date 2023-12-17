import torch

def make_homo(lhs):
    assert len(lhs.shape) == 3
    nodes = lhs.shape[1]
    srcs = torch.tile(torch.arange(nodes).unsqueeze(-1), (1, nodes)).flatten().unsqueeze(0)
    tgts = torch.tile(torch.arange(nodes), (nodes,)).unsqueeze(0)
    edges = torch.cat([srcs, tgts], dim=0).unsqueeze(0)
    starts = torch.arange(0, lhs.shape[0]) * (nodes)
    edges = edges + starts.view(-1, 1, 1)
    edges = edges.transpose(0, 1).reshape(2, -1)
    return edges

def make_hetero(lhs, rhs, master=None, return_data=True, return_edges=True, return_part=True):
    if return_data:
        assert isinstance(lhs, torch.Tensor)
        assert isinstance(rhs, torch.Tensor)
        assert len(lhs.shape) == 3
        assert len(rhs.shape) == 3
        assert lhs.shape[0] == rhs.shape[0]
        lhs_nodes = lhs.shape[1]
        rhs_nodes = rhs.shape[1]
        nodes = lhs_nodes + rhs_nodes
        bs = lhs.shape[0]
    else:
        assert isinstance(lhs, tuple)
        assert isinstance(rhs, tuple)
        assert len(lhs) == 3
        assert len(rhs) == 3
        assert lhs[0] == rhs[0]
        lhs_nodes = lhs[1]
        rhs_nodes = rhs[1]
        nodes = lhs_nodes + rhs_nodes
        bs = lhs[0]
    if return_edges:
        srcs = torch.tile(torch.arange(nodes).unsqueeze(-1), (1, nodes + 1)).flatten().unsqueeze(0)
        tgts = torch.tile(torch.arange(nodes + 1), (nodes,)).unsqueeze(0)
        edges = torch.cat([srcs, tgts], dim=0).unsqueeze(0)
        starts = torch.arange(0, bs) * (nodes + 1)
        edges = edges + starts.view(-1, 1, 1)
        edges = edges.transpose(0, 1).reshape(2, -1)
    else:
        edges = None
    if return_data:
        if master is None:
            master = torch.zeros(lhs.shape[0], 1, lhs.shape[-1], device=lhs.device)
        else:
            assert master.shape == (1, 1, lhs.shape[-1])
            master = torch.tile(master, (lhs.shape[0], 1, 1))
        data = torch.cat([lhs, rhs, master], dim=1)
        data = data.reshape(-1, lhs.shape[-1])
    else:
        data = None
    if return_part:
        part = torch.cat([torch.ones(lhs_nodes) * 1, torch.ones(rhs_nodes) * 2, torch.ones(1) * 3]).unsqueeze(0).long()
        part = torch.tile(part, (bs,1))
    else:
        part = None
    return data, edges, part
