import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import torch_scatter
from hw_as.utils.graph import make_homo, make_hetero


class GAT(MessagePassing):

    def __init__(self, in_dim, out_dim, dropout=0.0, **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_map = nn.Linear(out_dim, 1, bias=False)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(dropout)

        # activate
        self.act = nn.SELU(inplace=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att_proj.weight)
        nn.init.zeros_(self.att_proj.bias)
        nn.init.xavier_uniform_(self.att_map.weight)
        nn.init.xavier_uniform_(self.proj_with_att.weight)
        nn.init.zeros_(self.proj_with_att.bias)
        nn.init.xavier_uniform_(self.proj_without_att.weight)
        nn.init.zeros_(self.proj_without_att.bias)


    def forward(self, x, edge_index):
        assert len(x.shape) == 2 # (B x N) x C
        x = self.input_drop(x)
        weighted = self.propagate(edge_index, x=x)
        out = self.proj_with_att(weighted) + self.proj_without_att(x)
        out = self.act(self.bn(out))
        return out


    def message(self, x_i, x_j, index, ptr, size_i):
        alpha = torch.tanh(self.att_proj(x_i * x_j))
        alpha = self.att_map(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        return alpha * x_j



    def aggregate(self, inputs, index, dim_size = None):
        node_dim = self.node_dim
        out = torch_scatter.scatter(inputs, index, dim=node_dim, reduce='sum')
        return out


class GraphPool(nn.Module):
    def __init__(self, n_features, k, dropout=0.0):
        super().__init__()
        self.k = k
        self.y_proj = nn.Linear(n_features, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.y_proj(self.drop(x))
        y = torch.sigmoid(y)
        k = max(int(x.shape[1] * self.k), 1)
        tops = torch.topk(y.squeeze(-1), k, dim=-1)
        ids = torch.tile(tops.indices.unsqueeze(-1), (1, 1, x.shape[-1]))
        x_top = torch.gather(x, 1, ids)
        y_top = tops.values

        return x_top * y_top.unsqueeze(-1), tops.indices


class HSGAL(MessagePassing):

    def __init__(self, in_dim, out_dim, dropout=0.0, **kwargs):
        super(HSGAL, self).__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_map11 = nn.Linear(out_dim, 1, bias=False)
        self.att_map22 = nn.Linear(out_dim, 1, bias=False)
        self.att_map12 = nn.Linear(out_dim, 1, bias=False)
        self.att_mapM = nn.Linear(out_dim, 1, bias=False)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(dropout)

        # activate
        self.act = nn.SELU(inplace=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_type1.weight)
        nn.init.zeros_(self.proj_type1.bias)
        nn.init.xavier_uniform_(self.proj_type2.weight)
        nn.init.zeros_(self.proj_type2.bias)
        nn.init.xavier_uniform_(self.att_proj.weight)
        nn.init.zeros_(self.att_proj.bias)
        nn.init.xavier_uniform_(self.att_projM.weight)
        nn.init.zeros_(self.att_projM.bias)
        nn.init.xavier_uniform_(self.att_map11.weight)
        nn.init.xavier_uniform_(self.att_map22.weight)
        nn.init.xavier_uniform_(self.att_map12.weight)
        nn.init.xavier_uniform_(self.att_mapM.weight)
        nn.init.xavier_uniform_(self.proj_with_att.weight)
        nn.init.zeros_(self.proj_with_att.bias)
        nn.init.xavier_uniform_(self.proj_without_att.weight)
        nn.init.zeros_(self.proj_without_att.bias)
        nn.init.xavier_uniform_(self.proj_with_attM.weight)
        nn.init.zeros_(self.proj_with_attM.bias)
        nn.init.xavier_uniform_(self.proj_without_attM.weight)
        nn.init.zeros_(self.proj_without_attM.bias)




    def forward(self, x, edge_index, part, num_nodes, calc_master):
        assert len(x.shape) == 2 # (B * N) x C
        x[part == 1] = self.proj_type1(x[part == 1])
        x[part == 2] = self.proj_type2(x[part == 2])
        if calc_master:
            x[part == 3] = x.view(-1, num_nodes, self.in_dim)[:,:-1,:].mean(dim=1)

        x = self.input_drop(x)

        weighted = self.propagate(edge_index, x=x, part=part)
        out = torch.zeros(x.shape[0], self.out_dim, device=x.device)
        out[part == 3] = self.proj_with_attM(weighted[part == 3]) + self.proj_without_attM(x[part == 3])
        out[part != 3] = self.proj_with_att(weighted[part != 3]) + self.proj_without_att(x[part != 3])

        out[part != 3] = self.act(self.bn(out[part != 3]))
        return out


    def message(self, x_i, x_j, part_i, part_j, index, ptr, size_i):

        product = x_i * x_j
        transformed = torch.zeros(x_i.shape[0], self.out_dim, device=x_i.device)

        transformed[part_i != 3] = self.att_proj(product[part_i != 3])
        transformed[part_i == 3] = self.att_projM(product[part_i == 3])

        att_map = torch.tanh(transformed)
        alpha = torch.zeros(x_i.shape[0], 1, device=x_i.device)
        alpha[(part_i == 1) & (part_j == 1)] = self.att_map11(att_map[(part_i == 1) & (part_j == 1)])
        alpha[(part_i == 1) & (part_j == 2)] = self.att_map12(att_map[(part_i == 1) & (part_j == 2)])
        alpha[(part_i == 2) & (part_j == 1)] = self.att_map12(att_map[(part_i == 2) & (part_j == 1)])
        alpha[(part_i == 2) & (part_j == 2)] = self.att_map22(att_map[(part_i == 2) & (part_j == 2)])
        alpha[part_i == 3] = self.att_mapM(att_map[part_i == 3])
        alpha = softmax(alpha, index, ptr, size_i)

        return alpha * x_j



    def aggregate(self, inputs, index, dim_size = None):
        node_dim = self.node_dim
        out = torch_scatter.scatter(inputs, index, dim=node_dim, reduce='sum')
        return out


class GraphModule(nn.Module):
    def __init__(self, in_dim, out_dim, k, gat_dropout, pool_dropout):
        super().__init__()
        self.gat = GAT(in_dim, out_dim, gat_dropout)
        self.pool = GraphPool(out_dim, k, pool_dropout)

    def forward(self, x, edges=None):
        if edges is None:
            edges = make_homo(x).to(x.device)
        out = self.gat(x.reshape(-1, x.shape[-1]), edges)
        out, _ = self.pool(out.view(x.shape[0], x.shape[1], -1))
        return out


class MGO(nn.Module):
    def __init__(self, in_dims, out_dims, ks, gat_dropout=0.0, pool_dropout=0.0, feat_dropout=0.0, final_dropout=0.0):
        super().__init__()
        self.first_gals = nn.ModuleList([HSGAL(in_dim, out_dim, gat_dropout) for in_dim, out_dim in zip(in_dims, out_dims)])
        self.first_pools = nn.ModuleList([nn.ModuleList([GraphPool(n_features, k[0], pool_dropout), GraphPool(n_features, k[1], pool_dropout)]) for n_features, k in zip(out_dims, ks)])
        self.second_gals = nn.ModuleList([HSGAL(in_dim, out_dim, gat_dropout) for in_dim, out_dim in zip(in_dims, out_dims)])
        self.second_pools = nn.ModuleList([nn.ModuleList([GraphPool(n_features, k[0], pool_dropout), GraphPool(n_features, k[1], pool_dropout)]) for n_features, k in zip(out_dims, ks)])
        self.first_master = nn.Parameter(torch.randn(1, 1, in_dims[0]))
        self.second_master = nn.Parameter(torch.randn(1, 1, in_dims[0]))
        self.drop = nn.Dropout(feat_dropout)
        self.final_dropout = nn.Dropout(final_dropout)
        self.head = nn.Linear(out_dims[-1] * 5, 2)

    def forward(self, lhs, rhs, all_edges=None):
        if all_edges is None:
            all_edges = [None] * len(self.first_gals)
        else:
            assert len(all_edges) == len(self.first_gals)


        results = {}
        for (branch, master_state, gals, pools) in [("first", self.first_master, self.first_gals, self.first_pools), ("second", self.second_master, self.second_gals, self.second_pools)]:
            master = master_state
            data, _, _ = make_hetero(lhs, rhs, master, return_edges=False, return_part=False)
            lhs_shape = tuple(lhs.shape)
            rhs_shape = tuple(rhs.shape)
            bs = lhs_shape[0]
            for gal, pool, edges in zip(gals, pools, all_edges):
                num_nodes = data.shape[0] // bs
                if edges is None:
                    _, edges, parts = make_hetero(lhs_shape, rhs_shape, return_data=False)
                else:
                    _, _, parts = make_hetero(lhs_shape, rhs_shape, return_data=False, return_edges=False)
                edges = edges.to(lhs.device)
                parts = parts.to(lhs.device)
                data = gal(data, edges, parts.flatten(), num_nodes, calc_master=False)
                data = data.view(bs, num_nodes, -1)
                data_lhs, _ = pool[0](data[parts == 1].view(bs, -1, data.shape[-1]))
                data_rhs, _ = pool[1](data[parts == 2].view(bs, -1, data.shape[-1]))
                master = data[:,-1:,:]
                data = torch.cat([data_lhs, data_rhs, master], dim=1).view(-1, data.shape[-1])
                lhs_shape = tuple(data_lhs.shape)
                rhs_shape = tuple(data_rhs.shape)
            results[branch] = {"lhs": data_lhs, "rhs": data_rhs, "master": master}
        lhs = torch.max(self.drop(results["first"]["lhs"]), self.drop(results["second"]["lhs"]))
        rhs = torch.max(self.drop(results["first"]["rhs"]), self.drop(results["second"]["rhs"]))
        master = torch.max(self.drop(results["first"]["master"]), self.drop(results["second"]["master"]))
        lhs_max = torch.max(lhs, dim=1).values
        rhs_max = torch.max(rhs, dim=1).values
        lhs_mean = torch.mean(lhs, dim=1)
        rhs_mean = torch.mean(rhs, dim=1)
        x = torch.cat([lhs_max, lhs_mean, rhs_max, rhs_mean, master.squeeze(1)], dim=1)
        x = self.final_dropout(x)
        return self.head(x)
