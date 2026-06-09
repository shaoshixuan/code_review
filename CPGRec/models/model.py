"""
CPGRec Model - adapted for GroceryFood e-commerce dataset
Based on: CPGRec: Causal Preference-based Graph Recommendation
Original: https://github.com/CPGRec2024/CPGRec

Architecture:
- User-item bipartite graph with weighted GraphConv
- Item AND-graph (3 edge types: co_cat_brand, co_cat_feat, co_brand_feat)
- Item OR-graph (co_or: sharing any attribute)
- Adaptive weighted aggregation: h_item = w_and*h_and + w_or*h_or + w_self*h_self
- BPR loss with negative score reweighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.nn import GraphConv


class LightGCNConv(nn.Module):
    """LightGCN-style propagation (no activation, no bias)"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, graph, feat):
        with graph.local_scope():
            # Normalized adjacency
            graph.srcdata['h'] = feat
            graph.update_all(
                dgl.function.copy_u('h', 'm'),
                dgl.function.mean('m', 'neigh')
            )
            return self.fc(graph.dstdata['neigh'])


class ItemGraphConv(nn.Module):
    """GraphConv for item-item graphs (AND or OR)"""

    def __init__(self, in_dim, out_dim, etypes):
        super().__init__()
        self.convs = nn.ModuleDict({
            etype: GraphConv(in_dim, out_dim, norm='both', weight=True, bias=False, allow_zero_in_degree=True)
            for etype in etypes
        })
        self.fc_agg = nn.Linear(len(etypes) * out_dim, out_dim, bias=False)

    def forward(self, graph, feat):
        outs = []
        for etype, conv in self.convs.items():
            try:
                sub_g = graph.edge_type_subgraph([etype])
                h = conv(sub_g, feat)
            except Exception:
                # fallback: zeros if edge type causes issue
                out_dim = self.fc_agg.weight.shape[1]
                h = torch.zeros(feat.size(0), out_dim, device=feat.device)
            outs.append(h)
        return F.relu(self.fc_agg(torch.cat(outs, dim=-1)))


class CPGRec(nn.Module):
    """
    CPGRec model:
    - User/item embedding layers
    - User-item graph propagation (LightGCN-style, 1-2 layers)
    - Item AND-graph propagation (3 edge types)
    - Item OR-graph propagation (1 edge type)
    - Weighted aggregation with learned weights
    """

    def __init__(self, n_users, n_items, embed_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # Base embeddings
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        # LightGCN layers for user-item bipartite graph
        self.ui_convs_u2i = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(n_layers)
        ])
        self.ui_convs_i2u = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(n_layers)
        ])

        # AND-graph convolution (3 edge types)
        and_etypes = ['co_cat_brand', 'co_cat_feat', 'co_brand_feat']
        self.and_conv = ItemGraphConv(embed_dim, embed_dim, and_etypes)

        # OR-graph convolution
        self.or_conv = GraphConv(embed_dim, embed_dim, norm='both', weight=True, bias=False, allow_zero_in_degree=True)

        # Learnable aggregation weights
        self.w_and = nn.Parameter(torch.tensor(1.0 / 3))
        self.w_or = nn.Parameter(torch.tensor(1.0 / 3))
        self.w_self = nn.Parameter(torch.tensor(1.0 / 3))

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        for layer in self.ui_convs_u2i:
            nn.init.xavier_uniform_(layer.weight)
        for layer in self.ui_convs_i2u:
            nn.init.xavier_uniform_(layer.weight)

    def forward_graph_emb(self, ui_graph, and_graph, or_graph):
        """
        Compute user and item embeddings via graph propagation.
        Returns: user_final [n_users, d], item_final [n_items, d]
        """
        u_emb = self.user_emb.weight
        i_emb = self.item_emb.weight

        # LightGCN propagation on user-item bipartite graph
        u_layers = [u_emb]
        i_layers = [i_emb]

        u_cur = u_emb
        i_cur = i_emb

        with ui_graph.local_scope():
            for k in range(self.n_layers):
                # u -> i propagation
                ui_graph.srcnodes['user'].data['h'] = u_cur
                ui_graph.update_all(
                    dgl.function.copy_u('h', 'm'),
                    dgl.function.mean('m', 'agg'),
                    etype='play'
                )
                i_next = self.ui_convs_u2i[k](ui_graph.nodes['item'].data['agg'])

                # i -> u propagation
                ui_graph.srcnodes['item'].data['h'] = i_cur
                ui_graph.update_all(
                    dgl.function.copy_u('h', 'm'),
                    dgl.function.mean('m', 'agg'),
                    etype='played_by'
                )
                u_next = self.ui_convs_i2u[k](ui_graph.nodes['user'].data['agg'])

                u_cur = self.dropout(F.relu(u_next))
                i_cur = self.dropout(F.relu(i_next))
                u_layers.append(u_cur)
                i_layers.append(i_cur)

        # Mean pooling across layers (LightGCN aggregation)
        u_final = torch.stack(u_layers, dim=1).mean(dim=1)
        i_base = torch.stack(i_layers, dim=1).mean(dim=1)

        # AND-graph propagation
        h_and = self.and_conv(and_graph, i_base)
        h_and = self.dropout(h_and)

        # OR-graph convolution
        # Note: OR graph can be very large; use a simple mean aggregation
        try:
            if hasattr(or_graph, 'num_edge_types'):
                h_or = self.or_conv(or_graph, i_base)
            else:
                h_or = self.or_conv(or_graph, i_base)
        except Exception:
            h_or = i_base
        h_or = self.dropout(h_or)

        # Weighted aggregation
        w_sum = self.w_and.abs() + self.w_or.abs() + self.w_self.abs() + 1e-8
        w_and = self.w_and.abs() / w_sum
        w_or = self.w_or.abs() / w_sum
        w_self = self.w_self.abs() / w_sum

        i_final = w_and * h_and + w_or * h_or + w_self * i_base

        return u_final, i_final

    def predict(self, user_emb, item_emb, user_ids, item_ids):
        """Dot-product score for given user-item pairs"""
        u = user_emb[user_ids]
        i = item_emb[item_ids]
        return (u * i).sum(dim=-1)

    def predict_batch(self, user_emb, item_emb, user_ids, item_ids_list):
        """
        Predict scores for each user against a list of items.
        user_ids: [B]
        item_ids_list: [B, K]
        returns: [B, K]
        """
        u = user_emb[user_ids].unsqueeze(1)          # [B, 1, d]
        i = item_emb[item_ids_list]                    # [B, K, d]
        return (u * i).sum(dim=-1)                     # [B, K]

    def bpr_loss(self, user_emb, item_emb, user_ids, pos_item_ids, neg_item_ids, neg_weights=None):
        """
        BPR loss with optional negative reweighting.
        neg_weights: [B] — higher weight for hard negatives
        """
        u = user_emb[user_ids]
        pos = item_emb[pos_item_ids]
        neg = item_emb[neg_item_ids]

        pos_scores = (u * pos).sum(dim=-1)
        neg_scores = (u * neg).sum(dim=-1)

        loss = -F.logsigmoid(pos_scores - neg_scores)

        if neg_weights is not None:
            loss = loss * neg_weights

        return loss.mean()

    def reg_loss(self, user_ids, pos_item_ids, neg_item_ids):
        """L2 regularization on base embeddings"""
        u = self.user_emb(user_ids)
        p = self.item_emb(pos_item_ids)
        n = self.item_emb(neg_item_ids)
        return (u.norm(2).pow(2) + p.norm(2).pow(2) + n.norm(2).pow(2)) / (3 * user_ids.size(0))
