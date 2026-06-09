"""
CPGRec training & evaluation on GroceryFood dataset
Evaluates on 100-candidate set with 5 metrics:
- Recall@K, NDCG@K, Category Coverage@K, ILD@K, Novelty@K
"""

import os
import sys
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.model import CPGRec

# ============ Config ============
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Hyperparameters
EMBED_DIM = 64
N_LAYERS = 2
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-4
REG_WEIGHT = 1e-4
BATCH_SIZE = 2048
N_EPOCHS = 30
NEG_SAMPLES = 4
TOPK_LIST = [5, 10, 20]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============ Load Data ============
def load_data():
    cache_path = os.path.join(CACHE_DIR, 'cpgrec_data.pkl')
    print(f"Loading data from {cache_path} ...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    print(f"  n_users={data['n_users']}, n_items={data['n_items']}")
    print(f"  train_edges={data['train_src'].shape[0]}")
    print(f"  test_candidates={len(data['test_candidates'])}")
    return data


# ============ Build DGL Graphs ============
def build_graphs(data):
    n_users = data['n_users']
    n_items = data['n_items']
    ge = data['graph_edges']

    # User-item bipartite graph
    train_src = data['train_src']
    train_dst = data['train_dst']
    ui_graph = dgl.heterograph(
        {
            ('user', 'play', 'item'): (train_src, train_dst),
            ('item', 'played_by', 'user'): (train_dst, train_src),
        },
        num_nodes_dict={'user': n_users, 'item': n_items}
    )

    # AND-graph (item-item)
    def safe_tensor(t):
        if isinstance(t, torch.Tensor):
            return t
        return torch.tensor(t, dtype=torch.long)

    and_src_cb = safe_tensor(ge['co_cat_brand'][0])
    and_dst_cb = safe_tensor(ge['co_cat_brand'][1])
    and_src_cf = safe_tensor(ge['co_cat_feat'][0])
    and_dst_cf = safe_tensor(ge['co_cat_feat'][1])
    and_src_bf = safe_tensor(ge['co_brand_feat'][0])
    and_dst_bf = safe_tensor(ge['co_brand_feat'][1])

    and_graph_data = {}
    if len(and_src_cb) > 0:
        and_graph_data[('item', 'co_cat_brand', 'item')] = (and_src_cb, and_dst_cb)
    if len(and_src_cf) > 0:
        and_graph_data[('item', 'co_cat_feat', 'item')] = (and_src_cf, and_dst_cf)
    if len(and_src_bf) > 0:
        and_graph_data[('item', 'co_brand_feat', 'item')] = (and_src_bf, and_dst_bf)

    # Ensure all 3 edge types exist (add dummy if empty)
    dummy_src = torch.zeros(1, dtype=torch.long)
    dummy_dst = torch.zeros(1, dtype=torch.long)
    for etype in ['co_cat_brand', 'co_cat_feat', 'co_brand_feat']:
        key = ('item', etype, 'item')
        if key not in and_graph_data:
            and_graph_data[key] = (dummy_src, dummy_dst)

    and_graph = dgl.heterograph(and_graph_data, num_nodes_dict={'item': n_items})

    # OR-graph (item-item) — sample to cap at 10M edges for performance
    or_src = safe_tensor(ge['co_or'][0])
    or_dst = safe_tensor(ge['co_or'][1])
    if len(or_src) == 0:
        or_src = dummy_src
        or_dst = dummy_dst
    elif len(or_src) > 10_000_000:
        # Subsample for performance (200M edges is too large for CPU inference)
        idx = torch.randperm(len(or_src))[:10_000_000]
        or_src = or_src[idx]
        or_dst = or_dst[idx]
        print(f"  OR graph subsampled to {len(or_src)} edges")
    or_graph = dgl.graph((or_src, or_dst), num_nodes=n_items)

    print(f"  AND graph: {and_graph}")
    print(f"  OR graph edges: {or_graph.num_edges()}")

    return ui_graph, and_graph, or_graph


# ============ Negative Sampling ============
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, train_src, train_dst, n_items, n_neg=4):
        self.users = train_src
        self.pos_items = train_dst
        self.n_items = n_items
        self.n_neg = n_neg

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx].item()
        p = self.pos_items[idx].item()
        neg = torch.randint(0, self.n_items, (self.n_neg,)).tolist()
        return u, p, neg[0]  # single negative per sample (can extend)


def collate_fn(batch):
    users = torch.tensor([b[0] for b in batch], dtype=torch.long)
    pos = torch.tensor([b[1] for b in batch], dtype=torch.long)
    neg = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return users, pos, neg


# ============ Evaluation Metrics ============
def compute_metrics(scores_matrix, labels, item_cat, item_pop, topk_list):
    """
    scores_matrix: [n_users, n_candidates] float
    labels: [n_users] index of positive item in candidates (always 0 = first col is pos)
    item_cat: {item_new_id -> [cat_ids]}
    item_pop: {item_new_id -> count}
    topk_list: [5, 10, 20]
    Returns: dict of metric -> value
    """
    results = {}
    n_users = scores_matrix.shape[0]
    total_pop = sum(item_pop.values()) + 1

    # Pre-compute item category sets as numpy arrays for faster lookup
    all_cats_set = set()
    for c_list in item_cat.values():
        all_cats_set.update(c_list)
    n_all_cats = max(len(all_cats_set), 1)

    for k in topk_list:
        # Get top-k indices [n_users, k]
        topk_indices = np.argsort(-scores_matrix, axis=1)[:, :k]

        recalls = np.zeros(n_users)
        ndcgs = np.zeros(n_users)
        cat_covs = np.zeros(n_users)
        ilds = np.zeros(n_users)
        novelties = np.zeros(n_users)

        for i in range(n_users):
            topk = topk_indices[i]
            pos_rank = np.where(topk == labels[i])[0]

            # Recall@K
            if len(pos_rank) > 0:
                recalls[i] = 1.0
                ndcgs[i] = 1.0 / np.log2(pos_rank[0] + 2)

            # Category Coverage@K
            cats_in_topk = set()
            item_cats_list = []
            for item_idx in topk:
                c = item_cat.get(item_idx, [])
                item_cats_list.append(set(c))
                cats_in_topk.update(c)
            cat_covs[i] = len(cats_in_topk) / n_all_cats

            # ILD@K (vectorized over pairs)
            if k > 1:
                ild_sum = 0.0
                cnt = 0
                for a in range(k):
                    for b in range(a + 1, k):
                        cats_a = item_cats_list[a]
                        cats_b = item_cats_list[b]
                        union = cats_a | cats_b
                        if len(union) == 0:
                            sim = 1.0
                        else:
                            sim = len(cats_a & cats_b) / len(union)
                        ild_sum += (1.0 - sim)
                        cnt += 1
                ilds[i] = ild_sum / cnt

            # Novelty@K
            nov = 0.0
            for item_idx in topk:
                pop = item_pop.get(item_idx, 1) / total_pop
                nov -= np.log2(pop + 1e-10)
            novelties[i] = nov / k

        results[f'Recall@{k}'] = np.mean(recalls)
        results[f'NDCG@{k}'] = np.mean(ndcgs)
        results[f'CatCov@{k}'] = np.mean(cat_covs)
        results[f'ILD@{k}'] = np.mean(ilds)
        results[f'Novelty@{k}'] = np.mean(novelties)

    return results


# ============ Evaluate on 100-candidate set ============
@torch.no_grad()
def evaluate_candidates(model, data, ui_graph, and_graph, or_graph, split='test', device=DEVICE):
    model.eval()
    ui_graph = ui_graph.to(device)
    and_graph = and_graph.to(device)
    or_graph = or_graph.to(device)

    user_emb, item_emb = model.forward_graph_emb(ui_graph, and_graph, or_graph)
    user_emb = user_emb.cpu()
    item_emb = item_emb.cpu()

    candidates = data['test_candidates'] if split == 'test' else data.get('val_candidates', data['test_candidates'])
    item_cat_raw = data['item_cat']
    item_new_to_orig = data['item_new_to_orig']
    # item_cat_raw uses orig IDs; remap to new IDs for evaluation
    item_cat = {}
    for new_id, orig_id in item_new_to_orig.items():
        if orig_id in item_cat_raw:
            item_cat[new_id] = item_cat_raw[orig_id]
    item_pop = data['item_popularity']

    all_scores = []
    all_labels = []

    # Batch-compute scores
    print(f"  Computing scores for {len(candidates)} candidates...", flush=True)
    for idx, row in enumerate(candidates):
        if idx % 10000 == 0 and idx > 0:
            print(f"  Progress: {idx}/{len(candidates)}", flush=True)
        user_id = int(row[0])
        pos_item = int(row[1])
        neg_items = [int(x) for x in row[2:]]
        all_cands = [pos_item] + neg_items

        cand_tensor = torch.tensor(all_cands, dtype=torch.long)
        u_vec = user_emb[user_id].unsqueeze(0)           # [1, d]
        i_vecs = item_emb[cand_tensor]                    # [n_cands, d]
        scores = (u_vec * i_vecs).sum(dim=-1).numpy()     # [n_cands]

        all_scores.append(scores)
        all_labels.append(0)  # positive is always index 0

    scores_matrix = np.array(all_scores)
    labels = np.array(all_labels)

    results = compute_metrics(scores_matrix, labels, item_cat, item_pop, TOPK_LIST)
    return results


# ============ Validate (fast, on val_candidates) ============
@torch.no_grad()
def quick_val(model, data, ui_graph, and_graph, or_graph, device=DEVICE, max_users=3000):
    """Quick Recall@20 on a subset of val candidates"""
    model.eval()
    ui_graph = ui_graph.to(device)
    and_graph = and_graph.to(device)
    or_graph = or_graph.to(device)

    user_emb, item_emb = model.forward_graph_emb(ui_graph, and_graph, or_graph)
    user_emb = user_emb.cpu()
    item_emb = item_emb.cpu()

    # Build val candidates from cache
    val_cands_path = os.path.join(DATA_DIR, 'minimal', 'rec_val_candidate100.npz')
    val_cands_raw = np.load(val_cands_path, allow_pickle=True)['candidates']
    item_orig_to_new = data['item_orig_to_new']

    hits = 0
    total = 0
    for row in val_cands_raw[:max_users]:
        user_id = int(row[0])
        pos_item_orig = int(row[1])
        if pos_item_orig not in item_orig_to_new:
            continue
        pos_item_new = item_orig_to_new[pos_item_orig]
        neg_items_new = [item_orig_to_new[int(x)] for x in row[2:] if int(x) in item_orig_to_new]
        all_cands = [pos_item_new] + neg_items_new

        cand_tensor = torch.tensor(all_cands, dtype=torch.long)
        u_vec = user_emb[user_id].unsqueeze(0)
        i_vecs = item_emb[cand_tensor]
        scores = (u_vec * i_vecs).sum(dim=-1).numpy()
        topk = np.argsort(-scores)[:20]
        if 0 in topk:
            hits += 1
        total += 1

    return hits / max(total, 1)


# ============ Training ============
def train(data, ui_graph, and_graph, or_graph):
    n_users = data['n_users']
    n_items = data['n_items']

    model = CPGRec(
        n_users=n_users,
        n_items=n_items,
        embed_dim=EMBED_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_src = data['train_src']
    train_dst = data['train_dst']

    dataset = TrainDataset(train_src, train_dst, n_items, n_neg=1)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0
    )

    ui_graph_dev = ui_graph.to(DEVICE)
    and_graph_dev = and_graph.to(DEVICE)
    or_graph_dev = or_graph.to(DEVICE)

    best_recall = 0.0
    best_epoch = 0
    best_model_path = os.path.join(OUTPUT_DIR, 'cpgrec_best.pt')

    log_lines = []

    print("\n=== Training CPGRec ===")
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0

        # Compute graph embeddings once per epoch (full-batch graph propagation)
        # Detach to allow per-batch BPR loss backprop through base embeddings only
        with torch.no_grad():
            user_emb_graph, item_emb_graph = model.forward_graph_emb(
                ui_graph_dev, and_graph_dev, or_graph_dev)

        for batch in loader:
            users, pos_items, neg_items = [x.to(DEVICE) for x in batch]

            # Per-batch: use base embeddings for BPR so gradients flow to embedding tables
            u_emb = model.user_emb(users)
            p_emb = model.item_emb(pos_items)
            n_emb = model.item_emb(neg_items)

            pos_scores = (u_emb * p_emb).sum(dim=-1)
            neg_scores = (u_emb * n_emb).sum(dim=-1)
            bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

            reg = (u_emb.norm(2).pow(2) + p_emb.norm(2).pow(2) + n_emb.norm(2).pow(2)) / (3 * users.size(0))
            loss = bpr + REG_WEIGHT * reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Re-compute embeddings after optimizer step (for validation)
        with torch.no_grad():
            user_emb, item_emb = model.forward_graph_emb(
                ui_graph_dev, and_graph_dev, or_graph_dev)

        elapsed = time.time() - t0
        avg_loss = total_loss / len(loader)

        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            recall20 = quick_val(model, data, ui_graph, and_graph, or_graph, device=DEVICE)
            msg = f"Epoch {epoch:3d} | loss={avg_loss:.4f} | val_Recall@20={recall20:.4f} | time={elapsed:.1f}s"
            print(msg, flush=True)
            log_lines.append(msg)

            if recall20 > best_recall:
                best_recall = recall20
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Best model saved (val_Recall@20={best_recall:.4f})", flush=True)
        else:
            msg = f"Epoch {epoch:3d} | loss={avg_loss:.4f} | time={elapsed:.1f}s"
            print(msg, flush=True)
            log_lines.append(msg)

    print(f"\nBest val_Recall@20={best_recall:.4f} at epoch {best_epoch}")

    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    # Save training log
    with open(os.path.join(LOG_DIR, 'train_log.txt'), 'w') as f:
        f.write('\n'.join(log_lines))

    return model


# ============ Main ============
def main():
    print("=== CPGRec on GroceryFood Dataset ===", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    # Load cache
    data = load_data()

    # Build graphs
    print("\nBuilding DGL graphs...")
    ui_graph, and_graph, or_graph = build_graphs(data)

    # Train
    model = train(data, ui_graph, and_graph, or_graph)

    # Final test evaluation
    print("\n=== Final Evaluation on Test Set (100-candidate) ===")
    test_results = evaluate_candidates(model, data, ui_graph, and_graph, or_graph, split='test')

    print("\nMetrics:")
    for k in TOPK_LIST:
        print(f"  K={k}:")
        for metric in ['Recall', 'NDCG', 'CatCov', 'ILD', 'Novelty']:
            key = f"{metric}@{k}"
            print(f"    {key:<15} = {test_results[key]:.4f}")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write("=== CPGRec Test Results (100-candidate) ===\n\n")
        for k in TOPK_LIST:
            f.write(f"K={k}:\n")
            for metric in ['Recall', 'NDCG', 'CatCov', 'ILD', 'Novelty']:
                key = f"{metric}@{k}"
                f.write(f"  {key:<15} = {test_results[key]:.4f}\n")
            f.write("\n")

    print(f"\nResults saved to {results_path}")
    return test_results


if __name__ == '__main__':
    main()
