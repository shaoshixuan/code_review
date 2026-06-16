"""
IERec 100-candidate 评估脚本
在训练好的 IERec 模型上，用与其他方法相同的 100-candidate test set 进行评估

指标定义完全对齐 CPGRec/CMB/DRDW 的 compute_metrics：
  - Recall@K    : 正样本是否在 top-K 中
  - NDCG@K      : 1/log2(rank+2) if hit else 0（单正样本）
  - CatCov@K    : 全局 covered categories / total unique categories
  - ILD@K       : primary-category 两两不同的比例（pairwise）
  - Novelty@K   : mean(-ln(pop / total_pop))，total_pop = sum(counts)+1

用法:
  python3 eval_ierec_100cand.py --dataset grocery --model_path saved/NEW-xxx.pth
  python3 eval_ierec_100cand.py --dataset automotive --model_path saved/NEW-yyy.pth
  python3 eval_ierec_100cand.py --dataset toys --model_path saved/NEW-zzz.pth
"""
import argparse
import math
import sys
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[1]
IEREC_DIR = Path(__file__).resolve().parent

DATASET_DIRS = {
    'grocery':    REPO / 'data'            / 'minimal',
    'automotive': REPO / 'data_automotive' / 'minimal',
    'toys':       REPO / 'data_toys'       / 'minimal',
}
KG_DIRS = {
    'grocery':    REPO / 'data'            / 'KG-related_Files',
    'automotive': REPO / 'data_automotive' / 'KG-related_Files',
    'toys':       REPO / 'data_toys'       / 'KG-related_Files',
}

TOPK_LIST = [5, 10, 15, 20]


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def _pick_first_existing(base_dir, patterns):
    base_path = Path(base_dir)
    for pattern in patterns:
        matches = sorted(base_path.glob(pattern))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f"No file matched {patterns} under {base_dir}")


def load_item_categories(kg_dir):
    """
    从 KG 三元组文件加载 item -> [cat_name, ...] (list，与 CPGRec 一致)
    relation_id=8 是 category 关系（与 CPGRec data_converter.py 一致）
    返回: {orig_item_id -> [cat_name, ...]}
    """
    RELATION_ID_CAT = 8

    # 加载实体名
    entity_file = _pick_first_existing(
        kg_dir, ['kg_entities_*.txt', 'kg_other_entities_*.txt'])
    entity_id_to_name = {}
    with open(entity_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                entity_id_to_name[int(parts[1])] = parts[0]

    category_entities = {
        eid for eid, name in entity_id_to_name.items()
        if name.startswith('category::')
    }

    # 加载三元组
    # 格式: head\ttail\trelation（与 CPGRec data_converter.py 一致）
    triple_file = _pick_first_existing(kg_dir, ['kg_other_triples_*.txt'])
    item_cat = defaultdict(list)
    with open(triple_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                head     = int(parts[0])
                tail     = int(parts[1])   # tail 在中间位置
                relation = int(parts[2])   # relation 在最后
                if relation == RELATION_ID_CAT and tail in category_entities:
                    cat_name = entity_id_to_name[tail].replace('category::', '')
                    item_cat[head].append(cat_name)

    return dict(item_cat)  # {orig_item_id -> [cat_name, ...]}


def load_train_data(train_path):
    """返回 item_pop (orig_id->count), total_pop, user_seqs (uid->list of orig_iid)"""
    item_pop = defaultdict(int)
    user_seqs = defaultdict(list)
    with open(train_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                uid = int(parts[0])
                iid = int(parts[1])
                item_pop[iid] += 1
                user_seqs[uid].append(iid)
    total_pop = sum(item_pop.values()) + 1  # 与 CPGRec 完全一致
    return dict(item_pop), total_pop, dict(user_seqs)


# ─────────────────────────────────────────────────────────────────────────────
# 指标计算（完全对齐 CPGRec compute_metrics）
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(scores_matrix, item_cat, item_pop, total_pop, topk_list):
    """
    scores_matrix : np.ndarray [n_users, n_candidates]
                    第 0 列是正样本得分，其余是负样本
    item_cat      : {orig_item_id -> [cat_name, ...]}  (list, 允许多 category)
    item_pop      : {orig_item_id -> count}
    total_pop     : sum(item_pop.values()) + 1
    topk_list     : e.g. [5, 10, 15, 20]

    labels 固定为 0（第 0 列始终是正样本，与 CPGRec 一致）
    """
    results = {}
    n_users = scores_matrix.shape[0]

    # 全局 category 集合（用 all cats 覆盖的 unique 数量做分母，与 CPGRec 一致）
    all_cats_set = set()
    for c_list in item_cat.values():
        all_cats_set.update(c_list)
    n_all_cats = max(len(all_cats_set), 1)

    # item_cat 里的 key 是 orig_id；scores_matrix 的列对应 candidates 的 orig_id
    # 注意：此函数接受 cand_orig_ids（每行 candidates 的 item orig IDs）作为额外输入
    # 但为保持与 CPGRec 接口一致，我们在外层把 cand_orig_ids 传进来
    raise NotImplementedError("Use compute_metrics_with_ids instead")


def compute_metrics_with_ids(user_score_rows, item_cat, item_pop, total_pop, topk_list):
    """
    user_score_rows : list of (scores_arr, cand_orig_ids)
        scores_arr    : np.ndarray [n_candidates]，第 0 位是正样本
        cand_orig_ids : list[int] 长度 n_candidates，第 0 位是正样本 orig_id
    item_cat  : {orig_item_id -> [cat_name, ...]}
    item_pop  : {orig_item_id -> count}
    total_pop : sum(item_pop.values()) + 1
    topk_list : e.g. [5, 10, 15, 20]
    """
    results = {}
    n_users = len(user_score_rows)

    # 全局 category 集合
    all_cats_set = set()
    for c_list in item_cat.values():
        all_cats_set.update(c_list)
    n_all_cats = max(len(all_cats_set), 1)

    for k in topk_list:
        recalls     = np.zeros(n_users)
        ndcgs       = np.zeros(n_users)
        ilds        = np.zeros(n_users)
        novelties   = np.zeros(n_users)
        covered_global = set()

        for i, (scores, cand_orig_ids) in enumerate(user_score_rows):
            # top-k indices（按得分降序）
            topk_idx = np.argsort(-scores)[:k]

            # Recall@K
            if 0 in topk_idx:
                recalls[i] = 1.0
                pos_rank = int(np.where(topk_idx == 0)[0][0])
                ndcgs[i]  = 1.0 / math.log2(pos_rank + 2)

            # 取 top-k item 的 orig_id 列表
            topk_items = [cand_orig_ids[idx] for idx in topk_idx]

            # Category 信息（允许多 category，用 list）
            cats_in_topk = set()
            item_cats_list = []
            primary_cats = []
            for orig_id in topk_items:
                c = item_cat.get(orig_id, [])
                item_cats_list.append(set(c))
                cats_in_topk.update(c)
                primary_cats.append(c[0] if len(c) > 0 else None)

            # CatCov@K（全局累积）
            covered_global.update(cats_in_topk)

            # ILD@K（primary category pairwise difference）
            if k > 1:
                ild_sum = 0.0
                cnt = 0
                for a in range(k):
                    for b in range(a + 1, k):
                        ild_sum += 1.0 if primary_cats[a] != primary_cats[b] else 0.0
                        cnt += 1
                ilds[i] = ild_sum / cnt

            # Novelty@K（-ln(pop / total_pop) 均值）
            nov = 0.0
            for orig_id in topk_items:
                pop = item_pop.get(orig_id, 1)
                p   = pop / total_pop
                nov += (-math.log(p) if p > 0 else 0.0)
            novelties[i] = nov / k

        results[f'Recall@{k}']  = float(np.mean(recalls))
        results[f'NDCG@{k}']    = float(np.mean(ndcgs))
        results[f'CatCov@{k}']  = len(covered_global) / n_all_cats
        results[f'ILD@{k}']     = float(np.mean(ilds))
        results[f'Novelty@{k}'] = float(np.mean(novelties))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# IERec 模型推断
# ─────────────────────────────────────────────────────────────────────────────

def get_user_representations(model, dataset, user_seqs, candidates_raw, device):
    """
    用训练好的 IERec 序列模型生成每个测试用户的表示向量。
    返回：list of (scores_arr, cand_orig_ids)
    """
    max_seq_len = model.max_seq_length if hasattr(model, 'max_seq_length') else 50

    # RecBole internal item ID -> orig item ID 映射
    token_to_iid = dataset.field2token_id.get('item_id', {})
    iid_to_orig = {}
    for token_str, iid in token_to_iid.items():
        try:
            iid_to_orig[int(iid)] = int(token_str)
        except (ValueError, TypeError):
            pass
    orig_to_iid = {v: k for k, v in iid_to_orig.items()}

    # item embedding matrix（RecBole: index 0 = padding）
    item_emb_weight = model.item_embedding.weight.detach().cpu().numpy()

    model.eval()
    result_rows = []
    skipped = 0

    for row_idx, row in enumerate(candidates_raw):
        uid      = int(row[0])
        pos_orig = int(row[1])
        neg_orig = [int(x) for x in row[2:]]
        cand_orig_ids = [pos_orig] + neg_orig

        # 用户历史序列（orig item IDs）
        seq_orig = user_seqs.get(uid, [])
        seq_orig = seq_orig[-max_seq_len:]

        # 转为 RecBole internal item IDs
        seq_iids = [orig_to_iid[i] for i in seq_orig if i in orig_to_iid]
        if len(seq_iids) == 0:
            skipped += 1
            continue

        # 构建 batch tensor
        seq_tensor = torch.zeros(max_seq_len, dtype=torch.long)
        seq_len = min(len(seq_iids), max_seq_len)
        seq_tensor[-seq_len:] = torch.tensor(seq_iids[-seq_len:], dtype=torch.long)
        item_seq     = seq_tensor.unsqueeze(0).to(device)           # [1, max_len]
        item_seq_len = torch.tensor([seq_len], dtype=torch.long).to(device)

        with torch.no_grad():
            # IERec (NEW) forward requires global_seq argument
            # global_seq1 = model.global_seq1(zeros_like(item_seq))
            global_seq = model.global_seq1(torch.zeros_like(item_seq))  # [1, max_len, hidden]
            seq_out = model.forward(item_seq, item_seq_len, global_seq)  # [1, hidden]
            if seq_out.dim() == 3:
                seq_out = seq_out[:, -1, :]  # 取最后时间步（保险）
        u_vec = seq_out.squeeze(0).cpu().numpy()                    # [hidden]

        # 候选 item 的 internal ID -> embedding
        cand_iids = [orig_to_iid.get(orig, 0) for orig in cand_orig_ids]
        cand_emb  = item_emb_weight[cand_iids]                      # [n_cands, hidden]
        scores    = cand_emb @ u_vec                                 # [n_cands]

        result_rows.append((scores, cand_orig_ids))

        if (row_idx + 1) % 20000 == 0:
            print(f"  Progress: {row_idx+1}/{len(candidates_raw)}", flush=True)

    if skipped > 0:
        print(f"  Skipped {skipped} rows (no history in RecBole vocab)")

    return result_rows


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IERec 100-candidate evaluation (aligned with CPGRec/CMB/DRDW)")
    parser.add_argument('--dataset',    type=str, required=True,
                        choices=['grocery', 'automotive', 'toys'])
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved RecBole .pth file')
    parser.add_argument('--output',     type=str, default=None,
                        help='Output txt file path (default: output/{dataset}/ierec_100cand_results.txt)')
    args = parser.parse_args()

    minimal_dir = DATASET_DIRS[args.dataset]
    kg_dir      = KG_DIRS[args.dataset]
    device      = torch.device('cpu')

    print(f"\n=== IERec Evaluation: {args.dataset} ===", flush=True)
    print(f"Model: {args.model_path}", flush=True)

    # 1. 加载 RecBole 模型
    sys.path.insert(0, str(IEREC_DIR))
    from recbole.quick_start import load_data_and_model
    print("Loading model...", flush=True)
    config, model, dataset, train_data, valid_data, test_data = \
        load_data_and_model(args.model_path)
    model = model.to(device)
    model.eval()
    print(f"  Users={dataset.user_num}, Items={dataset.item_num}", flush=True)

    # 2. 加载 100-candidate test set
    test_cand_path = minimal_dir / 'rec_test_candidate100.npz'
    candidates_raw = np.load(test_cand_path, allow_pickle=True)['candidates']
    print(f"  Test candidates: {candidates_raw.shape}", flush=True)

    # 3. 加载训练数据（item_pop, total_pop, user_seqs）
    train_path = minimal_dir / 'rec_train.txt'
    item_pop, total_pop, user_seqs = load_train_data(str(train_path))
    print(f"  Items with pop data: {len(item_pop)}, total_pop={total_pop}", flush=True)

    # 4. 加载 item category（原始 orig_id -> [cat_name, ...]）
    print("Loading item categories...", flush=True)
    item_cat = load_item_categories(kg_dir)
    all_cats = set()
    for cl in item_cat.values():
        all_cats.update(cl)
    print(f"  Items with category: {len(item_cat)}, total unique cats: {len(all_cats)}",
          flush=True)

    # 5. 生成用户表示 & 打分
    print(f"Computing user representations & scoring...", flush=True)
    user_score_rows = get_user_representations(
        model, dataset, user_seqs, candidates_raw, device)
    print(f"  Valid rows: {len(user_score_rows)}", flush=True)

    # 6. 计算指标
    print("Computing metrics...", flush=True)
    results = compute_metrics_with_ids(
        user_score_rows, item_cat, item_pop, total_pop, TOPK_LIST)

    # 7. 打印
    print(f"\n{'='*55}")
    print(f"  IERec 100-Candidate Evaluation ({args.dataset})")
    print(f"{'='*55}")
    header = f"{'Metric':<12}" + "".join(f"  @{k:<7}" for k in TOPK_LIST)
    print(header)
    print("-" * len(header))
    for metric in ['Recall', 'NDCG', 'CatCov', 'ILD', 'Novelty']:
        row = f"{metric:<12}"
        for k in TOPK_LIST:
            row += f"  {results[f'{metric}@{k}']:.4f} "
        print(row)
    print(f"{'='*55}\n")

    # 8. 保存
    out_path = args.output or str(
        IEREC_DIR / 'output' / args.dataset / 'ierec_100cand_results.txt')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(f"=== IERec 100-Candidate Evaluation ({args.dataset}) ===\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test rows: {len(user_score_rows)}\n\n")
        for k in TOPK_LIST:
            f.write(f"K={k}:\n")
            for metric in ['Recall', 'NDCG', 'CatCov', 'ILD', 'Novelty']:
                f.write(f"  {metric}@{k:<3} = {results[f'{metric}@{k}']:.4f}\n")
            f.write("\n")

    print(f"Results saved to: {out_path}")


if __name__ == '__main__':
    main()
