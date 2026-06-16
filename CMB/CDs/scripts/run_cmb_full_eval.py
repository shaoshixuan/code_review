"""
CMB 完整流程：
1. 加载最佳基础模型
2. 运行 Epsilon-Greedy 多臂赌博机优化 action_delta
3. 用 action_delta 对 100 候选集进行重排序
4. 计算 Recall@K, NDCG@K, CatCov@K, ILD@K, Novelty@K

候选集格式：rec_test_candidate100.npz
  shape (145932, 102): [user_id, pos_item_orig, neg1_orig, ..., neg100_orig]
  ID 为原始 item id (57822-98515)，需转换为 new id (0-based)
"""
import torch
import numpy as np
import pickle
import os
import sys
import tqdm
import math
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from time import time
import datetime

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models import BaseRecModel, DivOptimizationModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR.parent / "data_automotive"),
                        help="dataset root containing minimal/ and KG-related_Files/")
    parser.add_argument("--dataset", type=str, default="automotive")
    parser.add_argument("--data_obj_path", type=str, default="./dataset_objs_auto/")
    parser.add_argument("--base_model_path", type=str, default="./logs/")
    parser.add_argument("--base_model_file", type=str, default="best.base.model.pth")
    parser.add_argument("--candidates_path", type=str, default=None)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--cuda", type=str, default='0')
    # CMB bandit 参数
    parser.add_argument("--bandit_epochs", type=int, default=50,
                        help="多臂赌博机优化轮次")
    parser.add_argument("--num_arms", type=int, default=61)
    parser.add_argument("--exp_rate", type=float, default=0.1)
    parser.add_argument("--mask_type", type=int, default=2,
                        help="1=user, 2=item, 3=both")
    # 评估参数
    parser.add_argument("--rec_k", type=int, default=20)
    parser.add_argument("--output", type=str, default="./output/cmb_full_eval_results.txt")
    parser.add_argument("--load_delta", type=str, default=None,
                        help="若已有 action_delta，跳过多臂赌博机直接加载评估")
    return parser.parse_args()


# ─── 多臂赌博机优化 ───────────────────────────────────────────────────────────

def run_bandit(base_model, rec_dataset, device, args):
    """
    用 Epsilon-Greedy 多臂赌博机在全候选集上优化 action_delta（item embedding perturbation）
    奖励信号：全体测试用户的平均 ILAD@20
    """

    div_args = type('DivArgs', (), {
        'mask_type': args.mask_type,
        'feature_dims': rec_dataset.feature_dims,
    })()

    opt_model = DivOptimizationModel(base_model, rec_dataset, device, div_args)

    item_num = rec_dataset.item_num
    feat_dim = rec_dataset.feature_dims
    n_agents = item_num * feat_dim
    num_arm = args.num_arms
    exp_rate = args.exp_rate

    arm = np.zeros((n_agents, num_arm))
    for agent in range(n_agents):
        arm[agent] = np.random.normal(0, 0.6, num_arm)

    estimated_rewards = np.zeros((n_agents, num_arm), dtype=float)
    action_delta = np.zeros(n_agents, dtype='float32')
    action_list = np.zeros(n_agents, dtype=int)
    num_pulls = np.zeros((n_agents, num_arm), dtype=int)

    best_ilad = -1.0
    best_action_delta = None

    print(f"\n=== CMB 多臂赌博机优化 ({args.bandit_epochs} epochs) ===")
    for epoch in tqdm.trange(args.bandit_epochs):
        t0 = time()

        # Epsilon-Greedy 选 action —— 向量化版本，避免 Python 循环
        rand_mask = np.random.random(n_agents) < exp_rate
        rand_actions = np.random.randint(0, num_arm, size=n_agents)
        greedy_actions = np.argmax(estimated_rewards, axis=1)
        action_list[:] = np.where(rand_mask, rand_actions, greedy_actions)
        action_delta[:] = arm[np.arange(n_agents), action_list].astype('float32')

        item_delta_matrix = action_delta.reshape(item_num, feat_dim)

        # 计算奖励：全体测试用户的平均 ILAD@20
        ilad_mean = compute_ilad_reward(base_model, rec_dataset, item_delta_matrix, device, k=20)
        rewards = ilad_mean

        # 更新 estimated_rewards —— 向量化
        a_idx = action_list
        num_pulls[np.arange(n_agents), a_idx] += 1
        n_i = num_pulls[np.arange(n_agents), a_idx]
        estimated_rewards[np.arange(n_agents), a_idx] = (
            (n_i - 1) * estimated_rewards[np.arange(n_agents), a_idx] + rewards
        ) / n_i

        if rewards > best_ilad:
            best_ilad = rewards
            best_action_delta = action_delta.copy()

        t1 = time()
        print(f"Epoch {epoch+1}: ILAD={rewards:.4f}, best={best_ilad:.4f}, time={t1-t0:.1f}s")
        logger.info(f"Epoch {epoch+1}: ILAD={rewards:.4f}")

    print(f"\n最佳 ILAD={best_ilad:.4f}")
    return best_action_delta.reshape(item_num, feat_dim)


def compute_ilad_reward(base_model, rec_dataset, item_delta_matrix, device, k=20):
    """快速计算全体用户平均 ILAD@k，用于多臂赌博机奖励（向量化 numpy 版）"""
    base_model.eval()
    with torch.no_grad():
        user_emb = base_model.user_embedding_matrix.weight.detach().cpu().numpy()
        item_emb = base_model.item_embedding_matrix.weight.detach().cpu().numpy()
    item_emb_shifted = item_emb + item_delta_matrix  # (item_num, dim)

    users = [row[0] for row in rec_dataset.test_data]
    batch_size = 1000
    ilads = []

    for i in range(0, len(users), batch_size):
        batch_u = users[i:i+batch_size]
        u_e = user_emb[batch_u]                               # (B, dim)
        scores = u_e @ item_emb_shifted.T                     # (B, item_num)
        top_idx = np.argpartition(scores, -k, axis=1)[:, -k:]  # (B, k) 近似 top-k
        for j in range(len(batch_u)):
            reps = item_emb_shifted[top_idx[j]]               # (k, dim)
            norms = np.linalg.norm(reps, axis=1, keepdims=True)
            norms[norms == 0] = 1e-8
            reps_n = reps / norms
            sim = reps_n @ reps_n.T                           # (k, k)
            dis = 1 - sim
            ilads.append(float(np.sum(dis) / 2 / (k * (k - 1))))

    return float(np.mean(ilads))


# ─── 评估（100 候选集） ────────────────────────────────────────────────────────

def evaluate_on_candidates(base_model, item_delta_matrix, rec_dataset,
                            candidates_raw, item_name_dict,
                            item_topic_dict, item_popularity, topic_num,
                            device, k_list=(5, 10, 15, 20)):
    """
    在 100 候选集上评估 CMB 完整方法。

    candidates_raw: (N, 102) array, [user_orig, pos_orig, neg1_orig, ..., neg100_orig]
    item_name_dict: orig_id -> new_id
    item_delta_matrix: (item_num, feat_dim) numpy array — CMB 的 item perturbation
    """
    base_model.eval()
    max_k = max(k_list)

    with torch.no_grad():
        user_emb = base_model.user_embedding_matrix.weight.detach().cpu().numpy()
        item_emb = base_model.item_embedding_matrix.weight.detach().cpu().numpy()
        if item_delta_matrix is not None:
            item_emb_shifted = item_emb + item_delta_matrix  # (item_num, feat_dim)
        else:
            item_emb_shifted = item_emb

    item_topics_set = {iid: set(item_topic_dict.get(iid, []))
                       for iid in rec_dataset.items}
    total_users = len(candidates_raw)

    results = defaultdict(list)

    for row in candidates_raw:
        user_orig = int(row[0])
        pos_orig = int(row[1])
        negs_orig = [int(x) for x in row[2:]]

        # 转换为 new id（0-based）
        pos_new = item_name_dict.get(pos_orig)
        if pos_new is None:
            continue  # 该 item 不在训练集映射里，跳过

        cand_new = [item_name_dict[i] for i in negs_orig if i in item_name_dict]
        cand_new_with_pos = [pos_new] + cand_new  # 共 1 + 最多100 = 101 items

        # 计算该用户对所有候选 item 的得分
        u_emb = user_emb[user_orig]              # (feat_dim,)
        cand_arr = np.array(cand_new_with_pos)   # (101,)
        item_reps = item_emb_shifted[cand_arr]   # (101, feat_dim)
        scores = item_reps @ u_emb               # (101,)

        sort_idx = np.argsort(scores)[::-1]
        ranked_new = [cand_new_with_pos[i] for i in sort_idx[:max_k]]

        pos_set = {pos_new}

        for k in k_list:
            rec_k = ranked_new[:k]
            rec_set = set(rec_k)

            # Recall@K
            recall = len(pos_set & rec_set) / len(pos_set)
            results[f'Recall@{k}'].append(recall)

            # NDCG@K
            dcg = sum(1.0 / math.log2(r + 2) for r, it in enumerate(rec_k) if it in pos_set)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(pos_set), k)))
            results[f'NDCG@{k}'].append(dcg / idcg if idcg > 0 else 0.0)

            # CatCov@K (global collector)
            covered = set()
            for it in rec_k:
                covered.update(item_topics_set.get(it, set()))
            results[f'CatCovItems@{k}'].append(covered)

            # ILD@K (primary-category difference, aligned)
            primary_topics = [sorted(item_topics_set.get(it, set()))[0] if item_topics_set.get(it, set()) else None for it in rec_k]
            n_pairs = k * (k - 1) / 2
            diff = sum(
                1 for i in range(k) for j in range(i + 1, k)
                if primary_topics[i] != primary_topics[j]
            )
            results[f'ILD@{k}'].append(diff / n_pairs if n_pairs > 0 else 0.0)

            # Novelty@K (-ln(pop / total_interactions), aligned)
            total_interactions = sum(item_popularity.values())
            novs = []
            for it in rec_k:
                pop = item_popularity.get(it, 1)
                p = pop / total_interactions if total_interactions > 0 else 0.0
                novs.append(-math.log(p) if p > 0 else 0.0)
            results[f'Novelty@{k}'].append(float(np.mean(novs)))

    final = {}
    for key, values in results.items():
        if key.startswith('CatCovItems@'):
            continue
        final[key] = float(np.mean(values))
    for k in k_list:
        covered_global = set()
        for covered in results[f'CatCovItems@{k}']:
            covered_global.update(covered)
        final[f'CatCov@{k}'] = len(covered_global) / topic_num if topic_num > 0 else 0.0
    return final


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.cuda}')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # 加载数据集
    ds_path = os.path.join(args.data_obj_path, args.dataset + "_dataset_obj.pickle")
    print(f"加载数据集: {ds_path}")
    with open(ds_path, 'rb') as f:
        rec_dataset = pickle.load(f)

    # 加载基础模型
    model_path = os.path.join(
        args.base_model_path,
        args.dataset + "_logs/base",
        args.base_model_file)
    print(f"加载基础模型: {model_path}")
    base_model = BaseRecModel(
        rec_dataset.topic_num, rec_dataset.feature_dims,
        rec_dataset.user_num, rec_dataset.item_num
    ).to(device)
    base_model.load_state_dict(torch.load(model_path, map_location=device))
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # 运行 CMB 多臂赌博机 或 加载已有 delta
    out_dir = os.path.dirname(args.output)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    delta_save_path = os.path.join(out_dir, "best_action_delta.npy")

    if args.load_delta and os.path.exists(args.load_delta):
        print(f"加载已有 action_delta: {args.load_delta}")
        item_delta_matrix = np.load(args.load_delta).reshape(
            rec_dataset.item_num, rec_dataset.feature_dims)
    else:
        torch.manual_seed(0)
        np.random.seed(0)
        item_delta_matrix = run_bandit(base_model, rec_dataset, device, args)
        np.save(delta_save_path, item_delta_matrix)
        print(f"action_delta 已保存: {delta_save_path}")

    # 加载 100 候选集（原始 ID）
    candidates_path = args.candidates_path or os.path.join(args.data_dir, "minimal", "rec_test_candidate100.npz")
    print(f"\n加载 100 候选集: {candidates_path}")
    candidates_raw = np.load(candidates_path, allow_pickle=True)['candidates']
    print(f"候选集 shape: {candidates_raw.shape}")

    # item 流行度（new id -> 训练集交互次数）
    item_popularity = {iid: len(users) for iid, users in rec_dataset.item_hist_inter_dict.items()}

    # 评估
    print(f"\n=== 在 100 候选集上评估 CMB 完整方法 ===")
    t0 = time()
    results = evaluate_on_candidates(
        base_model=base_model,
        item_delta_matrix=item_delta_matrix,
        rec_dataset=rec_dataset,
        candidates_raw=candidates_raw,
        item_name_dict=rec_dataset.item_name_dict,
        item_topic_dict=rec_dataset.item_topic_dict,
        item_popularity=item_popularity,
        topic_num=rec_dataset.topic_num,
        device=device,
        k_list=[5, 10, 15, 20],
    )
    print(f"评估耗时: {time()-t0:.1f}s")

    # 输出结果
    header_line = f"{'Metric':<20} {'@5':>10} {'@10':>10} {'@15':>10} {'@20':>10}"
    sep = "=" * 60
    rows = []
    for metric in ['Recall', 'NDCG', 'CatCov', 'ILD', 'Novelty']:
        vals = [results.get(f'{metric}@{k}', 0.0) for k in [5, 10, 15, 20]]
        rows.append(f"{metric:<20}" + "".join(f"{v:>10.4f}" for v in vals))

    print("\n" + sep)
    print("CMB 完整方法评估结果")
    print(sep)
    print(header_line)
    print("-" * 60)
    for r in rows:
        print(r)
    print(sep)

    # 保存
    with open(args.output, 'w') as f:
        f.write(f"CMB 完整方法评估结果\n")
        f.write(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(sep + "\n")
        f.write(header_line + "\n")
        f.write("-" * 60 + "\n")
        for r in rows:
            f.write(r + "\n")
        f.write(sep + "\n")
        f.write("\nRaw results:\n")
        for k in sorted(results):
            f.write(f"  {k}: {results[k]:.6f}\n")
    print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
