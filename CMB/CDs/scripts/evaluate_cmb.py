"""
CMB方法在GroceryFood数据集上的评估脚本
指标包括：Recall@K, NDCG@K, Category Coverage@K, ILD@K, Novelty@K
K = [5, 10, 15, 20]
"""
import torch
import numpy as np
import pickle
import os
import sys
import argparse
import math
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_metrics(test_data, items, item_topic_dict, item_popularity,
                    topic_num, model, device, k_list=(5, 10, 15, 20),
                    action_delta=None):
    """
    计算所有指标：Recall@K, NDCG@K, Category Coverage@K, ILD@K, Novelty@K

    Args:
        test_data: list of [user_id, [pos_item_ids]]
        items: all candidate items (list of new item ids, 0-based)
        item_topic_dict: {item_id: [topic_idx, ...]}
        item_popularity: {item_id: interaction_count}
        topic_num: total number of topics/categories
        model: BaseRecModel or DivOptimizationModel
        device: torch device
        k_list: list of K values to evaluate at
        action_delta: (item_num, feature_dims) numpy array for DivOptimizationModel, or None
    """
    model.eval()
    items_arr = np.array(items)
    n_items = len(items)
    max_k = max(k_list)

    # === 批量计算所有用户 item 得分 ===
    with torch.no_grad():
        if action_delta is not None and isinstance(action_delta, np.ndarray):
            # DivOptimizationModel: 应用 item embedding shift
            user_emb = model.base_model.user_embedding_matrix.weight.detach().cpu()
            item_emb = model.base_model.item_embedding_matrix.weight.detach().cpu()
            delta_t = torch.from_numpy(action_delta).float()
            item_emb_shifted = item_emb + delta_t
        else:
            user_emb = model.user_embedding_matrix.weight.detach().cpu()
            item_emb = model.item_embedding_matrix.weight.detach().cpu()
            item_emb_shifted = item_emb

    # 预计算所有item的topic集合（用set加速ILD计算）
    item_topics_set = {item_id: set(item_topic_dict.get(item_id, []))
                       for item_id in items}

    # 统计收集器
    all_results = defaultdict(list)  # {metric_name@K: [values]}

    # 批量计算分数
    batch_size = 500
    user_id_list = [row[0] for row in test_data]
    pos_items_list = [row[1] for row in test_data]

    all_scores = []
    for i in range(0, len(user_id_list), batch_size):
        batch_users = torch.tensor(user_id_list[i:i+batch_size])
        u_emb = user_emb[batch_users]          # (B, dim)
        scores = torch.mm(u_emb, item_emb_shifted.T)  # (B, n_items)
        all_scores.append(scores.numpy())

    all_scores = np.concatenate(all_scores, axis=0)  # (n_test_users, n_items)

    for idx, (user_id, pos_items) in enumerate(zip(user_id_list, pos_items_list)):
        scores = all_scores[idx]
        sort_index = np.argsort(scores)[::-1]
        ranked_items = [items[i] for i in sort_index[:max_k]]

        pos_set = set(pos_items)

        for k in k_list:
            rec_k = ranked_items[:k]
            rec_set = set(rec_k)

            # --- Recall@K ---
            n_hit = len(pos_set & rec_set)
            recall = n_hit / len(pos_set) if len(pos_set) > 0 else 0.0
            all_results[f'Recall@{k}'].append(recall)

            # --- NDCG@K ---
            dcg = 0.0
            for rank, item in enumerate(rec_k):
                if item in pos_set:
                    dcg += 1.0 / math.log2(rank + 2)
            ideal_len = min(len(pos_set), k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            all_results[f'NDCG@{k}'].append(ndcg)

            # --- Category Coverage@K ---
            covered_topics = set()
            for item in rec_k:
                covered_topics.update(item_topics_set.get(item, set()))
            cat_cov = len(covered_topics) / topic_num if topic_num > 0 else 0.0
            all_results[f'CatCov@{k}'].append(cat_cov)

            # --- ILD@K (Intra-List Diversity based on categories) ---
            # 所有两两组合中，类别不同的比例
            n_pairs = k * (k - 1) / 2
            if n_pairs > 0:
                diff_pairs = 0
                for i in range(k):
                    for j in range(i + 1, k):
                        ti = item_topics_set.get(rec_k[i], set())
                        tj = item_topics_set.get(rec_k[j], set())
                        # 两个item类别完全相同算"相同"，否则算"不同"
                        # 更合理的定义：交集为空则完全不同
                        if len(ti & tj) == 0:
                            diff_pairs += 1
                ild = diff_pairs / n_pairs
            else:
                ild = 0.0
            all_results[f'ILD@{k}'].append(ild)

            # --- Novelty@K ---
            # 流行度越低，新颖度越高；用 -log2(pop/n_users) 计算
            # pop = 训练集中item被交互的次数
            novelty_scores = []
            for item in rec_k:
                pop = item_popularity.get(item, 1)  # 至少1次，避免log(0)
                novelty_scores.append(-math.log2(pop / len(user_id_list)))
            all_results[f'Novelty@{k}'].append(np.mean(novelty_scores))

    # 对所有用户取平均
    final_results = {}
    for key, values in all_results.items():
        final_results[key] = np.mean(values)

    return final_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="grocery_food")
    parser.add_argument("--data_obj_path", type=str, default="./dataset_objs/")
    parser.add_argument("--base_model_path", type=str, default="./logs/")
    parser.add_argument("--model_file", type=str, default="best.base.model.pth",
                        help="模型文件名（位于 base_model_path/dataset_logs/base/ 下）")
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument("--output", type=str, default="./output/cmb_eval_results.txt")
    parser.add_argument("--use_div_delta", type=str, default=None,
                        help="若要评估CMB多样化后的结果，传入 best_action_delta.npy 路径")
    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.cuda))
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # 加载数据集
    dataset_pickle_path = os.path.join(args.data_obj_path, args.dataset + "_dataset_obj.pickle")
    print(f"加载数据集: {dataset_pickle_path}")
    with open(dataset_pickle_path, 'rb') as f:
        rec_dataset = pickle.load(f)

    print(f"user_num: {rec_dataset.user_num}, item_num: {rec_dataset.item_num}")
    print(f"topic_num: {rec_dataset.topic_num}")
    print(f"test_data: {len(rec_dataset.test_data)} users")

    # 加载模型
    from models.models import BaseRecModel, DivOptimizationModel
    base_model = BaseRecModel(
        rec_dataset.topic_num, rec_dataset.feature_dims,
        rec_dataset.user_num, rec_dataset.item_num
    ).to(device)

    model_path = os.path.join(
        args.base_model_path,
        args.dataset + "_logs/base",
        args.model_file)
    print(f"加载模型: {model_path}")
    base_model.load_state_dict(torch.load(model_path, map_location=device))
    base_model.eval()

    # 如果需要评估CMB多样化后的结果
    action_delta = None
    if args.use_div_delta and os.path.exists(args.use_div_delta):
        print(f"加载 action_delta: {args.use_div_delta}")
        action_delta = np.load(args.use_div_delta).reshape(
            rec_dataset.item_num, rec_dataset.feature_dims)
        model = DivOptimizationModel(base_model, rec_dataset, device, type('Args', (), {'mask_type': 2})())
        eval_model = model
    else:
        eval_model = base_model
        action_delta = None

    # 构建item流行度字典 (新item_id -> 训练集中的交互次数)
    item_popularity = {}
    for item_id, users_list in rec_dataset.item_hist_inter_dict.items():
        item_popularity[item_id] = len(users_list)

    print(f"\n=== 开始评估 ===")
    print(f"候选商品数: {len(rec_dataset.items)}")
    print(f"评估用户数: {len(rec_dataset.test_data)}")

    results = compute_metrics(
        test_data=rec_dataset.test_data,
        items=rec_dataset.items,
        item_topic_dict=rec_dataset.item_topic_dict,
        item_popularity=item_popularity,
        topic_num=rec_dataset.topic_num,
        model=eval_model,
        device=device,
        k_list=[5, 10, 15, 20],
        action_delta=action_delta
    )

    # 输出结果
    print("\n" + "="*60)
    print("CMB 方法评估结果 (GroceryFood 数据集)")
    print("="*60)
    header = f"{'Metric':<20} {'@5':>10} {'@10':>10} {'@15':>10} {'@20':>10}"
    print(header)
    print("-"*60)
    for metric_base in ['Recall', 'NDCG', 'CatCov', 'ILD', 'Novelty']:
        vals = [results.get(f'{metric_base}@{k}', 0.0) for k in [5, 10, 15, 20]]
        row = f"{metric_base:<20}" + "".join(f"{v:>10.4f}" for v in vals)
        print(row)
    print("="*60)

    # 保存结果
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write("CMB 方法评估结果 (GroceryFood 数据集)\n")
        f.write("="*60 + "\n")
        f.write(f"{'Metric':<20} {'@5':>10} {'@10':>10} {'@15':>10} {'@20':>10}\n")
        f.write("-"*60 + "\n")
        for metric_base in ['Recall', 'NDCG', 'CatCov', 'ILD', 'Novelty']:
            vals = [results.get(f'{metric_base}@{k}', 0.0) for k in [5, 10, 15, 20]]
            row = f"{metric_base:<20}" + "".join(f"{v:>10.4f}" for v in vals)
            f.write(row + "\n")
        f.write("="*60 + "\n")
        f.write("\nRaw results:\n")
        for k, v in sorted(results.items()):
            f.write(f"  {k}: {v:.6f}\n")
    print(f"\n结果已保存至: {args.output}")

    return results


if __name__ == "__main__":
    main()
