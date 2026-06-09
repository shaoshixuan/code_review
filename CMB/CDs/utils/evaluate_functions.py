import numpy as np
import torch
import math
from utils.alpha_nDCG import AlphaNDCG


def evaluate_model_batch(test_data, items, action_delta, topic_num,
                         user_feature_dict, item_feature_dict, k, model, device):
    """
    优化后的批量评估函数 - 一次性计算所有用户得分
    """
    model.eval()
    test_andcg = AlphaNDCG(query_topics=user_feature_dict, doc_topics=item_feature_dict)
    items_arr = np.array(items)
    n_items = len(items)

    with torch.no_grad():
        user_emb = model.user_embedding_matrix.weight.detach()  # (user_num, dim)
        item_emb = model.item_embedding_matrix.weight.detach()  # (item_num, dim)

        if isinstance(action_delta, np.ndarray) and action_delta.ndim == 2:
            # DivOptimizationModel: item embedding is shifted by action_delta
            item_emb_shifted = item_emb + torch.from_numpy(action_delta).float().to(device)
        else:
            item_emb_shifted = item_emb

        item_representations = item_emb_shifted.cpu().numpy()

    precisions, recalls, ndcgs, alpha_ndcgs, st_coverages, coverages, ILADs = [], [], [], [], [], [], []
    chunk_size = 4  # @5, @10, @15, @20

    # 批量计算得分
    batch_size = 1000
    all_user_scores = {}

    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        user_ids = np.array([row[0] for row in batch])
        u_emb = user_emb[torch.tensor(user_ids)].cpu()
        if isinstance(action_delta, np.ndarray) and action_delta.ndim == 2:
            scores_batch = torch.mm(u_emb, torch.from_numpy(item_representations).float().T).numpy()
        else:
            scores_batch = torch.mm(u_emb, item_emb.cpu().T).numpy()

        for j, row in enumerate(batch):
            all_user_scores[row[0]] = scores_batch[j]

    for row in test_data:
        user = row[0]
        item_labels = row[1]
        scores = all_user_scores[user]
        sort_index = np.argsort(scores)[::-1]
        sorted_items = [items[i] for i in sort_index[:max(20, k)]]

        for topk in [5, 10, 15, 20]:
            ndcg = ndcg_k([item_labels], [sorted_items], topk)
            precision = precision_at_k([item_labels], [sorted_items], topk)
            recall = recall_at_k([item_labels], [sorted_items], topk)
            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)
            st_coverage = test_andcg.calculate_single_SubTopic_Coverage(
                ranking=sorted_items, depth=topk) / topic_num
            st_coverages.append(st_coverage)
            coverages.append(sorted_items[:topk])
            ILAD = get_ILAD(sorted_items, item_representations, topk)
            ILADs.append(ILAD)

        ideal_ranking = test_andcg.get_ideal_ranking(query=user, atual_ranking=sorted_items)
        andcgs = test_andcg.compute_single_Alpha_nDCG(
            query=user, target_ranking=sorted_items, ideal_ranking=ideal_ranking)
        if len(andcgs) >= 20:
            alpha_ndcgs.append(andcgs[4])
            alpha_ndcgs.append(andcgs[9])
            alpha_ndcgs.append(andcgs[14])
            alpha_ndcgs.append(andcgs[19])
        elif len(andcgs) > 0:
            alpha_ndcgs.extend([andcgs[-1]] * 4)
        else:
            alpha_ndcgs.extend([0.0] * 4)

    avg_precision = np.mean(np.array(precisions).reshape(-1, chunk_size), axis=0)
    avg_recall = np.mean(np.array(recalls).reshape(-1, chunk_size), axis=0)
    ave_ndcg = np.mean(np.array(ndcgs).reshape(-1, chunk_size), axis=0)
    ave_andcg = np.mean(np.array(alpha_ndcgs).reshape(-1, chunk_size), axis=0)
    avg_st_coverage = np.mean(np.array(st_coverages).reshape(-1, chunk_size), axis=0)

    avg_coverage = np.zeros(chunk_size)
    for i in range(chunk_size):
        all_items_at_k = []
        for j in range(0, len(coverages), chunk_size):
            if j + i < len(coverages):
                all_items_at_k.extend(coverages[j + i])
        avg_coverage[i] = len(set(all_items_at_k)) / len(items)

    avg_ILAD = np.mean(np.array(ILADs).reshape(-1, chunk_size), axis=0)

    return avg_precision, avg_recall, ave_ndcg, ave_andcg, avg_st_coverage, avg_coverage, avg_ILAD


def evaluate_model(test_data, items, action_delta, topic_num,
                   user_feature_dict, item_feature_dict, k, model, device):
    model.eval()
    test_andcg = AlphaNDCG(query_topics=user_feature_dict, doc_topics=item_feature_dict)
    precisions, recalls, ndcgs, alpha_ndcgs, st_coverages, coverages, ILADs = [], [], [], [], [], [], []
    with torch.no_grad():
        for row in test_data:
            user = row[0]
            pre_items = items
            item_labels = row[1]
            user_features = np.array([user] * len(pre_items))
            item_features = np.array(pre_items)
            if model.__class__.__name__ == "BaseRecModel":
                scores = model(
                    torch.from_numpy(user_features).to(device),
                    torch.from_numpy(item_features).to(device)).squeeze()
                item_representations = model.item_embedding_matrix.weight
            elif model.__class__.__name__ == "DivOptimizationModel":
                scores, item_representations = model.base_model_new(
                    torch.from_numpy(user_features).to(device),
                    torch.from_numpy(item_features).to(device),
                    torch.from_numpy(action_delta).to(device))
                scores = scores.squeeze()
            scores = np.array(scores.to('cpu'))
            item_representations = np.array(item_representations.to('cpu'))
            sort_index = sorted(range(len(scores)),
                                key=lambda k: scores[k],
                                reverse=True)
            sorted_items = [pre_items[i] for i in sort_index[:k]]

            for k in [5, 10, 15, 20]:
                ndcg = ndcg_k([item_labels], [sorted_items], k)
                precision = precision_at_k([item_labels], [sorted_items], k)
                recall = recall_at_k([item_labels], [sorted_items], k)
                precisions.append(precision)
                recalls.append(recall)
                ndcgs.append(ndcg)
                st_coverage = test_andcg.calculate_single_SubTopic_Coverage(
                    ranking=sorted_items, depth=k) / topic_num
                st_coverages.append(st_coverage)
                coverage = sorted_items[:k]
                coverages.append(coverage)
                ILAD = get_ILAD(sorted_items, item_representations, k)
                ILADs.append(ILAD)

            ideal_ranking = test_andcg.get_ideal_ranking(
                query=user, atual_ranking=sorted_items)
            andcgs = test_andcg.compute_single_Alpha_nDCG(
                query=user,
                target_ranking=sorted_items,
                ideal_ranking=ideal_ranking)
            if len(andcgs) >= 20:
                alpha_ndcgs.append(andcgs[5 - 1])
                alpha_ndcgs.append(andcgs[10 - 1])
                alpha_ndcgs.append(andcgs[15 - 1])
                alpha_ndcgs.append(andcgs[-1])
            elif len(andcgs) > 0:
                alpha_ndcgs.extend([andcgs[-1]] * 4)
            else:
                alpha_ndcgs.extend([0.0] * 4)

    chunk_size = 4
    avg_precisions = [
        precisions[i:i + chunk_size]
        for i in range(0, len(precisions), chunk_size)
    ]
    avg_precision = np.mean(avg_precisions, axis=0)
    avg_recalls = [
        recalls[i:i + chunk_size] for i in range(0, len(recalls), chunk_size)
    ]
    avg_recall = np.mean(avg_recalls, axis=0)
    avg_ndcgs = [
        ndcgs[i:i + chunk_size] for i in range(0, len(ndcgs), chunk_size)
    ]
    ave_ndcg = np.mean(avg_ndcgs, axis=0)
    avg_andcgs = [
        alpha_ndcgs[i:i + chunk_size]
        for i in range(0, len(alpha_ndcgs), chunk_size)
    ]
    ave_andcg = np.mean(avg_andcgs, axis=0)
    avg_st_coverages = [
        st_coverages[i:i + chunk_size]
        for i in range(0, len(st_coverages), chunk_size)
    ]
    avg_st_coverage = np.mean(avg_st_coverages, axis=0)

    avg_coverages = [
        coverages[i:i + chunk_size]
        for i in range(0, len(coverages), chunk_size)
    ]
    avg_coverage = np.zeros(chunk_size)
    for i in range(chunk_size):
        all_items_at_k = []
        for user_chunk in avg_coverages:
            if i < len(user_chunk):
                all_items_at_k.extend(user_chunk[i])
        avg_coverage[i] = len(set(all_items_at_k)) / len(items)

    avg_ILADs = [
        ILADs[i:i + chunk_size] for i in range(0, len(ILADs), chunk_size)
    ]
    avg_ILAD = np.mean(avg_ILADs, axis=0)

    return avg_precision, avg_recall, ave_ndcg, ave_andcg, avg_st_coverage, avg_coverage, avg_ILAD


def get_ILAD(predicted, item_representations, topk):
    list_items = item_representations[predicted][:topk]
    norms = np.linalg.norm(list_items, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    list_items = list_items / norms
    dis_matrix = np.dot(list_items, list_items.T)
    dis_matrix = 1 - dis_matrix
    dis_sum = np.sum(dis_matrix) / 2
    n = dis_matrix.shape[0]
    if n <= 1:
        return 0.0
    ILAD = np.divide(dis_sum, n * (n - 1))
    return ILAD


def coverage_at_k(predicted, topk):
    items_list = []
    num_users = len(predicted)
    for i in range(num_users):
        item_set = set(predicted[i][:topk])
        items_list = list(item_set)
    return items_list / num_users


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users if true_users > 0 else 0.0


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([
            int(predicted[user_id][j] in set(actual[user_id])) /
            math.log(j + 2, 2) for j in range(topk)
        ])
        res += dcg_k / idcg
    return res / float(len(actual))


def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
