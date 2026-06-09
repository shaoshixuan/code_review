"""
GroceryFood 数据集上运行CMB多样化推荐
在训练好的基础模型基础上，使用 Epsilon-Greedy 多臂赌博机优化推荐多样性
并保存最终推荐结果供后续评估使用
"""
import torch
import numpy as np
import pickle
import os
import tqdm
from pathlib import Path
from utils.argument_grocery import arg_parse_div_optimize_grocery
from models.models import BaseRecModel, DivOptimizationModel
from utils.evaluate_functions import evaluate_model
from time import time
import datetime
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_div(div_args):
    if div_args.gpu and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(div_args.cuda))
        print(f"Using CUDA:{div_args.cuda}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    print("=== 加载数据集 ===")
    dataset_pickle_path = os.path.join(
        div_args.data_obj_path,
        div_args.dataset + "_dataset_obj.pickle")
    with open(dataset_pickle_path, 'rb') as inp:
        rec_dataset = pickle.load(inp)
    print(f"数据集加载完成: {dataset_pickle_path}")
    print(f"  user_num: {rec_dataset.user_num}")
    print(f"  item_num: {rec_dataset.item_num}")
    print(f"  topic_num: {rec_dataset.topic_num}")
    print(f"  feature_dims: {rec_dataset.feature_dims}")

    print("=== 加载基础推荐模型 ===")
    base_model = BaseRecModel(
        rec_dataset.topic_num,
        rec_dataset.feature_dims,
        rec_dataset.user_num,
        rec_dataset.item_num
    ).to(device)

    model_path = os.path.join(
        div_args.base_model_path,
        div_args.dataset + "_logs/base",
        f"epoch-{div_args.base_model_epoch}.base.model.pth")
    
    if not os.path.exists(model_path):
        # 尝试加载最终模型
        model_path = os.path.join(
            div_args.base_model_path,
            div_args.dataset + "_logs/base",
            "model.model")
    
    print(f"加载模型: {model_path}")
    base_model.load_state_dict(torch.load(model_path, map_location=device))
    base_model.eval()

    # 冻结基础模型参数
    for name, param in base_model.named_parameters():
        param.requires_grad = False

    print("基础模型embedding范围:")
    print(f"  item embedding max: {base_model.item_embedding_matrix.weight.data.max():.4f}")
    print(f"  item embedding min: {base_model.item_embedding_matrix.weight.data.min():.4f}")

    # 创建多样化优化模型
    # 注意: DivOptimizationModel 需要 mask_type 属性
    if not hasattr(div_args, 'mask_type'):
        div_args.mask_type = 2  # 默认mask item embedding

    opt_model = DivOptimizationModel(
        base_model=base_model,
        rec_dataset=rec_dataset,
        device=device,
        div_args=div_args,
    )

    out_path = os.path.join("./logs", div_args.dataset + "_logs/divs/ILAD")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    print("=== 初始化多臂赌博机 ===")
    num_arm = div_args.num_arms
    n_agents = rec_dataset.item_num * rec_dataset.feature_dims

    arm = np.zeros((n_agents, num_arm))
    for agent in range(n_agents):
        arm[agent] = np.random.normal(0, 0.6, num_arm)

    estimated_rewards = np.zeros((n_agents, num_arm), dtype=float)
    action_delta = np.zeros(n_agents, dtype='float32')
    action_list = np.zeros(n_agents, dtype=int)
    num = np.zeros((n_agents, num_arm), dtype=int)

    exp_rate = div_args.exp_rate
    total_rewards = 0.0

    best_ILAD = -1
    best_action_delta = None
    best_epoch = 0
    best_recommendations = None

    print(f"=== 开始CMB多样化优化 (共{div_args.epoch}个epoch) ===")

    for epoch in tqdm.trange(div_args.epoch):
        t1 = time()

        # Epsilon-Greedy 策略为每个agent选择action
        for agent in range(n_agents):
            if np.random.random() < exp_rate:
                action = np.random.randint(low=0, high=num_arm)
            else:
                action = np.argmax(estimated_rewards[agent])
            action_delta[agent] = arm[agent][action]
            action_list[agent] = action

        item_delta_matrix = action_delta.reshape(
            (rec_dataset.item_num, rec_dataset.feature_dims))

        precision, recall, ndcg, andcg, subtopic_coverage, coverage, ILAD = evaluate_model(
            rec_dataset.test_data, rec_dataset.items, item_delta_matrix,
            rec_dataset.topic_num, rec_dataset.user_topic_dict,
            rec_dataset.item_topic_dict, div_args.rec_k, opt_model, device)

        output_str = (
            f"Epoch {epoch+1}: "
            f"Precision={precision}, Recall={recall}, "
            f"NDCG={ndcg}, anDCG={andcg}, "
            f"ST_Coverage={subtopic_coverage}, Coverage={coverage}, "
            f"ILAD={ILAD}"
        )
        logger.info(output_str)

        # 以ILAD为奖励信号
        rewards = np.mean(ILAD)

        # 更新estimated_rewards
        for agent in range(n_agents):
            num[agent][action_list[agent]] += 1
            estimated_rewards[agent][action_list[agent]] = (
                (num[agent][action_list[agent]] - 1) *
                estimated_rewards[agent][action_list[agent]] +
                rewards) / num[agent][action_list[agent]]

        total_rewards += rewards
        average_rewards = total_rewards / (epoch + 1)

        # 记录最佳结果
        if np.mean(ILAD) > best_ILAD:
            best_ILAD = np.mean(ILAD)
            best_action_delta = action_delta.copy()
            best_epoch = epoch + 1
            # 生成并保存最佳推荐列表
            best_recommendations = get_recommendations(
                rec_dataset, opt_model, item_delta_matrix, div_args.rec_k, device)

        t2 = time()
        print(f"Epoch {epoch+1}: ILAD={np.mean(ILAD):.4f}, NDCG={np.mean(ndcg):.4f}, "
              f"avg_reward={average_rewards:.4f}, time={t2-t1:.1f}s")

    print(f"\n=== 优化完成 ===")
    print(f"最佳ILAD: {best_ILAD:.4f} at epoch {best_epoch}")

    # 保存推荐结果
    print("=== 保存推荐结果 ===")
    output_dir = os.path.dirname(div_args.output_recs)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    np.savez(div_args.output_recs,
             recommendations=best_recommendations,
             item_original_ids=np.array(getattr(rec_dataset, 'item_original_ids', rec_dataset.items)),
             best_epoch=np.array([best_epoch]),
             best_ILAD=np.array([best_ILAD]))
    print(f"推荐结果保存至: {div_args.output_recs}")
    print(f"推荐结果形状: {best_recommendations.shape}")

    # 保存最终的action delta
    np.save(os.path.join(out_path, 'best_action_delta.npy'), best_action_delta)

    return best_recommendations


def get_recommendations(rec_dataset, model, action_delta, k, device):
    """为测试集中的每个用户生成Top-K推荐列表"""
    model.eval()
    all_items = rec_dataset.items

    recommendations = []  # [[user_id, [rec_item1, ..., rec_itemK]], ...]

    with torch.no_grad():
        for row in rec_dataset.test_data:
            user = row[0]
            pre_items = all_items
            user_features = np.array([user] * len(pre_items))
            item_features = np.array(pre_items)

            if isinstance(action_delta, np.ndarray):
                scores, _ = model.base_model_new(
                    torch.from_numpy(user_features).to(device),
                    torch.from_numpy(item_features).to(device),
                    torch.from_numpy(action_delta).to(device))
            else:
                scores = model.base_model(
                    torch.from_numpy(user_features).to(device),
                    torch.from_numpy(item_features).to(device))

            scores = np.array(scores.to('cpu'))
            sort_index = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
            sorted_items = [pre_items[i] for i in sort_index[:k]]
            recommendations.append([user] + sorted_items)

    return np.array(recommendations)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    div_args = arg_parse_div_optimize_grocery()

    if div_args.gpu and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = div_args.cuda

    print(f"Args: {div_args}")
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start: {start_time}")

    generate_div(div_args)

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"End: {end_time}")
