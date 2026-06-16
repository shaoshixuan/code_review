"""
GroceryFood 数据集预处理 - 直接从原始npz格式数据构建CMB所需的数据对象
无需经过中间ratings.txt格式转换，直接使用数字ID

数据格式说明：
- user ids: 0 - 57821 (57822 users)
- item ids: 57822 - 98515 (40694 items)
- rec_train.txt: user_id\titem_id (训练交互)
- rec_test_candidate100.npz: [user_id, pos_item, neg1,...,neg100] (145932 rows)
- rec_val_candidate100.npz: [user_id, pos_item, neg1,...,neg100] (140480 rows)
- kg_other_triples: item -> category 关系
- kg_entities: entity_id -> entity_name
"""

import numpy as np
from collections import defaultdict
import random
import os
from pathlib import Path


def _pick_first_existing(base_dir, patterns):
    base_path = Path(base_dir)
    for pattern in patterns:
        matches = sorted(base_path.glob(pattern))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f"No file matched patterns {patterns} under {base_dir}")


class GroceryFoodDataset():
    def __init__(self, preprocessing_args):
        super().__init__()
        self.args = preprocessing_args
        self.user_name_dict = {}
        self.item_name_dict = {}
        self.topic_name_dict = {}

        self.topics = []
        self.users = []
        self.items = []

        self.user_hist_inter_dict = {}
        self.item_hist_inter_dict = {}

        self.user_num = None
        self.item_num = None
        self.topic_num = None
        self.feature_dims = self.args.feature_dims

        self.user_topic_dict = None
        self.item_topic_dict = None

        self.training_data = None
        self.test_data = None

        self.pre_processing()
        self.build_train_test()

    def pre_processing(self):
        data_dir = self.args.data_dir
        kg_dir = self.args.kg_dir

        print("=== 加载训练交互 ===")
        # 从 rec_train.txt 加载训练交互
        user_hist_inter_dict = defaultdict(list)
        item_hist_inter_dict = defaultdict(list)

        with open(os.path.join(data_dir, 'rec_train.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    user_id, item_id = int(parts[0]), int(parts[1])
                    user_hist_inter_dict[user_id].append(item_id)
                    item_hist_inter_dict[item_id].append(user_id)

        print(f"训练交互: {sum(len(v) for v in user_hist_inter_dict.values())} pairs")

        print("=== 加载测试/验证数据 ===")
        # 加载测试数据: [user_id, pos_item, neg1,...neg100]
        test_candidates = np.load(
            os.path.join(data_dir, 'rec_test_candidate100.npz'),
            allow_pickle=True)['candidates']
        val_candidates = np.load(
            os.path.join(data_dir, 'rec_val_candidate100.npz'),
            allow_pickle=True)['candidates']

        print(f"测试数据: {test_candidates.shape[0]} rows")
        print(f"验证数据: {val_candidates.shape[0]} rows")

        print("=== 加载KG类别信息 ===")
        entity_file = _pick_first_existing(
            kg_dir,
            ["kg_entities_*.txt", "kg_other_entities_*.txt"]
        )
        triple_file = _pick_first_existing(
            kg_dir,
            ["kg_other_triples_*.txt"]
        )

        # 加载实体ID -> 名称映射
        entity_id_to_name = {}
        with open(entity_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    entity_id_to_name[int(parts[1])] = parts[0]

        # 提取category实体
        categories = {}  # {entity_id: category_name}
        for eid, ename in entity_id_to_name.items():
            if ename.startswith('category::'):
                categories[eid] = ename[len('category::'):]

        print(f"总类别数: {len(categories)}")

        # 从 kg_other_triples 提取 item -> category (relation 8 = has_category)
        item_raw_categories = defaultdict(list)
        with open(triple_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    head, tail, relation = int(parts[0]), int(parts[1]), int(parts[2])
                    if relation == 8 and tail in categories:
                        cat_name = categories[tail]
                        if cat_name not in item_raw_categories[head]:
                            item_raw_categories[head].append(cat_name)

        print(f"有类别的item数: {len(item_raw_categories)}")

        print("=== 构建映射 ===")
        # 获取所有item（用item ID的集合）
        all_items_set = set(item_hist_inter_dict.keys())
        # 加入测试和验证中的item
        for row in test_candidates:
            all_items_set.add(int(row[1]))  # pos item
            for item in row[2:]:
                all_items_set.add(int(item))
        for row in val_candidates:
            all_items_set.add(int(row[1]))
            for item in row[2:]:
                all_items_set.add(int(item))

        all_users_set = set(user_hist_inter_dict.keys())

        # CMB内部使用从0开始的连续ID
        # user: 已经是0-57821的连续ID
        # item: 已经是57822-98515，但CMB需要0开始
        # 建立item映射: 原始item_id -> 新item_id (0-based)
        sorted_items = sorted(all_items_set)
        sorted_users = sorted(all_users_set)

        user_name_dict = {u: u for u in sorted_users}  # user ID保持不变
        item_name_dict = {item: idx for idx, item in enumerate(sorted_items)}  # item重新从0开始
        
        # 保存reverse映射，用于后续评估
        self.item_original_ids = sorted_items  # index -> original_id

        print(f"用户数: {len(sorted_users)}")
        print(f"物品数: {len(sorted_items)}")
        print(f"Item ID重映射示例: {sorted_items[0]} -> 0, {sorted_items[1]} -> 1")

        # 构建topic列表和映射
        all_topics = set()
        for item_id in all_items_set:
            cats = item_raw_categories.get(item_id, ['Unknown'])
            all_topics.update(cats)

        topic_list = sorted(all_topics)
        topic_name_dict = {t: idx for idx, t in enumerate(topic_list)}
        print(f"Topic数: {len(topic_list)}")

        # 构建item_topic_dict (新ID -> topic indices)
        item_topic_dict = {}
        for item_id in all_items_set:
            new_item_id = item_name_dict[item_id]
            cats = item_raw_categories.get(item_id, ['Unknown'])
            item_topic_dict[new_item_id] = [topic_name_dict[c] for c in cats]

        # 构建user_topic_dict (user_id -> topic indices)
        user_topic_dict = {}
        for user_id, interacted_items in user_hist_inter_dict.items():
            user_topics = set()
            for item_id in interacted_items:
                if item_id in item_name_dict:
                    new_item_id = item_name_dict[item_id]
                    user_topics.update(item_topic_dict.get(new_item_id, []))
            user_topic_dict[user_id] = list(user_topics)

        # 重建user/item历史交互字典（使用新的item ID）
        new_user_hist_inter_dict = {}
        for user_id in sorted_users:
            items_list = user_hist_inter_dict.get(user_id, [])
            new_items = [item_name_dict[i] for i in items_list if i in item_name_dict]
            if new_items:
                new_user_hist_inter_dict[user_id] = new_items

        new_item_hist_inter_dict = {}
        for item_id, users_list in item_hist_inter_dict.items():
            if item_id in item_name_dict:
                new_item_id = item_name_dict[item_id]
                new_item_hist_inter_dict[new_item_id] = users_list

        # 构建测试数据 [[user_id, [pos_item_new_ids], [neg_item_new_ids_list1], ...], ...]
        # 按用户聚合测试样本
        test_data_dict = defaultdict(lambda: {'pos': [], 'candidates': []})
        for row in test_candidates:
            user_id = int(row[0])
            pos_item_orig = int(row[1])
            neg_items_orig = [int(x) for x in row[2:]]

            if pos_item_orig in item_name_dict:
                pos_item_new = item_name_dict[pos_item_orig]
                test_data_dict[user_id]['pos'].append(pos_item_new)
                neg_new = [item_name_dict[i] for i in neg_items_orig if i in item_name_dict]
                test_data_dict[user_id]['candidates'].extend(neg_new)

        # 构建 test_data 格式: [[user, [pos_items]], ...]
        test_data = []
        for user_id in sorted(test_data_dict.keys()):
            pos_items = test_data_dict[user_id]['pos']
            test_data.append([user_id, pos_items])

        print(f"测试用户数: {len(test_data)}")

        # 存储
        self.user_name_dict = user_name_dict
        self.item_name_dict = item_name_dict
        self.topic_name_dict = topic_name_dict
        self.user_hist_inter_dict = new_user_hist_inter_dict
        self.item_hist_inter_dict = new_item_hist_inter_dict
        self.users = sorted_users
        self.items = list(range(len(sorted_items)))  # 0 - N-1
        self.topics = topic_list
        self.user_topic_dict = user_topic_dict
        self.item_topic_dict = item_topic_dict
        self.user_num = len(sorted_users)
        self.item_num = len(sorted_items)
        self.topic_num = len(topic_list)

        # 保存测试数据（以原始格式，用于后续evaluate）
        self._test_data_raw = np.array(test_data, dtype=object)

        return True

    def build_train_test(self):
        """直接使用原始的train/test划分而不是随机重新划分"""
        print("=== 构建训练数据 ===")
        sample_ratio = getattr(self.args, 'sample_ratio', 3)
        all_items_arr = np.array(self.items)

        training_data = []
        for user, items in self.user_hist_inter_dict.items():
            if len(items) == 0:
                continue
            pos_arr = np.array(items)
            neg_mask = ~np.isin(all_items_arr, pos_arr)
            negative_items = all_items_arr[neg_mask]
            # 对所有正样本批量采样负样本
            n_pos = len(items)
            n_neg = n_pos * sample_ratio
            if len(negative_items) >= n_neg:
                neg_sampled = np.random.choice(negative_items, n_neg, replace=False)
            else:
                neg_sampled = np.random.choice(negative_items, n_neg, replace=True)
            training_pos_items_repeated = pos_arr.repeat(sample_ratio)

            for p_item, n_item in zip(training_pos_items_repeated, neg_sampled):
                training_data.append([user, int(p_item), int(n_item)])

        print(f"# training samples: {len(training_data)}")
        self.training_data = np.array(training_data)
        self.test_data = self._test_data_raw
        print(f"# test samples: {len(self.test_data)}")
        print(f"valid user: {len(self.users)}")
        print(f"valid item: {len(self.items)}")
        print(f"valid topic length: {len(self.topics)}")
        print(f"user density: {len(training_data) / len(self.users):.2f}")

        return True


def grocery_preprocessing(pre_processing_args):
    rec_dataset = GroceryFoodDataset(pre_processing_args)
    return rec_dataset
