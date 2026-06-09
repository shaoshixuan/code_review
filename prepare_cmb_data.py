"""
准备CMB所需的数据格式：
1. 生成 ratings.txt: user@item@rating@date 格式
2. 生成 topics.txt: item@topic1|topic2 格式

我们的数据:
- rec_train.txt: user_id\titem_id (training interactions)
- rec_test_candidate100.npz: [user_id, pos_item, neg1,...,neg100] (test data)
- rec_val_candidate100.npz: [user_id, pos_item, neg1,...,neg100] (val data)
- kg_other_triples: item -> category 关系
- kg_entities: entity_id -> entity_name

CMB数据格式:
- ratings.txt: user@item@rating@date (无实际评分/时间戳，用0填充)
- topics.txt: item@topic1|topic2|...
"""

import numpy as np
from collections import defaultdict
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / 'data'
KG_DIR = DATA_DIR / 'KG-related_Files'
OUTPUT_DIR = REPO_ROOT / 'CMB' / 'CDs' / 'datasets' / 'GroceryFood'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== Step 1: 加载数据 ===")

# 加载训练交互
train_interactions = []
with open(DATA_DIR / 'minimal' / 'rec_train.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            user_id, item_id = int(parts[0]), int(parts[1])
            train_interactions.append((user_id, item_id))

print(f"训练交互数: {len(train_interactions)}")
print(f"User ID范围: {min(u for u,i in train_interactions)} - {max(u for u,i in train_interactions)}")
print(f"Item ID范围: {min(i for u,i in train_interactions)} - {max(i for u,i in train_interactions)}")

# 加载测试数据 (145932 x 102: [user_id, pos_item, neg1,...neg100])
test_data = np.load(DATA_DIR / 'minimal' / 'rec_test_candidate100.npz', allow_pickle=True)['candidates']
val_data = np.load(DATA_DIR / 'minimal' / 'rec_val_candidate100.npz', allow_pickle=True)['candidates']

print(f"测试数据形状: {test_data.shape}")
print(f"验证数据形状: {val_data.shape}")

# 从测试数据提取测试正样本 (col 0 = user, col 1 = pos_item)
test_interactions = [(int(row[0]), int(row[1])) for row in test_data]
val_interactions = [(int(row[0]), int(row[1])) for row in val_data]

print(f"测试正样本数: {len(test_interactions)}")
print(f"验证正样本数: {len(val_interactions)}")

print("\n=== Step 2: 加载KG类别信息 ===")

# 加载实体信息
entity_id_to_name = {}
with open(KG_DIR / 'kg_entities_Grocery_and_Gourmet_Food.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            entity_name = parts[0]
            entity_id = int(parts[1])
            entity_id_to_name[entity_id] = entity_name

# 提取category实体
categories = {}
for eid, ename in entity_id_to_name.items():
    if ename.startswith('category::'):
        cat_name = ename[len('category::'):]
        categories[eid] = cat_name

print(f"总category数: {len(categories)}")

# 从 kg_other_triples 提取 item -> category 关系 (relation 8 = has_category)
item_categories = defaultdict(list)
with open(KG_DIR / 'kg_other_triples_Grocery_and_Gourmet_Food.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            head, tail, relation = int(parts[0]), int(parts[1]), int(parts[2])
            if relation == 8:  # has_category
                if 57822 <= head <= 98515:  # item range
                    if tail in categories:
                        cat_name = categories[tail]
                        if cat_name not in item_categories[head]:
                            item_categories[head].append(cat_name)

print(f"有类别信息的item数: {len(item_categories)}")
sample_items = list(item_categories.items())[:3]
for item_id, cats in sample_items:
    print(f"  item {item_id}: {cats}")

# 获取所有出现在交互中的item
all_items_in_train = set(i for u, i in train_interactions)
all_items_in_test = set(int(row[1]) for row in test_data)
all_items_in_val = set(int(row[1]) for row in val_data)
all_items_in_neg_test = set()
for row in test_data:
    for item in row[2:]:
        all_items_in_neg_test.add(int(item))
all_items = all_items_in_train | all_items_in_test | all_items_in_val | all_items_in_neg_test

print(f"\n所有出现的item数: {len(all_items)}")
items_with_cats = sum(1 for i in all_items if i in item_categories)
print(f"有类别信息的item数 (在交互中): {items_with_cats}/{len(all_items)}")

# 对没有类别的item，赋予默认类别 "Unknown"
items_without_cats = [i for i in all_items if i not in item_categories]
print(f"无类别的item数: {len(items_without_cats)}")

print("\n=== Step 3: 生成ratings.txt ===")
# 格式: user@item@rating@date (所有交互合并，用0作为rating和date)
# CMB 用这个来构建 user_hist_inter_dict 并划分训练/测试集
# 由于我们已经有了划分好的train/val/test，我们只把训练交互写入ratings.txt
# CMB会内部做train_test_split，我们可以通过把test/val正样本也加入然后控制split ratio来匹配

all_rating_interactions = list(train_interactions) + list(test_interactions) + list(val_interactions)
# 去重
unique_interactions = list(set(all_rating_interactions))
print(f"总交互数(去重后): {len(unique_interactions)}")

# 写入ratings.txt
ratings_path = OUTPUT_DIR / 'GroceryFood_ratings.txt'
with open(ratings_path, 'w') as f:
    for i, (user_id, item_id) in enumerate(unique_interactions):
        # format: user@item@rating@date
        # 使用index作为date来模拟时间顺序（训练集排在前面）
        f.write(f"{user_id}@{item_id}@1@{i}\n")

print(f"ratings.txt 写入完成: {ratings_path}")
print(f"  总行数: {len(unique_interactions)}")

print("\n=== Step 4: 生成topics.txt ===")
# 格式: item@topic1|topic2|...
topics_path = OUTPUT_DIR / 'GroceryFood_topics.txt'
with open(topics_path, 'w') as f:
    for item_id in sorted(all_items):
        if item_id in item_categories:
            topics = item_categories[item_id]
        else:
            topics = ['Grocery & Gourmet Food']  # 默认类别
        # item_id已经是数字，直接用
        f.write(f"{item_id}@{'|'.join(topics)}\n")

print(f"topics.txt 写入完成: {topics_path}")
print(f"  总行数: {len(all_items)}")

print("\n=== Step 5: 统计信息 ===")
# 统计topics分布
topic_counts = defaultdict(int)
for item_id in all_items:
    cats = item_categories.get(item_id, ['Grocery & Gourmet Food'])
    for cat in cats:
        topic_counts[cat] += 1

print(f"总topic数: {len(topic_counts)}")
print(f"Top 10 topics:")
for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {topic}: {count}")

print("\n=== 完成！ ===")
print(f"生成文件:")
print(f"  {ratings_path}")
print(f"  {topics_path}")
