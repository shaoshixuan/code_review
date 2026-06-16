"""
将原始KG数据转换为CPGRec所需的格式
CPGRec需要的结构：
1. 用户-物品二部图 (user, play, item) / (item, played by, user)
2. 物品AND图 (co_genre_pub, co_genre_dev, co_dev_pub) - 共享两种属性的物品连边
3. 物品OR图 (co_or) - 共享任意一种属性的物品连边
4. 社交图 (user, friend of, user) - 我们的数据没有社交关系，跳过

映射关系：
  Steam的genre -> 我们的category (from kg_entities)
  Steam的developer -> 我们的brand (from kg_entities)  
  Steam的publisher -> 我们的feature (from kg_entities, 取前几个feature作为publisher类似物)
  Steam的play time -> 我们的purchase (所有边权重=1.0, sigmoid score)

用户ID: 0-57821 (57822 users, already 0-based)
物品ID: 重映射为0-based (原始 57822-98515)
"""

import numpy as np
from collections import defaultdict
import os
import sys
import pickle
import argparse
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=str(PROJECT_ROOT / 'data_automotive'))
parser.add_argument('--output_tag', type=str, default='automotive')
ARGS, _ = parser.parse_known_args()
DATA_DIR = Path(ARGS.data_dir)
KG_DIR = DATA_DIR / 'KG-related_Files'
MINIMAL_DIR = DATA_DIR / 'minimal'
CACHE_DIR = Path(__file__).resolve().parents[1] / 'data' / 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_PATH = CACHE_DIR / f'cpgrec_{ARGS.output_tag}_data.pkl'
GRAPH_PATH = CACHE_DIR / f'graph_user_item_{ARGS.output_tag}.bin'


def _pick_first_existing(base_dir, patterns):
    base_path = Path(base_dir)
    for pattern in patterns:
        matches = sorted(base_path.glob(pattern))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f"No file matched patterns {patterns} under {base_dir}")



def load_kg_entities():
    """加载KG实体ID到名称的映射"""
    entity_id_to_name = {}
    entity_file = _pick_first_existing(KG_DIR, ['kg_entities_*.txt', 'kg_other_entities_*.txt'])
    with open(entity_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                entity_id_to_name[int(parts[1])] = parts[0]
    return entity_id_to_name


def load_kg_triples():
    """加载KG三元组"""
    triples = []
    triple_file = _pick_first_existing(KG_DIR, ['kg_other_triples_*.txt'])
    with open(triple_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triples.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return triples


def load_kg_items():
    """加载item ID列表"""
    items = {}
    item_file = _pick_first_existing(KG_DIR, ['kg_items_*.txt'])
    with open(item_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                items[parts[0]] = int(parts[1])
    return items


def load_relations():
    """加载关系ID到名称的映射"""
    relations = {}
    relation_file = _pick_first_existing(KG_DIR, ['kg_relations_*.txt'])
    with open(relation_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                relations[int(parts[1])] = parts[0]
    return relations


def load_interactions():
    """加载训练/验证/测试交互数据"""
    # 训练交互
    train_interactions = []
    with open(os.path.join(MINIMAL_DIR, 'rec_train.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                train_interactions.append((int(parts[0]), int(parts[1])))

    # 测试候选集 (145932, 102): [user_id, pos_item, neg1,...,neg100]
    test_cands = np.load(os.path.join(MINIMAL_DIR, 'rec_test_candidate100.npz'),
                         allow_pickle=True)['candidates']
    val_cands = np.load(os.path.join(MINIMAL_DIR, 'rec_val_candidate100.npz'),
                        allow_pickle=True)['candidates']

    return train_interactions, test_cands, val_cands


def build_item_attributes(triples, entity_id_to_name, item_orig_ids):
    """
    从KG三元组中提取物品的三种属性：
    - category (relation 8: has_category)
    - brand (relation 7: has_brand)  
    - feature_type (relation 3: has_feature, 作为publisher的替代)
    
    返回: {item_orig_id: [attr_id_list]} 对每种属性
    """
    # 提取各类entity
    categories = {}  # entity_id -> category_name
    brands = {}      # entity_id -> brand_name
    features = {}    # entity_id -> feature_name

    for eid, ename in entity_id_to_name.items():
        if ename.startswith('category::'):
            categories[eid] = ename[len('category::'):]
        elif ename.startswith('brand::'):
            brands[eid] = ename[len('brand::'):]
        elif ename.startswith('feature::'):
            features[eid] = ename[len('feature::'):]

    item_orig_set = set(item_orig_ids)

    # 属性映射: item_orig_id -> list of attribute name IDs
    item_cat = defaultdict(list)   # item -> [cat_id, ...]
    item_brand = defaultdict(list) # item -> [brand_id, ...]
    item_feat = defaultdict(list)  # item -> [feat_id, ...]

    cat_name_to_id = {}
    brand_name_to_id = {}
    feat_name_to_id = {}

    for head, tail, relation in triples:
        if head not in item_orig_set:
            continue
        if relation == 8 and tail in categories:  # has_category
            name = categories[tail]
            if name not in cat_name_to_id:
                cat_name_to_id[name] = len(cat_name_to_id)
            cid = cat_name_to_id[name]
            if cid not in item_cat[head]:
                item_cat[head].append(cid)
        elif relation == 7 and tail in brands:  # has_brand
            name = brands[tail]
            if name not in brand_name_to_id:
                brand_name_to_id[name] = len(brand_name_to_id)
            bid = brand_name_to_id[name]
            if bid not in item_brand[head]:
                item_brand[head].append(bid)
        elif relation == 3 and tail in features:  # has_feature
            name = features[tail]
            if name not in feat_name_to_id:
                feat_name_to_id[name] = len(feat_name_to_id)
            fid = feat_name_to_id[name]
            if fid not in item_feat[head]:
                item_feat[head].append(fid)

    return (dict(item_cat), dict(item_brand), dict(item_feat),
            cat_name_to_id, brand_name_to_id, feat_name_to_id)


def build_and_or_graphs(item_cat, item_brand, item_feat, item_new_ids):
    """
    构建AND图和OR图
    AND图: 三种边类型 (co_cat_brand, co_cat_feat, co_brand_feat)
           共享两种属性的物品间连边
    OR图: 共享任意一种属性的物品间连边
    """
    n_items = len(item_new_ids)
    orig_to_new = {orig: new for new, orig in enumerate(sorted(item_new_ids))}

    # 构建AND边
    co_cat_brand_src, co_cat_brand_dst = [], []
    co_cat_feat_src, co_cat_feat_dst = [], []
    co_brand_feat_src, co_brand_feat_dst = [], []

    # OR边
    or_src, or_dst = [], []

    items_list = list(orig_to_new.keys())

    for i in range(len(items_list)):
        item_i = items_list[i]
        new_i = orig_to_new[item_i]
        cats_i = set(item_cat.get(item_i, []))
        brands_i = set(item_brand.get(item_i, []))
        feats_i = set(item_feat.get(item_i, []))

        for j in range(i + 1, len(items_list)):
            item_j = items_list[j]
            new_j = orig_to_new[item_j]
            cats_j = set(item_cat.get(item_j, []))
            brands_j = set(item_brand.get(item_j, []))
            feats_j = set(item_feat.get(item_j, []))

            share_cat = len(cats_i & cats_j) > 0
            share_brand = len(brands_i & brands_j) > 0
            share_feat = len(feats_i & feats_j) > 0

            # AND edges
            if share_cat and share_brand:
                co_cat_brand_src.extend([new_i, new_j])
                co_cat_brand_dst.extend([new_j, new_i])
            if share_cat and share_feat:
                co_cat_feat_src.extend([new_i, new_j])
                co_cat_feat_dst.extend([new_j, new_i])
            if share_brand and share_feat:
                co_brand_feat_src.extend([new_i, new_j])
                co_brand_feat_dst.extend([new_j, new_i])

            # OR edge
            if share_cat or share_brand or share_feat:
                or_src.extend([new_i, new_j])
                or_dst.extend([new_j, new_i])

    return {
        'co_cat_brand': (torch.tensor(co_cat_brand_src), torch.tensor(co_cat_brand_dst)),
        'co_cat_feat': (torch.tensor(co_cat_feat_src), torch.tensor(co_cat_feat_dst)),
        'co_brand_feat': (torch.tensor(co_brand_feat_src), torch.tensor(co_brand_feat_dst)),
        'co_or': (torch.tensor(or_src), torch.tensor(or_dst)),
    }


def main():
    print("=== Step 1: 加载原始数据 ===")
    entity_id_to_name = load_kg_entities()
    triples = load_kg_triples()
    kg_items = load_kg_items()
    relations = load_relations()
    train_interactions, test_cands, val_cands = load_interactions()

    print(f"  实体数: {len(entity_id_to_name)}")
    print(f"  三元组数: {len(triples)}")
    print(f"  KG物品数: {len(kg_items)}")
    print(f"  训练交互: {len(train_interactions)}")
    print(f"  测试候选: {test_cands.shape}")
    print(f"  验证候选: {val_cands.shape}")

    # 物品原始ID集合 (57822-98515)
    item_orig_ids = sorted(kg_items.values())
    # 训练集中也出现但KG中没有的item
    train_items = set(i for u, i in train_interactions)
    all_item_orig = sorted(set(item_orig_ids) | train_items)
    # 测试/验证中的item
    test_items = set()
    for row in test_cands:
        test_items.add(int(row[1]))
        for x in row[2:]:
            test_items.add(int(x))
    val_items = set()
    for row in val_cands:
        val_items.add(int(row[1]))
        for x in row[2:]:
            val_items.add(int(x))
    all_item_orig = sorted(set(all_item_orig) | test_items | val_items)

    # 物品ID重映射: orig_id -> new_id (0-based)
    item_orig_to_new = {orig: new for new, orig in enumerate(all_item_orig)}
    # 用户数从训练/测试候选中自动推断（用户ID已是 0-based 连续整数）
    all_user_ids = set(u for u, i in train_interactions)
    for row in test_cands:
        all_user_ids.add(int(row[0]))
    for row in val_cands:
        all_user_ids.add(int(row[0]))
    n_users = max(all_user_ids) + 1
    n_items = len(all_item_orig)
    print(f"  自动推断 n_users={n_users}, n_items={n_items}")

    print(f"\n=== Step 2: 构建物品属性 ===")
    item_cat, item_brand, item_feat, cat_map, brand_map, feat_map = \
        build_item_attributes(triples, entity_id_to_name, item_orig_ids)

    print(f"  有category属性的物品: {len(item_cat)}/{n_items}")
    print(f"  有brand属性的物品: {len(item_brand)}/{n_items}")
    print(f"  有feature属性的物品: {len(item_feat)}/{n_items}")
    print(f"  总category数: {len(cat_map)}")
    print(f"  总brand数: {len(brand_map)}")
    print(f"  总feature数: {len(feat_map)}")

    print(f"\n=== Step 3: 构建AND/OR图 ===")
    # 这个步骤对40694个物品非常耗时(O(n^2))
    # 先采样或使用更高效的构建方式
    graph_edges = build_and_or_graphs_efficient(
        item_cat, item_brand, item_feat, item_orig_to_new)

    print(f"  co_cat_brand edges: {len(graph_edges['co_cat_brand'][0])}")
    print(f"  co_cat_feat edges: {len(graph_edges['co_cat_feat'][0])}")
    print(f"  co_brand_feat edges: {len(graph_edges['co_brand_feat'][0])}")
    print(f"  co_or edges: {len(graph_edges['co_or'][0])}")

    print(f"\n=== Step 4: 构建用户-物品交互 ===")
    # 训练交互 (转换为new item ID)
    train_src = [u for u, i in train_interactions if i in item_orig_to_new]
    train_dst = [item_orig_to_new[i] for u, i in train_interactions if i in item_orig_to_new]

    # 训练集中每个用户交互的物品 (用于valid/test mask)
    user_train_items = defaultdict(set)
    for u, i in train_interactions:
        if i in item_orig_to_new:
            user_train_items[u].add(item_orig_to_new[i])

    # 验证/测试数据 (转换为new item ID)
    valid_data = {}  # {user_new_id: [item_new_ids]}
    test_data = {}
    
    # 聚合同一用户的多个验证/测试条目
    val_user_items = defaultdict(list)
    for row in val_cands:
        user_id = int(row[0])
        pos_item = int(row[1])
        if pos_item in item_orig_to_new:
            val_user_items[user_id].append(item_orig_to_new[pos_item])
    for u, items in val_user_items.items():
        valid_data[u] = items

    test_user_items = defaultdict(list)
    for row in test_cands:
        user_id = int(row[0])
        pos_item = int(row[1])
        if pos_item in item_orig_to_new:
            test_user_items[user_id].append(item_orig_to_new[pos_item])
    for u, items in test_user_items.items():
        test_data[u] = items

    # 测试/验证候选集 (100 neg items, 转换为new ID)
    test_candidates_new = []
    for row in test_cands:
        user_id = int(row[0])
        pos_item = int(row[1])
        neg_items = [int(x) for x in row[2:]]
        if pos_item in item_orig_to_new:
            new_row = [user_id, item_orig_to_new[pos_item]] + \
                      [item_orig_to_new[i] for i in neg_items if i in item_orig_to_new]
            test_candidates_new.append(new_row)

    print(f"  训练边数: {len(train_src)}")
    print(f"  验证用户数: {len(valid_data)}")
    print(f"  测试用户数: {len(test_data)}")
    print(f"  测试候选行数: {len(test_candidates_new)}")

    # 物品流行度 (训练集中的交互次数)
    item_popularity = defaultdict(int)
    for i in train_dst:
        item_popularity[i] += 1

    print(f"\n=== Step 5: 保存处理结果 ===")
    cache = {
        'n_users': n_users,
        'n_items': n_items,
        'train_src': torch.tensor(train_src, dtype=torch.long),
        'train_dst': torch.tensor(train_dst, dtype=torch.long),
        'train_weights': torch.ones(len(train_src), dtype=torch.float32),
        'valid_data': valid_data,
        'test_data': test_data,
        'test_candidates': test_candidates_new,
        'item_cat': item_cat,
        'item_brand': item_brand,
        'item_feat': item_feat,
        'graph_edges': graph_edges,
        'item_orig_to_new': item_orig_to_new,
        'item_new_to_orig': {v: k for k, v in item_orig_to_new.items()},
        'user_train_items': {k: list(v) for k, v in user_train_items.items()},
        'item_popularity': dict(item_popularity),
        'cat_map': cat_map,
        'brand_map': brand_map,
        'feat_map': feat_map,
    }

    cache_path = CACHE_PATH
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  数据已保存至: {cache_path}")

    # 也保存一份DGL图
    import dgl
    graph_data = {
        ('user', 'play', 'item'): (cache['train_src'], cache['train_dst']),
        ('item', 'played_by', 'user'): (cache['train_dst'], cache['train_src']),
    }
    graph = dgl.heterograph(graph_data,
                            num_nodes_dict={'user': n_users, 'item': n_items})
    graph.edges['play'].data['time'] = cache['train_weights']
    graph.edges['played_by'].data['time'] = cache['train_weights']

    graph_path = GRAPH_PATH
    dgl.save_graphs(str(graph_path), [graph])
    print(f"  DGL图已保存至: {graph_path}")
    print(f"\n=== 完成! ===")


def build_and_or_graphs_efficient(item_cat, item_brand, item_feat, item_orig_to_new,
                                   max_group_size=200):
    """
    高效构建AND/OR图 - 使用倒排索引避免O(n^2)
    对于每种属性，找到拥有该属性的物品集合，集合内两两连边。
    max_group_size: 某属性若包含的物品数超过此阈值，视为过于泛化，跳过连边，
                    避免 O(n^2) 爆炸（Automotive 数据有 18k+ feature，单组可达数千物品）。
    """
    n_items = len(item_orig_to_new)

    # 倒排索引: attr_id -> set of item_new_ids
    cat_to_items = defaultdict(set)
    brand_to_items = defaultdict(set)
    feat_to_items = defaultdict(set)

    for item_orig, attrs in item_cat.items():
        if item_orig in item_orig_to_new:
            new_id = item_orig_to_new[item_orig]
            for a in attrs:
                cat_to_items[a].add(new_id)

    for item_orig, attrs in item_brand.items():
        if item_orig in item_orig_to_new:
            new_id = item_orig_to_new[item_orig]
            for a in attrs:
                brand_to_items[a].add(new_id)

    for item_orig, attrs in item_feat.items():
        if item_orig in item_orig_to_new:
            new_id = item_orig_to_new[item_orig]
            for a in attrs:
                feat_to_items[a].add(new_id)

    def collect_edges(attr_to_items):
        """对每个属性组（大小 <= max_group_size），收集两两无向边对。"""
        edge_set = set()
        skipped = 0
        for items in attr_to_items.values():
            if len(items) > max_group_size:
                skipped += 1
                continue
            items_list = sorted(items)
            for i in range(len(items_list)):
                for j in range(i + 1, len(items_list)):
                    edge_set.add((items_list[i], items_list[j]))
        return edge_set, skipped

    print(f"  [图构建] max_group_size={max_group_size}")
    cat_edges,   s1 = collect_edges(cat_to_items)
    print(f"  cat_edges pairs: {len(cat_edges)}, skipped large groups: {s1}")
    brand_edges, s2 = collect_edges(brand_to_items)
    print(f"  brand_edges pairs: {len(brand_edges)}, skipped large groups: {s2}")
    feat_edges,  s3 = collect_edges(feat_to_items)
    print(f"  feat_edges pairs: {len(feat_edges)}, skipped large groups: {s3}")

    # AND edges
    co_cat_brand = cat_edges & brand_edges
    co_cat_feat  = cat_edges & feat_edges
    co_brand_feat = brand_edges & feat_edges
    co_or = cat_edges | brand_edges | feat_edges

    def edges_to_tensors(edge_set):
        src, dst = [], []
        for i, j in edge_set:
            src.extend([i, j])
            dst.extend([j, i])
        return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)

    return {
        'co_cat_brand': edges_to_tensors(co_cat_brand),
        'co_cat_feat':  edges_to_tensors(co_cat_feat),
        'co_brand_feat': edges_to_tensors(co_brand_feat),
        'co_or':        edges_to_tensors(co_or),
    }


if __name__ == "__main__":
    main()
