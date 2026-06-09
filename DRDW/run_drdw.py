"""
D-RDW main runner for GroceryFood dataset.

Data source: repository-root data directory
- Training: data/minimal/rec_train.txt
- Test: data/minimal/rec_test_candidate100.npz (145932 samples, 100 candidates each)
- KG: data/KG-related_Files/ (for item category/brand attributes)

D-RDW algorithm:
  - Builds bipartite user-item graph
  - Computes multi-hop random walk (9 hops) to score items per user
  - Re-ranks candidates by RDW score (diversity-aware via random walk structure)
  - Evaluated on the same 100-candidate test set as CMB and CPGRec

Metrics (aligned with evaluate_ours_and_xiaorong.py):
  - Recall@K, NDCG@K: relevance
  - CatCov@K: global category coverage
  - ILD@K: intra-list diversity (primary category comparison)
  - Novelty@K: -ln(pop/total_interactions)
"""

import os
import sys
import time
import math
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drdw.drdw_standalone import D_RDW

# ============================================================
# Paths  (resolved from repository root)
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
MINIMAL_DIR = os.path.join(DATA_DIR, "minimal")
KG_DIR = os.path.join(DATA_DIR, "KG-related_Files")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(MINIMAL_DIR, "rec_train.txt")
TEST_FILE  = os.path.join(MINIMAL_DIR, "rec_test_candidate100.npz")
KG_TRIPLES  = os.path.join(KG_DIR, "kg_other_triples_Grocery_and_Gourmet_Food.txt")
KG_ENTITIES = os.path.join(KG_DIR, "kg_entities_Grocery_and_Gourmet_Food.txt")
CACHE_FILE  = os.path.join(PROJECT_DIR, "data", "drdw_data.pkl")

TARGET_SIZE = 20
MAX_HOPS    = 9       # 9-hop random walk

# ============================================================
# Data loading utilities
# ============================================================

def load_entity_names():
    entity_name = {}
    with open(KG_ENTITIES) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity_name[int(parts[1])] = parts[0]
    return entity_name


def load_kg_attributes():
    item_cat_id = {}
    item_brand_id = {}
    with open(KG_TRIPLES) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
            if r == 8:    # has_category
                item_cat_id[h] = t
            elif r == 7:  # has_brand
                item_brand_id[h] = t
    return item_cat_id, item_brand_id


def load_train_interactions():
    interactions = []
    with open(TRAIN_FILE) as f:
        for line in f:
            u, i = line.strip().split('\t')
            interactions.append((int(u), int(i)))
    return interactions


def build_data():
    print("Loading training interactions from:", TRAIN_FILE)
    interactions = load_train_interactions()

    all_users = sorted(set(u for u, _ in interactions))
    all_items = sorted(set(i for _, i in interactions))
    n_users = max(all_users) + 1

    # Item ID remapping: original IDs [57822, 98515] -> 0-based
    item_orig2new = {orig: new for new, orig in enumerate(all_items)}
    item_new2orig = {new: orig for orig, new in item_orig2new.items()}
    n_items = len(all_items)

    print(f"  Users: {len(all_users)}, Items: {n_items}")

    # Sparse user-item matrix
    rows, cols = [], []
    for u, i in interactions:
        rows.append(u)
        cols.append(item_orig2new[i])
    data_ones = np.ones(len(rows), dtype=np.float32)
    train_sparse = csr_matrix((data_ones, (rows, cols)), shape=(n_users, n_items))

    print("Loading KG attributes from:", KG_TRIPLES)
    item_cat_id, item_brand_id = load_kg_attributes()
    entity_name = load_entity_names()

    # Build item_dataframe indexed by new item id (0-based)
    categories, brands = [], []
    for new_id in range(n_items):
        orig_id = item_new2orig[new_id]
        cat_id   = item_cat_id.get(orig_id)
        brand_id = item_brand_id.get(orig_id)
        cat_name   = entity_name.get(cat_id,   None) if cat_id   else None
        brand_name = entity_name.get(brand_id, None) if brand_id else None
        if cat_name   and '::' in cat_name:   cat_name   = cat_name.split('::')[1]
        if brand_name and '::' in brand_name: brand_name = brand_name.split('::')[1]
        categories.append(cat_name)
        brands.append(brand_name)

    item_df = pd.DataFrame({'category': categories, 'brand': brands}, index=range(n_items))

    # (Not needed for pure RDW mode - no LP distribution sampling)
    target_distributions = {}

    print(f"  item_df shape: {item_df.shape}, "
          f"with category: {item_df['category'].notna().sum()}")

    return {
        'train_sparse':      train_sparse,
        'item_df':           item_df,
        'item_orig2new':     item_orig2new,
        'item_new2orig':     item_new2orig,
        'n_users':           n_users,
        'n_items':           n_items,
        'target_distributions': target_distributions,
        'interactions':      interactions,
    }


# ============================================================
# Metric computation (aligned with evaluate_ours_and_xiaorong.py)
# ============================================================

def compute_metrics_ref(rec_results, item_cat_dict, item_pop_dict,
                        total_interactions, all_categories,
                        k_list=(5, 10, 15, 20)):
    """
    Metrics aligned with evaluate_ours_and_xiaorong.py:
    - CatCov: GLOBAL across all users
    - ILD: primary category comparison (cats[i] != cats[j])
    - Novelty: -ln(pop/total_interactions)
    - NDCG: 1/log2(rank+1), IDCG=1 for single positive
    """
    n_all_cats = len(all_categories)
    metrics = {}

    for k in k_list:
        recalls, ndcgs, ilds, novelties = [], [], [], []
        covered_cats_global = set()

        for entry in rec_results:
            uid, gt_item, topk_full = entry
            topk = topk_full[:k]

            # Recall
            hit = 1 if gt_item in topk else 0
            recalls.append(hit)

            # NDCG
            if gt_item in topk:
                rank = topk.index(gt_item) + 1
                ndcg = 1.0 / math.log2(rank + 1)
            else:
                ndcg = 0.0
            ndcgs.append(ndcg)

            # CatCov (global)
            for iid in topk:
                cat = item_cat_dict.get(iid)
                if cat is not None:
                    covered_cats_global.add(cat)

            # ILD
            cats = [item_cat_dict.get(iid) for iid in topk]
            n_pairs = k * (k - 1) / 2
            if n_pairs > 0:
                diff = sum(1 for a in range(k) for b in range(a + 1, k)
                           if cats[a] != cats[b])
                ilds.append(diff / n_pairs)
            else:
                ilds.append(0.0)

            # Novelty
            nov_scores = []
            for iid in topk:
                pop = item_pop_dict.get(iid, 1)
                p = pop / total_interactions
                nov_scores.append(-math.log(p) if p > 0 else 0.0)
            novelties.append(np.mean(nov_scores))

        cat_cov = len(covered_cats_global) / n_all_cats if n_all_cats > 0 else 0.0
        metrics[k] = {
            'Recall':  np.mean(recalls),
            'NDCG':    np.mean(ndcgs),
            'CatCov':  cat_cov,
            'ILD':     np.mean(ilds),
            'Novelty': np.mean(novelties),
        }
    return metrics


# ============================================================
# Main
# ============================================================

def main():
    # --- Load or build data ---
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE} ...")
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        # Rebuild sparse if missing (old cache used dense)
        if 'train_sparse' not in data:
            print("Rebuilding sparse matrix from dense ...")
            data['train_sparse'] = csr_matrix(data['train_matrix_dense'])
    else:
        data = build_data()
        print(f"Saving data cache to {CACHE_FILE} ...")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(data, f)

    train_sparse  = data.get('train_sparse') if 'train_sparse' in data else csr_matrix(data['train_matrix_dense'])
    item_df       = data['item_df']
    item_orig2new = data['item_orig2new']
    item_new2orig = data['item_new2orig']
    n_users       = data['n_users']
    n_items       = data['n_items']
    interactions  = data['interactions']

    print(f"Train matrix: {train_sparse.shape}, nnz={train_sparse.nnz}")

    # --- Initialize D-RDW model ---
    # Pure RDW score mode: no LP distribution sampling, 9-hop random walk,
    # rank candidates by RDW score (this IS the core D-RDW algorithm)
    print("Initializing D-RDW model (pure rdw_score mode) ...")
    model = D_RDW(
        train_set_rating=train_sparse,
        item_dataframe=item_df,
        diversity_dimension=[],        # no LP-based distribution sampling
        target_distributions={},
        targetSize=TARGET_SIZE,
        maxHops=MAX_HOPS,
        rankingType='rdw_score',
        sampleObjective='rdw_score',
    )

    # --- Load test candidates ---
    print(f"Loading test candidates from: {TEST_FILE}")
    test_data = np.load(TEST_FILE)
    test_candidates = test_data['candidates']   # shape (N, 102): [user, gt, c1..c100]
    print(f"Test samples: {len(test_candidates)}, "
          f"unique users: {len(np.unique(test_candidates[:,0]))}")

    # --- Item popularity and category dicts ---
    item_pop = Counter()
    for _, i in interactions:
        item_pop[item_orig2new[i]] += 1
    total_interactions = sum(item_pop.values())

    item_cat_dict = {}
    for new_id in range(n_items):
        cat = item_df.loc[new_id, 'category']
        if cat is not None and not (isinstance(cat, float) and math.isnan(cat)):
            item_cat_dict[new_id] = cat
    all_categories = set(item_cat_dict.values())
    print(f"Items with category: {len(item_cat_dict)}, total categories: {len(all_categories)}")

    # --- Evaluate ---
    print("Running D-RDW evaluation on full test set ...")
    rec_results = []
    t0 = time.time()
    errors = 0

    for idx, row in enumerate(test_candidates):
        if idx % 10000 == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(test_candidates) - idx) / rate / 60 if rate > 0 else 0
            print(f"  [{idx}/{len(test_candidates)}] {elapsed:.0f}s elapsed, "
                  f"ETA {eta:.1f} min")

        user_id      = int(row[0])
        gt_item_orig = int(row[1])
        candidates_orig = row[1:].tolist()   # gt + 100 negatives (101 total)

        candidates_new = [item_orig2new[c] for c in candidates_orig if c in item_orig2new]
        gt_item_new    = item_orig2new.get(gt_item_orig, -1)

        if user_id >= train_sparse.shape[0]:
            errors += 1
            continue

        try:
            ranked, scores = model.rank_for_user(user_id, given_item_pool=candidates_new)
            rec_results.append((user_id, gt_item_new, ranked))
        except Exception as e:
            errors += 1

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min), "
          f"errors={errors}, successful={len(rec_results)}")

    # --- Compute metrics ---
    print("Computing metrics ...")
    metrics = compute_metrics_ref(
        rec_results, item_cat_dict, item_pop,
        total_interactions, all_categories,
        k_list=[5, 10, 15, 20]
    )

    # --- Print and save ---
    result_lines = [
        "D-RDW Evaluation Results (GroceryFood Dataset)\n",
        f"Data: {DATA_DIR}\n",
        f"Train file: {TRAIN_FILE}\n",
        f"Test file: {TEST_FILE}\n",
        f"Total test samples: {len(rec_results)}\n",
        f"MaxHops: {MAX_HOPS}, TargetSize: {TARGET_SIZE}\n",
        f"Mode: pure rdw_score ranking (no LP distribution sampling)\n",
        f"Elapsed: {elapsed:.1f}s\n\n",
    ]

    for k in [5, 10, 15, 20]:
        m = metrics[k]
        line = (f"@{k:2d}: Recall={m['Recall']:.4f}  NDCG={m['NDCG']:.4f}  "
                f"CatCov={m['CatCov']:.4f}  ILD={m['ILD']:.4f}  Novelty={m['Novelty']:.4f}")
        print(line)
        result_lines.append(line + "\n")

    out_path = os.path.join(OUTPUT_DIR, "drdw_results.txt")
    with open(out_path, 'w') as f:
        f.writelines(result_lines)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
