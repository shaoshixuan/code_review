"""
Standalone evaluation script for CPGRec
Loads best saved model and runs full evaluation on test set
"""

import os
import sys
import pickle
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.model import CPGRec
from main import build_graphs, compute_metrics, EMBED_DIM, N_LAYERS, DROPOUT, TOPK_LIST, OUTPUT_DIR, DATA_DIR, CACHE_DIR, DEVICE

def main():
    # Load data
    print("Loading data...", flush=True)
    cache_path = os.path.join(CACHE_DIR, 'cpgrec_data.pkl')
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    print("Building graphs...", flush=True)
    ui_graph, and_graph, or_graph = build_graphs(data)

    # Load model
    model = CPGRec(
        n_users=data['n_users'],
        n_items=data['n_items'],
        embed_dim=EMBED_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    best_model_path = os.path.join(OUTPUT_DIR, 'cpgrec_best.pt')
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {best_model_path}", flush=True)

    # Compute embeddings
    print("Computing embeddings...", flush=True)
    ui_graph_dev = ui_graph.to(DEVICE)
    and_graph_dev = and_graph.to(DEVICE)
    or_graph_dev = or_graph.to(DEVICE)
    
    with torch.no_grad():
        user_emb, item_emb = model.forward_graph_emb(ui_graph_dev, and_graph_dev, or_graph_dev)
    user_emb = user_emb.cpu()
    item_emb = item_emb.cpu()

    # Compute all scores
    candidates = data['test_candidates']
    item_cat_raw = data['item_cat']
    item_new_to_orig = data['item_new_to_orig']
    # item_cat_raw uses orig IDs; remap to new IDs
    item_cat = {}
    for new_id, orig_id in item_new_to_orig.items():
        if orig_id in item_cat_raw:
            item_cat[new_id] = item_cat_raw[orig_id]
    item_pop = data['item_popularity']

    print(f"Scoring {len(candidates)} test candidates...", flush=True)
    t0 = time.time()
    all_scores = []
    for idx, row in enumerate(candidates):
        if idx % 20000 == 0:
            print(f"  {idx}/{len(candidates)} ({time.time()-t0:.1f}s)", flush=True)
        user_id = int(row[0])
        pos_item = int(row[1])
        neg_items = [int(x) for x in row[2:]]
        all_cands = [pos_item] + neg_items
        cand_tensor = torch.tensor(all_cands, dtype=torch.long)
        u_vec = user_emb[user_id].unsqueeze(0)
        i_vecs = item_emb[cand_tensor]
        scores = (u_vec * i_vecs).sum(dim=-1).numpy()
        all_scores.append(scores)
    print(f"Scoring done in {time.time()-t0:.1f}s", flush=True)

    scores_matrix = np.array(all_scores)
    labels = np.zeros(len(candidates), dtype=int)

    print("Computing metrics...", flush=True)
    results = compute_metrics(scores_matrix, labels, item_cat, item_pop, TOPK_LIST)

    # Print results
    print("\n=== CPGRec Test Results (100-candidate) ===")
    for k in TOPK_LIST:
        print(f"  K={k}:")
        for metric in ['Recall', 'NDCG', 'CatCov', 'ILD', 'Novelty']:
            key = f"{metric}@{k}"
            print(f"    {key:<15} = {results[key]:.4f}")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write("=== CPGRec Test Results (100-candidate) ===\n\n")
        for k in TOPK_LIST:
            f.write(f"K={k}:\n")
            for metric in ['Recall', 'NDCG', 'CatCov', 'ILD', 'Novelty']:
                key = f"{metric}@{k}"
                f.write(f"  {key:<15} = {results[key]:.4f}\n")
            f.write("\n")
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
