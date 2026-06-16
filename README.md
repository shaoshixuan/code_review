# Recommendation Diversity / Accuracy Reproduction Experiments

This repository contains reproduction code, dataset adapters, and evaluation results for multiple recommendation methods on three Amazon-style recommendation datasets:

- `data/` — GroceryFood (original dataset already used in earlier experiments)
- `data_automotive/` — Automotive dataset
- `data_toys/` — Toys and Games dataset

The main reproduced methods are:

1. **CMB** — Category-aware Multi-armed Bandit diversity optimization
2. **CPGRec** — Category/brand/feature graph recommendation
3. **D-RDW** — Diversity-aware Random Walk recommendation
4. **IERec** — Interest Entropy sequential recommendation based on RecBole

The evaluation uses the same 100-candidate setting where each row has:

```text
[user_id, positive_item_id, negative_item_1, ..., negative_item_100]
```

Metrics reported across methods:

- `Recall@K`
- `NDCG@K`
- `CatCov@K` (global category coverage)
- `ILD@K` (primary-category intra-list diversity)
- `Novelty@K` (`-ln(pop / total_interactions)`)

Unless otherwise noted, results are reported at `K = 5, 10, 15, 20`.

---

## Repository Layout

```text
.
├── data/                       # GroceryFood dataset
├── data_automotive/            # Automotive dataset
├── data_toys/                  # Toys and Games dataset
├── CMB/                        # CMB code and evaluation scripts
├── CPGRec/                     # CPGRec code and evaluation scripts
├── DRDW/                       # D-RDW code and evaluation scripts
├── IERec/                      # IERec reproduction code and custom 100-candidate evaluator
├── CMB_automotive/             # Notes for Automotive CMB reproduction
├── CPGRec_automotive/          # Notes for Automotive CPGRec reproduction
├── DRDW_automotive/            # Notes for Automotive D-RDW reproduction
├── run_pipeline_v2.sh          # Serial pipeline used for CPGRec/CMB/D-RDW follow-up runs
├── run_all_experiments.sh      # General serial experiment runner
└── wait_and_run.sh             # Legacy helper for waiting and serial execution
```

Generated model checkpoints, caches, tensorboard logs, and large intermediate artifacts are intentionally ignored by `.gitignore`.

---

## Environment Notes

Recommended baseline:

```bash
python3 -m pip install numpy scipy pandas torch tqdm pyyaml colorlog tabulate texttable tensorboard thop
```

IERec ships with its own RecBole-based code under `IERec/recbole/`. The local implementation includes compatibility fixes for newer NumPy / PyTorch versions.

---

## Data Format

Each dataset follows this structure:

```text
data_xxx/
├── minimal/
│   ├── rec_train.txt
│   ├── rec_test_candidate100.npz
│   └── rec_val_candidate100.npz
└── KG-related_Files/
    ├── kg_items_*.txt
    ├── kg_other_entities_*.txt or kg_entities_*.txt
    ├── kg_other_triples_*.txt
    ├── kg_relations_*.txt
    └── kg_users_*.txt
```

`rec_test_candidate100.npz` contains the 100-candidate evaluation set. The first candidate item is the positive item, followed by 100 negatives.

---

## Running Experiments

### 1. D-RDW

Example for Automotive:

```bash
python3 DRDW/run_drdw.py \
  --data_dir ./data_automotive \
  --output_tag automotive
```

Example for Toys:

```bash
python3 DRDW/run_drdw.py \
  --data_dir ./data_toys \
  --output_tag toys
```

Results are saved under:

```text
DRDW/output/<dataset>/drdw_results.txt
```

---

### 2. CPGRec

Data conversion:

```bash
python3 CPGRec/utils/data_converter.py \
  --data_dir ./data_automotive \
  --output_tag automotive
```

Training + evaluation:

```bash
python3 CPGRec/main.py \
  --data_dir ./data_automotive \
  --output_tag automotive
```

For Toys, replace `data_automotive` / `automotive` with `data_toys` / `toys`.

Results are saved under:

```text
CPGRec/output/<dataset>/test_results.txt
```

---

### 3. CMB

CMB uses a base model followed by bandit-based diversity optimization. The complete 100-candidate evaluation entry point is:

```bash
cd CMB/CDs
python3 scripts/run_cmb_full_eval.py \
  --data_dir ../../data_automotive \
  --dataset automotive \
  --data_obj_path ./dataset_objs_auto/ \
  --output ./output/automotive/cmb_automotive_results.txt
```

Results are saved under:

```text
CMB/CDs/output/<dataset>/
```

Only text result files are tracked; model checkpoints and `best_action_delta.npy` are intentionally ignored.

---

### 4. IERec

IERec requires converting `rec_train.txt` into RecBole `.inter` format before training.

Convert data:

```bash
cd IERec
python3 prepare_ierec_data.py --dataset grocery
python3 prepare_ierec_data.py --dataset automotive
python3 prepare_ierec_data.py --dataset toys
```

Run the full IERec pipeline:

```bash
cd IERec
bash run_ierec_pipeline.sh
```

Or run only Automotive + Toys after Grocery has finished:

```bash
cd IERec
bash run_ierec_pipeline_v2.sh
```

The final aligned 100-candidate evaluator is:

```bash
python3 eval_ierec_100cand.py \
  --dataset automotive \
  --model_path saved/<checkpoint>.pth \
  --output output/automotive/ierec_100cand_results.txt
```

Results are saved under:

```text
IERec/output/<dataset>/ierec_100cand_results.txt
```

---

## Existing Result Files

Tracked text result files include:

```text
CMB/CDs/output/automotive/cmb_automotive_results.txt
CMB/CDs/output/toys/cmb_toys_results.txt
CPGRec/output/automotive/test_results.txt
CPGRec/output/toys/test_results.txt
DRDW/output/automotive/drdw_results.txt
DRDW/output/toys/drdw_results.txt
IERec/output/grocery/ierec_100cand_results.txt
IERec/output/automotive/ierec_100cand_results.txt
IERec/output/toys/ierec_100cand_results.txt
```

These files can be used to inspect the reproduced metrics without rerunning the full training pipelines.

---

## Important Implementation Notes

### IERec evaluation fix

The IERec `NEW.forward` method requires a `global_seq` argument. The custom evaluator constructs it via:

```python
global_seq = model.global_seq1(torch.zeros_like(item_seq))
seq_out = model.forward(item_seq, item_seq_len, global_seq)
```

This avoids falling back to a raw last-item embedding and ensures that Transformer sequence representations are used.

### KG triple format

The KG triple files use:

```text
head_id    tail_id    relation_id
```

Category relation id is `8` (`has_category`). Evaluation scripts parse this order explicitly.

### Path handling

Pipeline scripts derive their repository path from the script location, so they can run after cloning into a different directory. Log directories can be overridden with environment variables, e.g.:

```bash
LOG_DIR=./logs bash run_pipeline_v2.sh
```

---

## What Is Not Tracked

The repository intentionally excludes:

- model checkpoints (`*.pth`, `*.pt`)
- pickled caches (`*.pkl`, `*.pickle`)
- TensorBoard event files
- training logs
- generated binary graph/cache files
- `data_phones/`

This keeps the repository focused on reproducible code, allowed datasets, and compact text results.
