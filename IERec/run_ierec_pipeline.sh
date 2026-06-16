#!/usr/bin/env bash
# =============================================================================
# IERec 全数据集实验流水线
# 依次在 grocery / automotive / toys 上:
#   1. 数据格式转换（rec_train.txt → RecBole .inter）
#   2. IERec 训练（RecBole leave-one-out, BPR loss, uni100 valid）
#   3. 100-candidate 评估（与 CPGRec/CMB/DRDW 完全对齐的 5 项指标）
#
# 用法:
#   nohup caffeinate -i bash run_ierec_pipeline.sh > /tmp/ierec_pipeline.log 2>&1 &
# =============================================================================
set -euo pipefail

IEREC=/Users/shaoshixuan/Desktop/code_review/IERec
LOG_DIR=/tmp/ierec_logs
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ts()   { date '+%Y-%m-%d %H:%M:%S'; }
log()  { echo -e "${GREEN}[$(ts)] $*${NC}"; }
warn() { echo -e "${YELLOW}[$(ts)] $*${NC}"; }
err()  { echo -e "${RED}[$(ts)] $*${NC}"; exit 1; }
step() { echo -e "${CYAN}[$(ts)] ===== $* =====${NC}"; }

cd "$IEREC"

for DATASET in grocery automotive toys; do
    step "Dataset: $DATASET"

    # ── 1. 数据转换 ──────────────────────────────────────────────────────────
    INTER_FILE="$IEREC/dataset/$DATASET/$DATASET.inter"
    if [[ -f "$INTER_FILE" ]]; then
        warn "Inter file exists, skip: $INTER_FILE"
    else
        log "Converting $DATASET data to RecBole format..."
        python3 prepare_ierec_data.py --dataset "$DATASET" \
            2>&1 | tee "$LOG_DIR/${DATASET}_data_prep.log"
        log "Data conversion done."
    fi

    # ── 2. 训练 ──────────────────────────────────────────────────────────────
    LATEST_MODEL=$(ls "$IEREC/saved/"NEW-"$DATASET"*.pth 2>/dev/null | sort | tail -1 || echo "")
    if [[ -n "$LATEST_MODEL" ]]; then
        warn "Checkpoint found, skip training: $LATEST_MODEL"
    else
        log "Training IERec on $DATASET..."
        python3 -c "
import sys; sys.path.insert(0,'.')
from recbole.quick_start import run_recbole
run_recbole(
    model='NEW',
    dataset='$DATASET',
    config_file_list=['ierec_custom.yaml'],
    config_dict={
        'use_gpu': False,
        'data_path': 'dataset/',
        'show_progress': False,
        'checkpoint_dir': 'saved/',
    }
)" 2>&1 | tee "$LOG_DIR/${DATASET}_train.log"
        log "Training done."
    fi

    # ── 3. 100-candidate 评估 ─────────────────────────────────────────────────
    OUTPUT_FILE="$IEREC/output/$DATASET/ierec_100cand_results.txt"
    if [[ -f "$OUTPUT_FILE" ]]; then
        warn "Eval results exist, skip: $OUTPUT_FILE"
    else
        LATEST_MODEL=$(ls "$IEREC/saved/"NEW-"$DATASET"*.pth 2>/dev/null | sort | tail -1 || echo "")
        if [[ -z "$LATEST_MODEL" ]]; then
            err "No saved model for $DATASET after training step!"
        fi
        log "Evaluating $DATASET with 100-candidate set..."
        mkdir -p "$IEREC/output/$DATASET"
        python3 eval_ierec_100cand.py \
            --dataset "$DATASET" \
            --model_path "$LATEST_MODEL" \
            --output "$OUTPUT_FILE" \
            2>&1 | tee "$LOG_DIR/${DATASET}_eval.log"
        log "Evaluation done."
    fi

    log "===== $DATASET COMPLETE ====="
    echo ""
done

log "=========================================="
log "所有 IERec 实验完成！评估结果位置："
for DATASET in grocery automotive toys; do
    log "  $DATASET: $IEREC/output/$DATASET/ierec_100cand_results.txt"
done
log "=========================================="
