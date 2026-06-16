#!/usr/bin/env bash
# =============================================================================
# 等待 DRDW automotive 完成，然后串行跑所有剩余实验
# 用法: nohup bash wait_and_run.sh > /tmp/pipeline_master.log 2>&1 &
# =============================================================================
set -euo pipefail

REPO=/Users/shaoshixuan/Desktop/code_review
LOG_DIR=/tmp/exp_logs
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ts()   { date '+%Y-%m-%d %H:%M:%S'; }
log()  { echo -e "${GREEN}[$(ts)] $*${NC}"; }
warn() { echo -e "${YELLOW}[$(ts)] $*${NC}"; }
err()  { echo -e "${RED}[$(ts)] $*${NC}"; exit 1; }

run_step() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    log ">>> START: $name"
    if "$@" 2>&1 | tee "$logfile"; then
        log ">>> DONE:  $name"
    else
        err ">>> FAIL:  $name  (log: $logfile)"
    fi
}

# -----------------------------------------------------------------------
# 0. 等待当前 DRDW automotive 进程结束
# -----------------------------------------------------------------------
DRDW_PID=23637
if ps -p "$DRDW_PID" > /dev/null 2>&1; then
    log "DRDW automotive (PID $DRDW_PID) 仍在运行，等待其完成..."
    while ps -p "$DRDW_PID" > /dev/null 2>&1; do
        ELAPSED=$(ps -p "$DRDW_PID" -o etime= 2>/dev/null | tr -d ' ' || echo "?")
        PROGRESS=$(tail -n 1 /tmp/drdw_automotive.log 2>/dev/null || echo "...")
        warn "  等待 DRDW... elapsed=$ELAPSED | $PROGRESS"
        sleep 120
    done
    log "DRDW automotive 已结束！"
else
    warn "DRDW automotive PID $DRDW_PID 未找到，跳过等待"
fi

# 检查 DRDW 是否成功产出结果
DRDW_OUT="$REPO/DRDW/output/automotive"
if ls "$DRDW_OUT"/*.txt 2>/dev/null | head -1 | grep -q txt; then
    log "DRDW automotive 结果已存在: $DRDW_OUT"
else
    warn "DRDW automotive 结果未找到，重新运行..."
    run_step "automotive_drdw" \
        python3 "$REPO/DRDW/run_drdw.py" \
            --data_dir "$REPO/data_automotive" \
            --output_tag automotive
fi

# -----------------------------------------------------------------------
# 1. AUTOMOTIVE - CPGRec
# -----------------------------------------------------------------------
CPGREC_CACHE="$REPO/CPGRec/data/cache/cpgrec_automotive_data.pkl"
if [[ -f "$CPGREC_CACHE" ]]; then
    warn "CPGRec automotive 缓存已存在，跳过数据转换"
else
    run_step "automotive_cpgrec_convert" \
        python3 "$REPO/CPGRec/utils/data_converter.py" \
            --data_dir "$REPO/data_automotive" \
            --output_tag automotive
fi

run_step "automotive_cpgrec_train" \
    python3 "$REPO/CPGRec/main.py" \
        --data_dir "$REPO/data_automotive" \
        --output_tag automotive

# -----------------------------------------------------------------------
# 2. AUTOMOTIVE - CMB
# -----------------------------------------------------------------------
CMB_CKPT="$REPO/CMB/CDs/logs/automotive_logs/base/best.base.model.pth"
if [[ -f "$CMB_CKPT" ]]; then
    warn "CMB automotive checkpoint 已存在，跳过训练"
else
    cd "$REPO/CMB/CDs"
    run_step "automotive_cmb_train" \
        python3 scripts/train_base_grocery.py \
            --dataset automotive \
            --data_dir "$REPO/data_automotive/minimal" \
            --kg_dir   "$REPO/data_automotive/KG-related_Files" \
            --save_path ./dataset_objs_auto \
            --epoch 30 --eval_every 5 --early_stop_patience 3
fi

cd "$REPO/CMB/CDs"
mkdir -p ./output/automotive
run_step "automotive_cmb_eval" \
    python3 scripts/run_cmb_full_eval.py \
        --data_dir "$REPO/data_automotive" \
        --dataset  automotive \
        --output   ./output/automotive/cmb_automotive_results.txt
cd "$REPO"

log "======== AUTOMOTIVE 全部完成，开始 TOYS ========"

# -----------------------------------------------------------------------
# 3. TOYS - DRDW
# -----------------------------------------------------------------------
DRDW_TOYS_OUT="$REPO/DRDW/output/toys"
if ls "$DRDW_TOYS_OUT"/*.txt 2>/dev/null | head -1 | grep -q txt; then
    warn "DRDW toys 结果已存在，跳过"
else
    run_step "toys_drdw" \
        python3 "$REPO/DRDW/run_drdw.py" \
            --data_dir "$REPO/data_toys" \
            --output_tag toys
fi

# -----------------------------------------------------------------------
# 4. TOYS - CPGRec
# -----------------------------------------------------------------------
CPGREC_TOYS="$REPO/CPGRec/data/cache/cpgrec_toys_data.pkl"
if [[ -f "$CPGREC_TOYS" ]]; then
    warn "CPGRec toys 缓存已存在，跳过数据转换"
else
    run_step "toys_cpgrec_convert" \
        python3 "$REPO/CPGRec/utils/data_converter.py" \
            --data_dir "$REPO/data_toys" \
            --output_tag toys
fi

run_step "toys_cpgrec_train" \
    python3 "$REPO/CPGRec/main.py" \
        --data_dir "$REPO/data_toys" \
        --output_tag toys

# -----------------------------------------------------------------------
# 5. TOYS - CMB
# -----------------------------------------------------------------------
CMB_TOYS_CKPT="$REPO/CMB/CDs/logs/toys_logs/base/best.base.model.pth"
if [[ -f "$CMB_TOYS_CKPT" ]]; then
    warn "CMB toys checkpoint 已存在，跳过训练"
else
    cd "$REPO/CMB/CDs"
    run_step "toys_cmb_train" \
        python3 scripts/train_base_grocery.py \
            --dataset toys \
            --data_dir "$REPO/data_toys/minimal" \
            --kg_dir   "$REPO/data_toys/KG-related_Files" \
            --save_path ./dataset_objs_auto \
            --epoch 30 --eval_every 5 --early_stop_patience 3
fi

cd "$REPO/CMB/CDs"
mkdir -p ./output/toys
run_step "toys_cmb_eval" \
    python3 scripts/run_cmb_full_eval.py \
        --data_dir "$REPO/data_toys" \
        --dataset  toys \
        --output   ./output/toys/cmb_toys_results.txt
cd "$REPO"

log "========================================"
log "所有实验完成！结果位置："
log "  DRDW automotive : $REPO/DRDW/output/automotive/"
log "  CPGRec automotive: $REPO/CPGRec/output/automotive/"
log "  CMB automotive  : $REPO/CMB/CDs/output/automotive/"
log "  DRDW toys       : $REPO/DRDW/output/toys/"
log "  CPGRec toys     : $REPO/CPGRec/output/toys/"
log "  CMB toys        : $REPO/CMB/CDs/output/toys/"
log "========================================"
