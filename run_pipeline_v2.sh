#!/usr/bin/env bash
# =============================================================================
# 全自动串行实验流水线 v2
# 剩余任务（DRDW automotive 已完成，跳过）:
#   1. CPGRec  automotive  (重新训练，修复梯度+早停)
#   2. CMB     automotive  (重新训练 epoch=80, 早停=5, 然后评估)
#   3. DRDW    toys
#   4. CPGRec  toys
#   5. CMB     toys
#
# 特性：
#   - caffeinate 防止 macOS 熄屏/睡眠
#   - 串行执行，单进程满负荷
#   - 日志每 30 分钟打印一次等待进度
#   - 每步完成自动进入下一步，中途失败立即报告
#
# 用法:
#   nohup caffeinate -i bash run_pipeline_v2.sh > /tmp/pipeline_v2.log 2>&1 &
# =============================================================================
set -euo pipefail

REPO=/Users/shaoshixuan/Desktop/code_review
LOG_DIR=/tmp/exp_logs_v2
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ts()   { date '+%Y-%m-%d %H:%M:%S'; }
log()  { echo -e "${GREEN}[$(ts)] $*${NC}"; }
warn() { echo -e "${YELLOW}[$(ts)] $*${NC}"; }
err()  { echo -e "${RED}[$(ts)] $*${NC}"; exit 1; }
step() { echo -e "${CYAN}[$(ts)] ===== $* =====${NC}"; }

run_step() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    step "START: $name"
    if "$@" 2>&1 | tee "$logfile"; then
        log "DONE: $name"
    else
        err "FAIL: $name  →  $logfile"
    fi
}

# -----------------------------------------------------------------------
# 1. AUTOMOTIVE - CPGRec  (重新训练，修复端到端梯度 + 早停)
# -----------------------------------------------------------------------
step "AUTOMOTIVE CPGRec (重新训练)"
# 旧结果已不可信（loss 恒为 0.6931），删除旧输出，保留缓存（数据正确）
rm -f "$REPO/CPGRec/output/automotive/cpgrec_best.pt" \
      "$REPO/CPGRec/output/automotive/test_results.txt" 2>/dev/null || true

run_step "automotive_cpgrec_train" \
    python3 "$REPO/CPGRec/main.py" \
        --data_dir  "$REPO/data_automotive" \
        --output_tag automotive

# -----------------------------------------------------------------------
# 2. AUTOMOTIVE - CMB  (重新训练 epoch=80, patience=5, 然后评估)
# -----------------------------------------------------------------------
step "AUTOMOTIVE CMB (训练 epoch=80 + 评估)"
# 删除旧 checkpoint 以强制重新训练
rm -f "$REPO/CMB/CDs/logs/automotive_logs/base/best.base.model.pth" \
      "$REPO/CMB/CDs/logs/automotive_logs/base/model.model" 2>/dev/null || true

cd "$REPO/CMB/CDs"
run_step "automotive_cmb_train" \
    python3 scripts/train_base_grocery.py \
        --dataset   automotive \
        --data_dir  "$REPO/data_automotive/minimal" \
        --kg_dir    "$REPO/data_automotive/KG-related_Files" \
        --save_path ./dataset_objs_auto \
        --epoch     80 \
        --eval_every 5 \
        --early_stop_patience 5

mkdir -p ./output/automotive
run_step "automotive_cmb_eval" \
    python3 scripts/run_cmb_full_eval.py \
        --data_dir  "$REPO/data_automotive" \
        --dataset   automotive \
        --output    ./output/automotive/cmb_automotive_results.txt
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
            --data_dir  "$REPO/data_toys" \
            --output_tag toys
fi

# -----------------------------------------------------------------------
# 4. TOYS - CPGRec
# -----------------------------------------------------------------------
CPGREC_TOYS_CACHE="$REPO/CPGRec/data/cache/cpgrec_toys_data.pkl"
if [[ -f "$CPGREC_TOYS_CACHE" ]]; then
    warn "CPGRec toys 缓存已存在，跳过数据转换"
else
    run_step "toys_cpgrec_convert" \
        python3 "$REPO/CPGRec/utils/data_converter.py" \
            --data_dir  "$REPO/data_toys" \
            --output_tag toys
fi

run_step "toys_cpgrec_train" \
    python3 "$REPO/CPGRec/main.py" \
        --data_dir  "$REPO/data_toys" \
        --output_tag toys

# -----------------------------------------------------------------------
# 5. TOYS - CMB
# -----------------------------------------------------------------------
cd "$REPO/CMB/CDs"
CMB_TOYS_CKPT="$REPO/CMB/CDs/logs/toys_logs/base/best.base.model.pth"
if [[ -f "$CMB_TOYS_CKPT" ]]; then
    warn "CMB toys checkpoint 已存在，跳过训练"
else
    run_step "toys_cmb_train" \
        python3 scripts/train_base_grocery.py \
            --dataset   toys \
            --data_dir  "$REPO/data_toys/minimal" \
            --kg_dir    "$REPO/data_toys/KG-related_Files" \
            --save_path ./dataset_objs_auto \
            --epoch     80 \
            --eval_every 5 \
            --early_stop_patience 5
fi

mkdir -p ./output/toys
run_step "toys_cmb_eval" \
    python3 scripts/run_cmb_full_eval.py \
        --data_dir  "$REPO/data_toys" \
        --dataset   toys \
        --output    ./output/toys/cmb_toys_results.txt
cd "$REPO"

# -----------------------------------------------------------------------
# 汇总
# -----------------------------------------------------------------------
log "========================================"
log "所有实验完成！结果位置："
log "  DRDW automotive  : $REPO/DRDW/output/automotive/drdw_results.txt  [已有]"
log "  CPGRec automotive: $REPO/CPGRec/output/automotive/test_results.txt"
log "  CMB automotive   : $REPO/CMB/CDs/output/automotive/cmb_automotive_results.txt"
log "  DRDW toys        : $REPO/DRDW/output/toys/drdw_results.txt"
log "  CPGRec toys      : $REPO/CPGRec/output/toys/test_results.txt"
log "  CMB toys         : $REPO/CMB/CDs/output/toys/cmb_toys_results.txt"
log "========================================"
