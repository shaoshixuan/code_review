#!/usr/bin/env bash
# =============================================================================
# 全自动串行实验流水线
# 依次运行: automotive DRDW -> automotive CPGRec -> automotive CMB
#           toys DRDW      -> toys CPGRec      -> toys CMB
# 每步完成后才启动下一步，避免 CPU/内存争抢
# 用法: bash run_all_experiments.sh [--skip-auto-drdw]
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/exp_logs}"
mkdir -p "$LOG_DIR"

# 颜色输出
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log()  { echo -e "${GREEN}[$(ts)] $*${NC}"; }
warn() { echo -e "${YELLOW}[$(ts)] $*${NC}"; }
err()  { echo -e "${RED}[$(ts)] $*${NC}"; exit 1; }

# 运行一个命令，打印实时输出，失败则退出
run_step() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    log "=== 开始: $name ==="
    if "$@" 2>&1 | tee "$logfile"; then
        log "=== 完成: $name ==="
    else
        err "=== 失败: $name (见 $logfile) ==="
    fi
}

# -----------------------------------------------------------------------
# 0. 检查是否跳过 automotive DRDW（已在外部跑）
# -----------------------------------------------------------------------
SKIP_AUTO_DRDW=0
for arg in "$@"; do [[ "$arg" == "--skip-auto-drdw" ]] && SKIP_AUTO_DRDW=1; done

# -----------------------------------------------------------------------
# 1. AUTOMOTIVE - DRDW
# -----------------------------------------------------------------------
if [[ $SKIP_AUTO_DRDW -eq 0 ]]; then
    run_step "automotive_drdw" \
        python3 "$REPO/DRDW/run_drdw.py" \
            --data_dir "$REPO/data_automotive" \
            --output_tag automotive
else
    warn "跳过 automotive DRDW（外部已跑或正在跑）"
fi

# -----------------------------------------------------------------------
# 2. AUTOMOTIVE - CPGRec (converter + main)
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
# 3. AUTOMOTIVE - CMB (train + eval)
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
    cd "$REPO"
fi

cd "$REPO/CMB/CDs"
run_step "automotive_cmb_eval" \
    python3 scripts/run_cmb_full_eval.py \
        --data_dir "$REPO/data_automotive" \
        --dataset  automotive \
        --output   ./output/automotive/cmb_automotive_results.txt
cd "$REPO"

log "========================================"
log "AUTOMOTIVE 全部完成，开始 TOYS 数据集"
log "========================================"

# -----------------------------------------------------------------------
# 4. TOYS - DRDW
# -----------------------------------------------------------------------
run_step "toys_drdw" \
    python3 "$REPO/DRDW/run_drdw.py" \
        --data_dir "$REPO/data_toys" \
        --output_tag toys

# -----------------------------------------------------------------------
# 5. TOYS - CPGRec (converter + main)
# -----------------------------------------------------------------------
CPGREC_TOYS_CACHE="$REPO/CPGRec/data/cache/cpgrec_toys_data.pkl"
if [[ -f "$CPGREC_TOYS_CACHE" ]]; then
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
# 6. TOYS - CMB (train + eval)
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
    cd "$REPO"
fi

cd "$REPO/CMB/CDs"
run_step "toys_cmb_eval" \
    python3 scripts/run_cmb_full_eval.py \
        --data_dir "$REPO/data_toys" \
        --dataset  toys \
        --output   ./output/toys/cmb_toys_results.txt
cd "$REPO"

log "========================================"
log "所有实验完成！"
log "结果目录："
log "  DRDW automotive : $REPO/DRDW/output/automotive/"
log "  CPGRec automotive: $REPO/CPGRec/output/automotive/"
log "  CMB automotive  : $REPO/CMB/CDs/output/automotive/"
log "  DRDW toys       : $REPO/DRDW/output/toys/"
log "  CPGRec toys     : $REPO/CPGRec/output/toys/"
log "  CMB toys        : $REPO/CMB/CDs/output/toys/"
log "========================================"
