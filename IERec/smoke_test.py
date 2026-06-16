"""
IERec 流程冒烟测试：2 epoch 训练 + 100-cand 评估
"""
import sys
sys.path.insert(0, '.')
from recbole.quick_start import run_recbole
import os, glob

# 快速训练 2 epoch
result = run_recbole(
    model='NEW',
    dataset='grocery',
    config_file_list=['ierec_custom.yaml'],
    config_dict={
        'use_gpu': False,
        'data_path': 'dataset/',
        'epochs': 2,
        'eval_step': 2,
        'stopping_step': 1,
        'show_progress': False,
        'checkpoint_dir': 'saved/',
    }
)
print(f"Training done. best_valid_score={result.get('best_valid_score')}")

# 找最新保存的模型
models = sorted(glob.glob('saved/NEW-grocery*.pth'))
if models:
    latest = models[-1]
    print(f"Model: {latest}")
    # 运行 100-cand 评估
    os.system(f"python3 eval_ierec_100cand.py --dataset grocery --model_path '{latest}'")
else:
    print("No model saved!")
