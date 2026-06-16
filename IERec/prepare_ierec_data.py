"""
将我们的数据格式转换为 RecBole 序列推荐格式 (.inter 文件)

RecBole sequential 格式需要:
  user_id:token  item_id:token  timestamp:float

我们的 rec_train.txt: user_id\titem_id (按顺序排列，用行号作为 timestamp)
RecBole 会按 timestamp 排序构建交互序列，然后用 leave-one-out 切分。

策略:
- 用全局行号作为 timestamp，保证序列顺序
- 对于 leave-one-out: RecBole 会自动把每个用户最后1个 item 作为 test
  第二到最后1个作为 valid，其余作为 train
- 同时保留我们的 100-candidate 测试集，用于最终评估

用法:
  python3 prepare_ierec_data.py --dataset grocery
  python3 prepare_ierec_data.py --dataset automotive
  python3 prepare_ierec_data.py --dataset toys
"""
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[1]

DATASET_DIRS = {
    'grocery':    REPO / 'data'            / 'minimal',
    'automotive': REPO / 'data_automotive' / 'minimal',
    'toys':       REPO / 'data_toys'       / 'minimal',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['grocery', 'automotive', 'toys'])
    args = parser.parse_args()

    minimal_dir = DATASET_DIRS[args.dataset]
    out_dir = Path(__file__).resolve().parent / 'dataset' / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    inter_path = minimal_dir / 'rec_train.txt'
    print(f"Reading interactions from: {inter_path}")

    # 读取训练交互（保序）
    interactions = []  # (user_id, item_id, timestamp)
    with open(inter_path) as f:
        for ts, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) >= 2:
                uid, iid = int(parts[0]), int(parts[1])
                interactions.append((uid, iid, ts))

    print(f"Total interactions: {len(interactions)}")

    # 写 .inter 文件
    out_inter = out_dir / f'{args.dataset}.inter'
    with open(out_inter, 'w') as f:
        f.write('user_id:token\titem_id:token\ttimestamp:float\n')
        for uid, iid, ts in interactions:
            f.write(f'{uid}\t{iid}\t{ts}\n')

    print(f"Written: {out_inter}  ({len(interactions)} rows)")

    # 统计
    users = set(x[0] for x in interactions)
    items = set(x[1] for x in interactions)
    print(f"Users: {len(users)}, Items: {len(items)}")

    # 检查 test candidate 文件
    test_cand = minimal_dir / 'rec_test_candidate100.npz'
    if test_cand.exists():
        cands = np.load(test_cand, allow_pickle=True)['candidates']
        print(f"Test candidates: {cands.shape}")
    print("Done.")


if __name__ == '__main__':
    main()
