"""
GroceryFood 数据集上训练CMB基础推荐模型
优化版：减少epoch，加入early stopping，减少评估频率
"""
import torch
import numpy as np
import os
import sys
import tqdm
import pickle
import argparse
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from torch.utils.data import DataLoader
from scripts.preprocessing.dataset_init import dataset_init_grocery
from models.data_loaders import UserItemInterDataset
from models.models import BaseRecModel
from utils.evaluate_functions import evaluate_model_batch
import torch.nn.functional as F
import logging
from time import time


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CMB base model on GroceryFood dataset')
    # Preprocessing args
    parser.add_argument("--dataset", type=str, default="automotive")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR.parent / "data_automotive" / "minimal"))
    parser.add_argument("--kg_dir", type=str, default=str(DEFAULT_DATA_DIR.parent / "data_automotive" / "KG-related_Files"))
    parser.add_argument("--sample_ratio", type=int, default=3)
    parser.add_argument("--feature_dims", type=int, default=50)
    parser.add_argument("--save_path", type=str, default="./dataset_objs/")
    # Training args
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epoch", type=int, default=50,
                        help="训练轮数，根据实验观察50轮后指标基本收敛")
    parser.add_argument("--batch_size", type=int, default=10240)
    parser.add_argument("--rec_k", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5,
                        help="每隔多少epoch进行一次评估")
    parser.add_argument("--early_stop_patience", type=int, default=3,
                        help="early stopping patience（按评估次数计），设0禁用")
    parser.add_argument("--load_cached_dataset", action="store_true", default=False,
                        help="是否加载已缓存的数据集对象（跳过数据预处理）")
    return parser.parse_args()


def train_base_recommendation(args):
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.cuda))
        print(f"Using CUDA:{args.cuda}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    dataset_pickle_path = os.path.join(args.save_path, args.dataset + "_dataset_obj.pickle")

    if args.load_cached_dataset and os.path.exists(dataset_pickle_path):
        print(f"=== 加载缓存的数据集对象: {dataset_pickle_path} ===")
        with open(dataset_pickle_path, 'rb') as f:
            rec_dataset = pickle.load(f)
    else:
        print("=== 加载/预处理数据集 ===")
        rec_dataset = dataset_init_grocery(args)
        with open(dataset_pickle_path, 'wb') as outp:
            pickle.dump(rec_dataset, outp, pickle.HIGHEST_PROTOCOL)
        print(f"数据集对象保存至: {dataset_pickle_path}")

    train_loader = DataLoader(
        dataset=UserItemInterDataset(rec_dataset.training_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4)

    print(f"feature_dims: {rec_dataset.feature_dims}")
    print(f"topic_num: {rec_dataset.topic_num}")
    print(f"user_num: {rec_dataset.user_num}")
    print(f"item_num: {rec_dataset.item_num}")
    print(f"training_samples: {len(rec_dataset.training_data)}")
    print(f"epochs: {args.epoch}, eval_every: {args.eval_every}")

    model = BaseRecModel(
        rec_dataset.topic_num,
        rec_dataset.feature_dims,
        rec_dataset.user_num,
        rec_dataset.item_num
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

    out_path = os.path.join("./logs", args.dataset + "_logs/base")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    # Early stopping 相关
    best_ndcg = -1
    best_epoch = -1
    no_improve_count = 0
    best_model_path = os.path.join(out_path, "best.base.model.pth")

    print("\n=== 开始训练 ===")
    for epoch in tqdm.trange(args.epoch):
        t0 = time()
        model.train()
        optimizer.zero_grad()
        losses = []
        for user_feat, pos_item_feat, neg_item_feat in train_loader:
            user_feat = user_feat.to(device)
            pos_item_feat = pos_item_feat.to(device)
            neg_item_feat = neg_item_feat.to(device)
            pos_score = model(user_feat, pos_item_feat)
            neg_score = model(user_feat, neg_item_feat)
            loss = -F.logsigmoid(pos_score - neg_score).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.to('cpu').detach().numpy())

        ave_train = np.mean(np.array(losses))
        t1 = time()

        if (epoch + 1) % args.eval_every == 0:
            precision, recall, ndcg, andcg, subtopic_coverage, coverage, ILAD = evaluate_model_batch(
                rec_dataset.test_data, rec_dataset.items, 0,
                rec_dataset.topic_num, rec_dataset.user_topic_dict,
                rec_dataset.item_topic_dict, args.rec_k, model, device)
            t2 = time()

            cur_ndcg = float(ndcg[-1])  # NDCG@20
            output_str = (
                f"Epoch {epoch+1}: loss={ave_train:.4f}, "
                f"P={[f'{x:.4f}' for x in precision]}, "
                f"R={[f'{x:.4f}' for x in recall]}, "
                f"NDCG={[f'{x:.4f}' for x in ndcg]}, "
                f"ILAD={[f'{x:.4f}' for x in ILAD]}, "
                f"train_time={t1-t0:.0f}s, eval_time={t2-t1:.0f}s"
            )
            print(output_str)
            logger.info(output_str)

            # 保存当前checkpoint
            torch.save(
                model.state_dict(),
                os.path.join(out_path, f'epoch-{epoch+1}.base.model.pth'))

            # Early stopping 判断
            if cur_ndcg > best_ndcg:
                best_ndcg = cur_ndcg
                best_epoch = epoch + 1
                no_improve_count = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  >>> 新的最佳模型: NDCG@20={best_ndcg:.4f} at epoch {best_epoch}")
            else:
                no_improve_count += 1
                print(f"  No improvement ({no_improve_count}/{args.early_stop_patience}). Best: NDCG@20={best_ndcg:.4f} at epoch {best_epoch}")
                if args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
                    print(f"\n=== Early stopping triggered at epoch {epoch+1} ===")
                    break
        else:
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: loss={ave_train:.4f}, time={t1-t0:.0f}s")

    torch.save(model.state_dict(), os.path.join(out_path, "model.model"))
    print(f"\n=== 训练完成 ===")
    print(f"最终模型保存至: {out_path}/model.model")
    print(f"最佳模型 (NDCG@20={best_ndcg:.4f}) 保存至: {best_model_path}")
    print(f"最佳epoch: {best_epoch}")
    return best_epoch


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    args = parse_args()
    print(f"Args: {args}")
    best_epoch = train_base_recommendation(args)
    print(f"\nDone. Best epoch: {best_epoch}")
