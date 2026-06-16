import argparse
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def arg_parser_preprocessing_grocery():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        dest="dataset",
                        type=str,
                        default="automotive")
    parser.add_argument("--data_dir",
                        dest="data_dir",
                        type=str,
                        default=str(DEFAULT_DATA_DIR.parent / "data_automotive" / "minimal"),
                        help="path to the minimal data directory")
    parser.add_argument("--kg_dir",
                        dest="kg_dir",
                        type=str,
                        default=str(DEFAULT_DATA_DIR.parent / "data_automotive" / "KG-related_Files"),
                        help="path to the KG files directory")
    parser.add_argument("--sample_ratio",
                        dest="sample_ratio",
                        type=int,
                        default=3,
                        help="the (negative: positive sample) ratio for training BPR loss")
    parser.add_argument("--feature_dims",
                        dest="feature_dims",
                        type=int,
                        default=50,
                        help="dims of features")
    parser.add_argument("--save_path",
                        dest="save_path",
                        type=str,
                        default="./dataset_objs/",
                        help="The path to save the preprocessed dataset object")
    return parser.parse_args()


def arg_parse_train_base_grocery():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        dest="dataset",
                        type=str,
                        default="automotive")
    parser.add_argument("--gpu",
                        dest="gpu",
                        action="store_true",
                        default=False,
                        help="whether to use gpu")
    parser.add_argument("--cuda",
                        dest="cuda",
                        type=str,
                        default='0',
                        help="which cuda")
    parser.add_argument("--weight_decay",
                        dest="weight_decay",
                        type=float,
                        default=1e-3,
                        help="L2 norm to the weights")
    parser.add_argument("--lr",
                        dest="lr",
                        type=float,
                        default=0.005,
                        help="learning rate for training")
    parser.add_argument("--epoch",
                        dest="epoch",
                        type=int,
                        default=200,
                        help="training epoch")
    parser.add_argument("--batch_size",
                        dest="batch_size",
                        type=int,
                        default=10240,
                        help="batch size for training base rec model")
    parser.add_argument("--rec_k",
                        dest="rec_k",
                        type=int,
                        default=20,
                        help="length of rec list")
    return parser.parse_args()


def arg_parse_div_optimize_grocery():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        dest="dataset",
                        type=str,
                        default="automotive")
    parser.add_argument("--base_model_path",
                        dest="base_model_path",
                        type=str,
                        default="./logs/")
    parser.add_argument("--base_model_epoch",
                        dest="base_model_epoch",
                        type=int,
                        default=200,
                        help="epoch of the base model to load")
    parser.add_argument("--gpu",
                        dest="gpu",
                        action="store_true",
                        default=False,
                        help="whether to use gpu")
    parser.add_argument("--cuda",
                        dest="cuda",
                        type=str,
                        default='0',
                        help="which cuda")
    parser.add_argument("--data_obj_path",
                        dest="data_obj_path",
                        type=str,
                        default="./dataset_objs/",
                        help="the path to the saved dataset object in the training phase")
    parser.add_argument("--rec_topk",
                        dest="rec_topk",
                        type=int,
                        default=20,
                        help="length of rec list to calculate the ILAD")
    parser.add_argument("--rec_k",
                        dest="rec_k",
                        type=int,
                        default=20,
                        help="length of rec list to evaluate")
    parser.add_argument("--epoch",
                        dest="epoch",
                        type=int,
                        default=200,
                        help="# of epochs in optimization")
    parser.add_argument("--num_arms",
                        dest="num_arms",
                        type=int,
                        default=61,
                        help="nums of arms")
    parser.add_argument("--exp_rate",
                        dest="exp_rate",
                        type=float,
                        default=0.1,
                        help="the epsilon rate of greedy")
    parser.add_argument("--mask_type",
                        dest="mask_type",
                        type=int,
                        default=2,
                        help="mask type: 1=user, 2=item, 3=both")
    parser.add_argument("--save_path",
                        dest="save_path",
                        type=str,
                        default="./explanation_objs/",
                        help="save the counterfactual explanation results")
    parser.add_argument("--output_recs",
                        dest="output_recs",
                        type=str,
                        default="./output/cmb_recommendations.npz",
                        help="path to save final recommendation results")
    return parser.parse_args()
