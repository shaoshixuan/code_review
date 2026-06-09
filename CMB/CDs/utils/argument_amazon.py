import argparse


def arg_parser_preprocessing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        dest="dataset",
                        type=str,
                        default="cds_and_vinyl")
    parser.add_argument("--ratings_dir",
                        dest="ratings_dir",
                        type=str,
                        default="./datasets/CDs/CDs_ratings.txt",
                        help="path to pre-extracted ratings data")
    parser.add_argument("--topics_dir",
                        dest="topics_dir",
                        type=str,
                        default="./datasets/CDs/CDs_topics.txt",
                        help="path to original item topic data")
    parser.add_argument("--sample_ratio",
                        dest="sample_ratio",
                        type=int,
                        default=3,
                        help="the (negative: positive sample) ratio for training BPR loss")
    parser.add_argument("--split_ratio",
                        dest="split_ratio",
                        type=int,
                        default=0.2,
                        help="split the datasets")
    parser.add_argument("--feature_dims",
                        dest="feature_dims",
                        type=int,
                        default=50,
                        help="dims of featues")
    parser.add_argument("--save_path",
                        dest="save_path",
                        type=str,
                        default="./dataset_objs/",
                        help="The path to save the preprocessed dataset object")
    return parser.parse_args()


def arg_parse_train_base():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        dest="dataset",
                        type=str,
                        default="cds_and_vinyl")
    parser.add_argument("--gpu",
                        dest="gpu",
                        action="store_false",
                        help="whether to use gpu")
    parser.add_argument("--cuda",
                        dest="cuda",
                        type=str,
                        default='4',
                        help="which cuda")
    parser.add_argument("--weight_decay",
                        dest="weight_decay",
                        type=float,
                        default='1e-3',
                        help="L2 norm to the wights")
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


def arg_parse_div_optimize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        dest="dataset",
                        type=str,
                        default="cds_and_vinyl")
    parser.add_argument("--base_model_path",
                        dest="base_model_path",
                        type=str,
                        default="./logs/")
    parser.add_argument("--gpu",
                        dest="gpu",
                        action="store_false",
                        help="whether to use gpu")
    parser.add_argument("--cuda",
                        dest="cuda",
                        type=str,
                        default='4',
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
                        help="length of rec list to evulation")
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
    parser.add_argument("--val_arms",
                        dest="val_arms",
                        type=int,
                        default=0.3,
                        help="values scale of arms")
    parser.add_argument("--lambda_value",
                        dest="lambda_value",
                        type=int,
                        default=0.9,
                        help="trade-off parameter")
    parser.add_argument("--L1_norm",
                        dest="L1_norm",
                        type=int,
                        default=5,
                        help="L1 norm to the wights")
    parser.add_argument("--exp_rate",
                        dest="exp_rate",
                        type=float,
                        default=0.1,
                        help="the eposile rate of greedy")
    parser.add_argument("--save_path",
                        dest="save_path",
                        type=str,
                        default="./explanation_objs/",
                        help="save the conterfactual explanation results")
    return parser.parse_args()
