import torch
import numpy as np
import os
import tqdm
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from scripts.preprocessing.dataset_init import dataset_init
from utils.argument_amazon import arg_parse_train_base, arg_parser_preprocessing
from models.data_loaders import UserItemInterDataset
from models.models import BaseRecModel
from utils.evaluate_functions import evaluate_model
import torch.nn.functional as F
import logging
from time import time


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def train_base_recommendation(train_args, pre_processing_args):
    if train_args.gpu:
        device = torch.device('cuda')
    else:
        device = 'cpu'

    rec_dataset = dataset_init(pre_processing_args)
    Path(pre_processing_args.save_path).mkdir(parents=True, exist_ok=True)
    with open(
            os.path.join(pre_processing_args.save_path,
                         pre_processing_args.dataset + "_dataset_obj.pickle"),
            'wb') as outp:
        pickle.dump(rec_dataset, outp, pickle.HIGHEST_PROTOCOL)

    train_loader = DataLoader(dataset=UserItemInterDataset(
        rec_dataset.training_data),
                              batch_size=train_args.batch_size,
                              shuffle=True,
                              num_workers=8)

    print("rec_dataset.feature_dims: ")
    print(rec_dataset.feature_dims)
    model = BaseRecModel(rec_dataset.topic_num, rec_dataset.feature_dims,
                         rec_dataset.user_num, rec_dataset.item_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=train_args.lr,
                                 weight_decay=train_args.weight_decay)

    out_path = os.path.join("./logs", train_args.dataset + "_logs/base")
    Path(out_path).mkdir(parents=True, exist_ok=True)


    # have not used in the evulation stage
    zero_delta = 0

    t00 = time()
    init_metric = evaluate_model(rec_dataset.test_data, rec_dataset.items,
                                 zero_delta, rec_dataset.topic_num,
                                 rec_dataset.user_topic_dict,
                                 rec_dataset.item_topic_dict, train_args.rec_k,
                                 model, device)
    t01 = time()
    output_eva_time = "[%.1f s]" % (t01 - t00)
    print("One evulation time: " + output_eva_time)
    logger.info(output_eva_time)
    print(
        'init precision recall ndcg alpha_ndcg, sub-topic coverage, coverage, ILAD: ',
        init_metric)

    for epoch in tqdm.trange(train_args.epoch):
        t0 = time()
        model.train()
        optimizer.zero_grad()
        losses = []
        for user_behaviour_feature, pos_item_aspect_feature, neg_item_aspect_feature in train_loader:
            user_behaviour_feature = user_behaviour_feature.to(device)
            pos_item_aspect_feature = pos_item_aspect_feature.to(device)
            neg_item_aspect_feature = neg_item_aspect_feature.to(device)
            pos_score = model(user_behaviour_feature,
                              pos_item_aspect_feature).to(device)
            neg_score = model(user_behaviour_feature,
                              neg_item_aspect_feature).to(device)
            # compute the BPR loss
            # loss = -F.logsigmoid(pos_score - neg_score).mean().to(device)
            loss = -F.logsigmoid(pos_score - neg_score).sum().to(device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.to('cpu').detach().numpy())
            ave_train = np.mean(np.array(losses))
        # print('epoch %d: ' % epoch, 'training loss: ', ave_train)
        # compute evaluation metrics
        if (epoch + 1) % 5 == 0:
            precision, recall, ndcg, andcg, subtopic_coverage, coverage, ILAD = evaluate_model(
                rec_dataset.test_data, rec_dataset.items, zero_delta,
                rec_dataset.topic_num, rec_dataset.user_topic_dict,
                rec_dataset.item_topic_dict, train_args.rec_k, model, device)
            output_str = "epoch %d: " % (epoch + 1) + "training loss: " + str(
                ave_train) + ", K = " + str(
                    train_args.rec_k
                ) + ", Precision: " + str(precision) + ", Recall: " + str(
                    recall) + ", NDCG: " + str(ndcg) + ", anDCG: " + str(
                        andcg) + ", Subtopic_Coverage: " + str(
                            subtopic_coverage) + ", Coverage: " + str(
                                coverage) + ", ILAD: " + str(ILAD)
            print(output_str)
            logging.info(output_str)
            print("base_model.item_embedding_matrix.weight.data: ")
            print(model.item_embedding_matrix.weight.data.max())
            print(model.item_embedding_matrix.weight.data.min())
            torch.save(
                model.state_dict(),
                os.path.join(out_path,
                             'epoch-{}.base.model.pth'.format(epoch + 1)))


        t1 = time()
        print('epoch %d: ' % (epoch + 1), 'training loss: ', ave_train,
              'time: %.1f s' % (t1 - t0))

    logging.info("\n")
    logging.info("\n")
    torch.save(model.state_dict(), os.path.join(out_path, "model.model"))
    return 0


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    t_args = arg_parse_train_base()  # training arguments
    p_args = arg_parser_preprocessing()  # pre processing arguments
    if t_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = t_args.cuda
        print("Using CUDA", t_args.cuda)
    else:
        print("Using CPU")
    print(p_args)
    print(t_args)
    train_base_recommendation(t_args, p_args)
