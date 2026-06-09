import torch
import numpy as np
import pickle
import os
import tqdm
from pathlib import Path
from utils.argument_amazon import arg_parse_div_optimize
from models.models import BaseRecModel, DivOptimizationModel
from utils.evaluate_functions import evaluate_model
from time import time
import datetime
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def generate_div(div_args):
    if div_args.gpu:
        device = torch.device('cuda')
        # device = torch.device('cuda:%s' % div_args.cuda)
    else:
        device = 'cpu'
    # import dataset
    with open(
            os.path.join(div_args.data_obj_path,
                         div_args.dataset + "_dataset_obj.pickle"),
            'rb') as inp:
        rec_dataset = pickle.load(inp)

    base_model = BaseRecModel(rec_dataset.topic_num, rec_dataset.feature_dims,
                              rec_dataset.user_num,
                              rec_dataset.item_num).to(device)
    base_model.load_state_dict(
        torch.load(
            os.path.join(div_args.base_model_path,
                         div_args.dataset + "_logs/base",
                         "epoch-155.base.model.pth")))
    base_model.eval()

    #  fix the rec model
    for name, param in base_model.named_parameters():
        param.requires_grad = False

    print("base_model.item_embedding_matrix.weight.data: ")
    print(base_model.item_embedding_matrix.weight.data.max())
    print(base_model.item_embedding_matrix.weight.data.min())

    # Create optimization model
    opt_model = DivOptimizationModel(
        base_model=base_model,
        rec_dataset=rec_dataset,
        device=device,
        div_args=div_args,
    )


    out_path = os.path.join(
        "./logs", div_args.dataset + "_logs/divs/ILAD-noshare")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    # the multi agents multi armed bandit code
    num_arm = div_args.num_arms

    # tensor(2.3097, device='cuda:0')
    # tensor(-2.0238, device='cuda:0')
    arm = np.zeros((rec_dataset.item_num * rec_dataset.feature_dims, num_arm))
    for agent in range(rec_dataset.item_num * rec_dataset.feature_dims):
        arm[agent] = np.random.normal(0, 0.6, num_arm)

    estimated_rewards = np.zeros(
        (rec_dataset.item_num * rec_dataset.feature_dims, num_arm),
        dtype=float)

    total_rewards = 0.0

    action_delta = np.zeros(rec_dataset.item_num * rec_dataset.feature_dims,
                            dtype='float32')
    action_list = np.zeros(rec_dataset.item_num * rec_dataset.feature_dims,
                           dtype=int)

    ## for epsilon greedy
    exp_rate = div_args.exp_rate
    num = np.zeros((rec_dataset.item_num * rec_dataset.feature_dims, num_arm),
                   dtype=int)

    # ## for ucb
    # # ucb_c = div_args.ucb_c ## 2
    # up_bound = np.zeros((rec_dataset.feature_dims, num_arm))
    # num = np.ones((rec_dataset.feature_dims, num_arm))

    # t00 = time()
    # init_metric = evaluate_model(
    #     rec_dataset.test_data, rec_dataset.items,
    #     action_delta.reshape(rec_dataset.item_num, rec_dataset.feature_dims),
    #     rec_dataset.topic_num, rec_dataset.user_topic_dict,
    #     rec_dataset.item_topic_dict, div_args.rec_k, opt_model, device)
    # t01 = time()
    # output_eva_time = "[%.1f s]" % (t01 - t00)
    # print("One evulation time: " + output_eva_time)
    # logger.info(output_eva_time)
    # print(
    #     'init precision recall ndcg alpha_ndcg, sub-topic coverage, coverage, ILAD: ',
    #     init_metric)

    # The epsilon greedy code
    for epoch in tqdm.trange(div_args.epoch):
        t1 = time()
        for agent in range(rec_dataset.item_num * rec_dataset.feature_dims):
            ## choose an action for each agent
            if np.random.random() < exp_rate:
                action = np.random.randint(low=0, high=num_arm)
            else:
                action = np.argmax(estimated_rewards[agent])
            action_delta[agent] = arm[agent][action]
            # action_delta[agent] = arm[action]
            action_list[agent] = action

    # # UCB code
    # for epoch in tqdm.trange(div_args.epoch):
    #     t1 = time()
    #     for agent in range(rec_dataset.feature_dims):
    #         ## choose an action for each agent
    #         for a in range(num_arm):
    #             if num[agent][a] > 0:
    #                 up_bound[agent][a] = estimated_rewards[agent][a] + np.sqrt(
    #                     2 * np.log(epoch + 1) / num[agent][a])
    #             else:
    #                 up_bound[agent][a] = 1e500
    #         action = np.argmax(up_bound[agent])
    #         action_delta[agent] = arm[agent][action]
    #         action_list[agent] = action

        item_delta_matrix = action_delta.reshape(
            (rec_dataset.item_num, rec_dataset.feature_dims))
        precision, recall, ndcg, andcg, subtopic_coverage, coverage, ILAD = evaluate_model(
            rec_dataset.test_data, rec_dataset.items, item_delta_matrix,
            rec_dataset.topic_num, rec_dataset.user_topic_dict,
            rec_dataset.item_topic_dict, div_args.rec_k, opt_model, device)
        output_str = "epoch %d: " % (epoch + 1) + ", K = " + str(
            div_args.rec_k) + ", Precision: " + str(
                precision) + ", Recall: " + str(recall) + ", NDCG: " + str(
                    ndcg) + ", anDCG: " + str(
                        andcg) + ", Subtopic_Coverage: " + str(
                            subtopic_coverage) + ", Coverage: " + str(
                                coverage) + ", ILAD: " + str(ILAD)
        print(output_str)
        logging.info(output_str)


        x = np.mean(ILAD)
        y = np.mean(ndcg)
        # rewards = 2 * x * y / (x + y)
        rewards = x

        # The epsilon greedy code, the UCB is same
        for agent in range(rec_dataset.item_num * rec_dataset.feature_dims):
            num[agent][action_list[agent]] += 1
            estimated_rewards[agent][action_list[agent]] = (
                (num[agent][action_list[agent]] - 1) *
                estimated_rewards[agent][action_list[agent]] +
                rewards) / num[agent][action_list[agent]]

        total_rewards += rewards
        logging.info(total_rewards)
        average_rewards = total_rewards / (epoch + 1)
        print("Average Rewards: " + str(average_rewards))

        ## save the delta each epoch

        t2 = time()
        output_time_str = "Epoch %d [%.1f s]" % (epoch + 1, t2 - t1)
        print("One epoch time: " + output_time_str)
        logger.info(output_time_str)

    logging.info("\n")
    logging.info("\n")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    div_args = arg_parse_div_optimize()
    if div_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = div_args.cuda
        print("Using CUDA", div_args.cuda)
    else:
        print("Using CPU")
    print(div_args)
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(start_time)
    generate_div(div_args)
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(end_time)
