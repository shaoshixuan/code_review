import numpy as np
from sklearn.model_selection import train_test_split
import random


class AmazonDataset():
    def __init__(self, preprocessing_args):
        super().__init__()
        self.args = preprocessing_args
        self.user_name_dict = {}  # rename users to integer names
        self.item_name_dict = {}
        self.topic_name_dict = {}

        self.topics = []  # topics list
        self.users = []
        self.items = []

        # the interacted items for each user, sorted with date {user:[i1, i2, i3, ...], user:[i1, i2, i3, ...]}
        self.user_hist_inter_dict = {}
        # the interacted users for each item
        self.item_hist_inter_dict = {}

        self.user_num = None
        self.item_num = None
        self.topic_num = None  # number of features
        self.feature_dims = self.args.feature_dims

        self.user_topic_dict = None  # user topic dict
        self.item_topic_dict = None  # item topic dict

        self.training_data = None
        self.test_data = None
        self.pre_processing()
        self.split_dataset(seed=0)

    def get_user_item_dict(self, ):
        user_dict = {}
        item_dict = {}
        with open(self.args.ratings_dir, 'r') as f:
            line = f.readline().strip()
            while line:
                user = line.split('@')[0]
                item = line.split('@')[1]
                if user not in user_dict:
                    user_dict[user] = [item]
                else:
                    user_dict[user].append(item)
                if item not in item_dict:
                    item_dict[item] = [user]
                else:
                    item_dict[item].append(user)
                line = f.readline().strip()
        return user_dict, item_dict

    def get_topic_list(self, ):
        item_topic = {}
        topic_list = []
        with open(self.args.topics_dir, 'r', encoding='unicode_escape') as f:
            line = f.readline().strip()
            while line:
                item = line.split('@')[0]
                topics = line.split('@')[1]
                for topic in topics.split('|'):
                    if topic not in topic_list:
                        topic_list.append(topic)
                    if item not in item_topic:
                        item_topic[item] = [topic]
                    else:
                        item_topic[item].append(topic)
                line = f.readline().strip()
        topic_list = np.array(topic_list)
        return topic_list, item_topic

    def pre_processing(self, ):
        user_dict, item_dict = self.get_user_item_dict(
        )  # not sorted with time
        topics, item_topic = self.get_topic_list()
        user_item_date_dict = {
        }  # {(user, item): date, (user, item): date ...}  # used to remove duplicate

        with open(self.args.ratings_dir, 'r', encoding='unicode_escape') as f:
            line = f.readline().strip()
            while line:
                user = line.split('@')[0]
                item = line.split('@')[1]
                date = line.split('@')[3]
                if user in user_dict and item in user_dict[user] and (
                        user, item) not in user_item_date_dict:
                    user_item_date_dict[(user, item)] = date
                line = f.readline().strip()

        for key in list(user_item_date_dict.keys()):
            if key[0] not in user_dict or key[1] not in user_dict[key[0]]:
                del user_item_date_dict[key]

        # rename users, items, and topics to integer names
        user_name_dict = {}
        item_name_dict = {}
        topic_name_dict = {}
        print(topics)

        count = 0
        for user in user_dict:
            if user not in user_name_dict:
                user_name_dict[user] = count
                count += 1
        count = 0
        for item in item_dict:
            if item not in item_name_dict:
                item_name_dict[item] = count
                count += 1
        count = 0
        for topic in topics:
            if topic not in topic_name_dict:
                topic_name_dict[topic] = count
                count += 1

        # reindex the item_topic and user_topic for diversity metric
        item_topic_dict = {}
        for key, values in item_topic.items():
            for value in values:
                if key in item_dict and item_name_dict[
                        key] not in item_topic_dict:
                    item_topic_dict[item_name_dict[key]] = [
                        topic_name_dict[value]
                    ]
                elif key in item_dict and item_name_dict[
                        key] in item_topic_dict:
                    item_topic_dict[item_name_dict[key]].append(
                        topic_name_dict[value])

        user_topic_dict = {}
        for key, values in user_dict.items():
            for value in values:
                if user_name_dict[key] not in user_topic_dict:
                    user_topic_dict[user_name_dict[key]] = [
                        item_topic_dict[item_name_dict[value]]
                    ]
                else:
                    user_topic_dict[user_name_dict[key]].append(
                        item_topic_dict[item_name_dict[value]])
        for key, value in user_topic_dict.items():
            user_topic_dict[key] = list(set(sum(value, [])))

        renamed_user_item_date_dict = {}
        for key, value in user_item_date_dict.items():
            renamed_user_item_date_dict[user_name_dict[key[0]],
                                        item_name_dict[key[1]]] = value
        user_item_date_dict = renamed_user_item_date_dict

        # sort with date
        user_item_date_dict = dict(
            sorted(user_item_date_dict.items(), key=lambda item: item[1]))

        user_hist_inter_dict = {
        }  # {"u1": [i1, i2, i3, ...], "u2": [i1, i2, i3, ...]}, sort with time
        item_hist_inter_dict = {}
        # ranked_user_item_dict = {}  # {"u1": [i1, i2, i3, ...], "u2": [i1, i2, i3, ...]}
        for key, value in user_item_date_dict.items():
            user = key[0]
            item = key[1]
            if user not in user_hist_inter_dict:
                user_hist_inter_dict[user] = [item]
            else:
                user_hist_inter_dict[user].append(item)
            if item not in item_hist_inter_dict:
                item_hist_inter_dict[item] = [user]
            else:
                item_hist_inter_dict[item].append(user)

        user_hist_inter_dict = dict(sorted(user_hist_inter_dict.items()))
        item_hist_inter_dict = dict(sorted(item_hist_inter_dict.items()))

        users = list(user_hist_inter_dict.keys())
        items = list(item_hist_inter_dict.keys())

        # self.sentiment_data = sentiment_data
        self.user_name_dict = user_name_dict
        self.item_name_dict = item_name_dict
        self.topic_name_dict = topic_name_dict
        self.user_hist_inter_dict = user_hist_inter_dict
        self.item_hist_inter_dict = item_hist_inter_dict
        self.users = users
        self.items = items
        self.topics = topics
        self.user_topic_dict = user_topic_dict
        self.item_topic_dict = item_topic_dict
        self.user_num = len(users)
        self.item_num = len(items)
        self.topic_num = len(topics)
        return True

    def split_dataset(self, seed=0):
        train_u_i_set = set()
        # item_set = set(self.items)
        training_data = []
        # user_item_label_list = []  # [[u, [item1, item2, ...], [l1, l2, ...]], ...]
        user_item_label_list = []  # [[u, [item1, item2, ...]]
        for user, items in self.user_hist_inter_dict.items():
            training_pos_items, test_pos_items = train_test_split(
                items, test_size=self.args.split_ratio, random_state=seed)
            # generate the training pairs
            negative_items = []
            # the not interacted items of user
            negative_items = [item for item in self.items if item not in items]
            # negative_length = len(training_pos_items) * self.args.sample_ratio
            # # negative_length = min(len(negative_items), negative_length)
            # if len(negative_items) >= negative_length:
            #     training_neg_items = np.random.choice(np.array(negative_items),
            #                                           negative_length,
            #                                           replace=False)
            # else:
            #     training_neg_items = np.random.choice(np.array(negative_items),
            #                                           negative_length,
            #                                           replace=True)
            # training_pos_items = training_pos_items * self.args.sample_ratio

            training_neg_items = []
            for item in training_pos_items:
                training_neg_items += random.sample(negative_items, self.args.sample_ratio)
            training_pos_items = np.array(training_pos_items).repeat(self.args.sample_ratio)

            for p_item, n_item in zip(training_pos_items, training_neg_items):
                training_data.append([user, p_item, n_item])

            # generate the test data
            user_item_label_list.append([user, test_pos_items])

            for item in training_pos_items:
                train_u_i_set.add((user, item))

        print('# training samples :', len(training_data))
        self.training_data = np.array(training_data)
        print('# test samples :', len(user_item_label_list))
        self.test_data = np.array(user_item_label_list)
        print("valid user: ", len(self.users))
        print('valid item : ', len(self.items))
        print("valid topic length: ", len(self.topics))
        print('user dense is:', len(training_data) / len(self.users))

        return True

    def save(self, save_path):
        return True

    def load(self):
        return False


def amazon_preprocessing(pre_processing_args):
    rec_dataset = AmazonDataset(pre_processing_args)
    return rec_dataset
