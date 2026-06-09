import torch


class BaseRecModel(torch.nn.Module):
    def __init__(self, feature_length, feature_dims, user_num, item_num):
        super(BaseRecModel, self).__init__()
        self.feature_length = feature_length
        self.feature_dims = feature_dims
        self.user_number = user_num
        self.item_number = item_num

        self.user_embedding_matrix = torch.nn.Embedding(
            self.user_number, self.feature_dims)
        self.item_embedding_matrix = torch.nn.Embedding(
            self.item_number, self.feature_dims)
        self.user_embedding_matrix.weight.data = torch.nn.init.xavier_uniform_(
            self.user_embedding_matrix.weight.data)
        self.item_embedding_matrix.weight.data = torch.nn.init.xavier_uniform_(
            self.item_embedding_matrix.weight.data)

    def forward(self, user_feature, item_feature):
        user_embeddings = self.user_embedding_matrix(user_feature)
        item_embeddings = self.item_embedding_matrix(item_feature)

        out = torch.mul(user_embeddings, item_embeddings)
        out = torch.sum(out, -1)

        return out

class DivOptimizationModel(torch.nn.Module):
    def __init__(self, base_model, rec_dataset, device, div_args):
        super(DivOptimizationModel, self).__init__()
        self.base_model = base_model
        self.rec_dataset = rec_dataset
        self.device = device
        self.div_args = div_args

        self.delta_user = torch.nn.Parameter(
            torch.empty(self.rec_dataset.feature_dims,
                        device=self.device).uniform_(-0.0, 0.0))

    def get_masked_user_features(self, user_feature):
        user_feature_stars = torch.add(user_feature, self.delta_user)
        return user_feature_stars

    def get_masked_item_features(self, item_feature_matrix, action_delta):
        item_feature_stars = torch.add(item_feature_matrix, action_delta)
        return item_feature_stars

    def base_model_new(self, user_features, item_features, action_delta):
        # get top_k_items first
        user_embedding = self.base_model.user_embedding_matrix.weight.detach()
        item_embedding = self.base_model.item_embedding_matrix.weight.detach()
        user_embedding_masked = self.get_masked_user_features(user_embedding)
        item_embedding_masked = self.get_masked_item_features(
            item_embedding, action_delta)
        item_feature_matrixs = item_embedding_masked

        # cauclate the scores in base model
        if self.div_args.mask_type == 1:
            scores = torch.sum(
                torch.mul(user_embedding_masked[user_features],
                          item_embedding[item_features]), -1).squeeze()
        elif self.div_args.mask_type == 2:
            scores = torch.sum(
                torch.mul(user_embedding[user_features],
                          item_embedding_masked[item_features]), -1).squeeze()
        elif self.div_args.mask_type == 3:
            scores = torch.sum(
                torch.mul(user_embedding_masked[user_features],
                          item_embedding_masked[item_features]), -1).squeeze()

        return scores, item_feature_matrixs
