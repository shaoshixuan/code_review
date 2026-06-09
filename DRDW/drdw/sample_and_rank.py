# Pipeline Overview
# Step 1: Perform n-hop random walks to identify candidate items.
# Step 2: Apply sampling strategies to select items meeting target distributions.
# Step 3: Rank the sampled items.

from .graph_recommender import GraphRec
import numpy as np
import random
from .rank_core import ItemRanker
import pandas as pd
from scipy.sparse import csr_matrix
from .sample_core import DistributionSampler


class Sample_And_Rank(object):
    def __init__(self, train_set_rating, articlesDataframe):
        self.itemPool = np.array([])
        self.Model_RDW = GraphRec(train_set_rating)
        self.articlesDataframe = articlesDataframe
        self.articleRdwScore = np.array([])
        self.train_set_rating = train_set_rating
        self.articleNum = train_set_rating.shape[1]
        self.CANDIDATESOLD = []

    def filterHeuristics(self, user_idx, itemPool, filteringCriteria, given_item_pool=None):
        if itemPool is None or len(itemPool) == 0:
            return np.array([])
        filteredItems = np.asarray(itemPool)

        if filteringCriteria is not None:
            filterDim = filteringCriteria['filterDim']
            threshold = filteringCriteria['filterThreshold']
            comparison = filteringCriteria['comparison']
            filtered_rows = self.articlesDataframe.loc[itemPool]
            if filterDim in self.articlesDataframe.columns:
                filtered_df = filtered_rows
                if comparison == "larger":
                    filtered_df = filtered_rows[filtered_rows[filterDim] > threshold]
                elif comparison == "larger_equal":
                    filtered_df = filtered_rows[filtered_rows[filterDim] >= threshold]
                elif comparison == "less":
                    filtered_df = filtered_rows[filtered_rows[filterDim] < threshold]
                elif comparison == "less_equal":
                    filtered_df = filtered_rows[filtered_rows[filterDim] <= threshold]
                elif comparison == "equal":
                    filtered_df = filtered_rows[filtered_rows[filterDim] == threshold]
                elif comparison == "not_equal":
                    filtered_df = filtered_rows[filtered_rows[filterDim] != threshold]
                else:
                    raise ValueError(f"Unknown comparison type: {comparison}")
            filteredItems = filtered_df.index.to_numpy()

        if given_item_pool is not None and len(given_item_pool) > 0:
            given_item_pool_set = set(given_item_pool)
            mask = np.isin(filteredItems, list(given_item_pool_set))
            if mask.shape == filteredItems.shape:
                filteredItems = filteredItems[mask]
            else:
                return np.array([])

        if hasattr(self.train_set_rating, 'toarray'):
            row_data = self.train_set_rating[user_idx].toarray().flatten()
        else:
            row_data = np.asarray(self.train_set_rating[user_idx]).flatten()
        historyArticles = np.where(row_data == 1)[0]
        filteredItems = np.setdiff1d(filteredItems, historyArticles)
        return filteredItems

    def sampleArticles(self, targetDimensions, targetDistributions, targetSize, linear_program_coefficient):
        candidateItems = []
        if len(self.itemPool) == 0:
            return {}, []

        if (linear_program_coefficient is not None
                and linear_program_coefficient in self.articlesDataframe.columns
                and not self.articlesDataframe[linear_program_coefficient].isna().any()):
            C = np.ones(self.itemPool.shape[0])
            subset = self.articlesDataframe.loc[self.itemPool, linear_program_coefficient].values
            numeric_subset = pd.to_numeric(subset, errors='coerce')
            all_numeric = not np.isnan(numeric_subset).any()
            if all_numeric:
                C = subset
        elif linear_program_coefficient == "rdw_score":
            C = np.asarray(self.articleRdwScore[self.itemPool])
            C = C * -1
        else:
            C = np.ones(self.itemPool.shape[0])

        sampler = DistributionSampler(self.articlesDataframe)
        target_num_items, candidateItems = sampler.sample_by_multi_distributions(
            self.itemPool, targetDimensions, targetDistributions, targetSize, C)
        return target_num_items, candidateItems

    def rankArticles(self, candidateItems, targetSize, rankingType,
                     rankingObjectives=None, mappingList=None, ascending=None):
        rankedArticles = []
        if rankingType == "rdw_score":
            rdwScore = self.articleRdwScore[candidateItems]
            indices = np.argsort(rdwScore)[::-1][:targetSize]
            rankedArticles = candidateItems[indices].tolist()
            scores = rdwScore[indices]
        elif rankingType == "graph_coloring":
            if isinstance(rankingObjectives, list) and len(rankingObjectives) > 0:
                ranking_dim = rankingObjectives[0]
            elif isinstance(rankingObjectives, str):
                ranking_dim = rankingObjectives
            else:
                raise ValueError("For graph_coloring, rankingObjectives must be a valid column name.")
            gc_solver = ItemRanker(list(candidateItems), self.articlesDataframe, ranking_dim)
            rankedArticles = gc_solver.rank()
            rankedArticles = rankedArticles[:targetSize]
            scores = self.articleRdwScore[rankedArticles]
        else:
            rankedArticles = candidateItems[:targetSize].tolist()
            scores = self.articleRdwScore[rankedArticles]
        return rankedArticles, scores

    def newHop(self, user_id, targetDimensions, targetDistributions, targetSize,
               sampleObjective, currentHop, filteringCriteria, given_item_pool=None):
        candidateItems = []
        isEmptyHistory = np.all(self.train_set_rating[user_id, :] == 0) if not hasattr(self.train_set_rating, 'nnz') else (self.train_set_rating[user_id].nnz == 0)
        if isEmptyHistory:
            tarSize = targetSize * random.randint(10, 20)
            poolSize = tarSize if tarSize <= self.articleNum else int(self.articleNum)
            self.itemPool = random.sample(range(0, self.articleNum), poolSize)
            self.articleRdwScore = np.round(
                np.random.random(size=self.articleNum) * (1 - 1e-6) + 1e-6, 3)
        else:
            self.Model_RDW.set_current_user(user_id)
            prob = self.Model_RDW.performMultiHop(currentHop)
            if not isinstance(prob, csr_matrix):
                pass  # _PerUserProxy handles indexing natively
            start_col = self.train_set_rating.shape[0]
            recs = prob[user_id, start_col:]
            recs_dense = recs.toarray().flatten()
            self.articleRdwScore = recs_dense
            self.itemPool = np.nonzero(recs_dense)[0]

        self.itemPool = self.filterHeuristics(
            user_id, self.itemPool, filteringCriteria, given_item_pool=given_item_pool)
        target_num_items, candidateItems = self.sampleArticles(
            targetDimensions, targetDistributions, targetSize, sampleObjective)
        return candidateItems

    def addRandomArticles(self, targetDimensions, targetDistributions, targetSize,
                          sampleObjective, given_item_pool=None):
        sampledItems = []
        for j in range(targetSize - 1, 0, -1):
            target_num_items, sampledItems = self.sampleArticles(
                targetDimensions, targetDistributions, j, sampleObjective)
            if len(sampledItems) == j:
                break
        num_articles_to_add = targetSize - len(sampledItems)
        if not isinstance(given_item_pool, (list, np.ndarray)) or len(given_item_pool) == 0:
            all_articles = range(0, self.articleNum)
        else:
            all_articles = list(given_item_pool)
        remaining_articles = list(set(all_articles) - set(sampledItems))
        if len(remaining_articles) >= num_articles_to_add:
            additional_articles = np.random.choice(
                remaining_articles, num_articles_to_add, replace=False).tolist()
        else:
            additional_articles = remaining_articles
        sampledItems.extend(additional_articles)
        return sampledItems

    def checkListParity(self, candidatesOld, candidatesNew):
        return set(candidatesOld) == set(candidatesNew)

    def performSampling(self, user_id, listSize, targetDimensions, targetDistribution,
                        maxHops, filteringCriteria, sampleObjective, rankingType,
                        rankingObjectives, mappingList, ascending, given_item_pool=None):
        if listSize > self.articleNum:
            listSize = self.articleNum
        candidateItems = []
        self.itemPool = np.array([])
        self.articleRdwScore = np.array([])
        self.CANDIDATESOLD = []
        initialHop = 3
        currentHop = initialHop
        terminateHop = maxHops

        while currentHop <= terminateHop:
            candidateItems = self.newHop(user_id, targetDimensions, targetDistribution,
                                         listSize, sampleObjective, currentHop, filteringCriteria,
                                         given_item_pool=given_item_pool)
            isIdentical = self.checkListParity(candidateItems, self.CANDIDATESOLD)
            if len(candidateItems) >= listSize:
                break
            elif len(self.CANDIDATESOLD) > 0 and isIdentical:
                break
            currentHop = currentHop + 2
            self.CANDIDATESOLD = candidateItems

        if len(candidateItems) == 0:
            candidateItems = self.addRandomArticles(
                targetDimensions, targetDistribution, listSize, sampleObjective,
                given_item_pool=given_item_pool)

        candidateItems = np.array(candidateItems)
        candidateItems, scores = self.rankArticles(
            candidateItems, listSize, rankingType, rankingObjectives, mappingList, ascending)
        return candidateItems, scores
