"""
Standalone D-RDW runner (no Cornac dependency).
"""
import numpy as np
from .sample_and_rank import Sample_And_Rank


class D_RDW:
    """
    Standalone D-RDW recommender.

    Parameters
    ----------
    train_set_rating : np.ndarray
        Binary user-item interaction matrix of shape (n_users, n_items).
    item_dataframe : pd.DataFrame
        DataFrame indexed by item IDs (0-based), with attribute columns.
    diversity_dimension : list of str
        Column names in item_dataframe to diversify over.
    target_distributions : dict
        Target distributions per dimension (discrete/continuous).
    targetSize : int
        Recommendation list length.
    maxHops : int
        Maximum random walk hops (>= 3).
    rankingType : str
        'rdw_score' | 'graph_coloring' | 'multi_objectives'
    rankingObjectives : list, optional
    sampleObjective : str
    filteringCriteria : dict, optional
    """

    def __init__(self, train_set_rating, item_dataframe,
                 diversity_dimension=None, target_distributions=None,
                 targetSize=20, maxHops=15, rankingType="rdw_score",
                 rankingObjectives=None, mappingList=None, ascending=None,
                 sampleObjective="rdw_score", filteringCriteria=None):
        self.train_set_rating = train_set_rating
        self.item_dataframe = item_dataframe
        self.diversity_dimension = diversity_dimension or []
        self.targetDistribution = target_distributions or {}
        self.targetSize = targetSize
        if maxHops < 3:
            raise ValueError(f"maxHops must be >= 3, got {maxHops}")
        self.maxHops = maxHops
        self.rankingType = rankingType
        self.rankingObjectives = rankingObjectives
        self.mappingList = mappingList
        self.ascending = ascending
        self.sampleObjective = sampleObjective
        self.filteringCriteria = filteringCriteria
        self.sampleRank = Sample_And_Rank(train_set_rating, item_dataframe)

    def rank_for_user(self, user_idx, given_item_pool=None):
        """
        Generate ranked recommendations for a single user.

        Parameters
        ----------
        user_idx : int
        given_item_pool : list of int, optional
            If provided, restrict ranking to this candidate pool.

        Returns
        -------
        ranked_items : list of int
        scores : np.ndarray
        """
        selectedTarget = [self.targetDistribution[d] for d in self.diversity_dimension]

        ranked_items, scores = self.sampleRank.performSampling(
            user_id=user_idx,
            listSize=self.targetSize,
            targetDimensions=self.diversity_dimension,
            targetDistribution=selectedTarget,
            maxHops=self.maxHops,
            filteringCriteria=self.filteringCriteria,
            sampleObjective=self.sampleObjective,
            rankingType=self.rankingType,
            rankingObjectives=self.rankingObjectives,
            mappingList=self.mappingList,
            ascending=self.ascending,
            given_item_pool=given_item_pool
        )
        return ranked_items, scores
