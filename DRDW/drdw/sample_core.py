import numpy as np
import pandas as pd
from scipy.optimize import linprog
import math
from scipy.sparse import csc_matrix
import ast


def processPartyData(test_str):
    if isinstance(test_str, list):
        return test_str
    if pd.isna(test_str):
        return []
    if isinstance(test_str, str):
        try:
            parties = ast.literal_eval(test_str)
            if not isinstance(parties, list):
                return []
        except:
            return []
        return parties
    return []


def is_valid_party_list(x):
    if x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, list) and len(x) == 0):
        return True
    if isinstance(x, list):
        return all(isinstance(i, str) for i in x)
    return False


class DistributionSampler:
    def __init__(self, item_dataframe):
        self.item_dataframe = item_dataframe
        self.target_num_items_per_category = {}

    def _generate_cache_key(self, key_type, feature_dim, target_proportion):
        if key_type == 'discrete':
            tar_key = ','.join([f"{k}:{v}" for k, v in sorted(target_proportion.items())])
            return f"{key_type}:{feature_dim}:{tar_key}"
        elif key_type == 'continuous':
            ranges_key = ','.join([f"{item['min']}-{item['max']}:{item['prob']}" for item in target_proportion])
            return f"{key_type}:{feature_dim}:{ranges_key}"
        return f"{key_type}:{feature_dim}"

    def items_per_discrete_attribute(self, target_proportion, targetSize, feature_dim):
        cache_key = self._generate_cache_key('discrete', feature_dim, target_proportion)
        if cache_key in self.target_num_items_per_category:
            return self.target_num_items_per_category[cache_key]
        for key, value in target_proportion.items():
            if not (0 <= value <= 1):
                raise ValueError(f"Distribution value for '{key}' is not between 0 and 1.")
        if not np.isclose(sum(target_proportion.values()), 1.0, atol=1e-8):
            raise ValueError("Sum of the distribution values must equal 1.")
        items_per_category = {}
        totalSize = 0
        fractional_remainders = []
        for x, y in target_proportion.items():
            fractional_items = y * targetSize
            itemNum = np.floor(fractional_items).astype(int)
            remainder = fractional_items - itemNum
            new_x = feature_dim + ',' + x
            items_per_category[new_x] = itemNum
            totalSize += itemNum
            fractional_remainders.append((new_x, remainder))
        remainder_items_needed = targetSize - totalSize
        if remainder_items_needed > 0:
            fractional_remainders.sort(key=lambda x: x[1], reverse=True)
            for i in range(remainder_items_needed):
                items_per_category[fractional_remainders[i][0]] += 1
        self.target_num_items_per_category[cache_key] = items_per_category
        return items_per_category

    def generateMaskedMatrixDiscrete(self, data, itemPool, targetDimension, items_per_category, cornacId_to_newId):
        if targetDimension not in data.columns:
            raise ValueError(f"Column '{targetDimension}' not found in data.")
        lowered_column = data[targetDimension].astype(str).str.strip().str.lower()
        
        # Build lookup: category_name -> list of newIds (vectorized)
        cat_to_newids = {}
        for item_id, cat_name in zip(data.index, lowered_column):
            if item_id in cornacId_to_newId:
                new_id = cornacId_to_newId[item_id]
                cat_to_newids.setdefault(cat_name, []).append(new_id)
        
        maskedMatrix = {}
        pool_size = itemPool.shape[0]
        for category_key, target_count in items_per_category.items():
            mMatrix = np.zeros(pool_size, dtype=int)
            try:
                category_name = category_key.split(",")[1].strip().lower()
            except IndexError:
                raise ValueError(f"Invalid category_key format: '{category_key}'")
            new_ids = cat_to_newids.get(category_name, [])
            if new_ids:
                mMatrix[new_ids] = 1
            maskedMatrix[category_key] = mMatrix
        return maskedMatrix

    def prepareLinearProgramming(self, df, itemPool, targetDimension, targetDistributions, targetSize):
        originalIndex = np.asarray(itemPool)
        data = df.loc[originalIndex]
        newIndex = np.arange(len(originalIndex))
        newId_to_cornacId = dict(enumerate(originalIndex))
        cornacId_to_newId = dict(zip(originalIndex, newIndex))
        super_dict_matrix = {}
        super_dict_number = {}
        for i in range(len(targetDistributions)):
            targetDistribution = targetDistributions[i]
            if targetDistribution["type"] == "discrete":
                tar = targetDistribution["distr"]
                items_per_category = self.items_per_discrete_attribute(tar, targetSize, targetDimension[i])
                masked_matrix_dict = self.generateMaskedMatrixDiscrete(
                    data, itemPool, targetDimension[i], items_per_category, cornacId_to_newId)
                super_dict_matrix.update(masked_matrix_dict)
                super_dict_number.update(items_per_category)
        return super_dict_matrix, super_dict_number, newId_to_cornacId, cornacId_to_newId

    def sample_by_multi_distributions(self, itemPool, targetDimension, targetDistributions, targetSize,
                                       Objective_to_be_minimized):
        if not isinstance(Objective_to_be_minimized, np.ndarray):
            return {}, []
        if np.ndim(Objective_to_be_minimized) != 1:
            return {}, []
        totalData = self.item_dataframe
        super_dict1, super_dict2, newId_to_cornacId, cornacId_to_newId = self.prepareLinearProgramming(
            totalData, itemPool, targetDimension, targetDistributions, targetSize)
        all_constraints = []
        all_b_value = []
        for key, value in super_dict1.items():
            constraints = value
            b_value = super_dict2[key]
            all_constraints.append(constraints)
            all_b_value.append(b_value)
        all_constraints.append(np.ones(itemPool.shape[0]))
        all_b_value.append(targetSize)
        all_constraints = np.concatenate([all_constraints], axis=0)
        bound = (0, 1)
        A_eq_sparse = csc_matrix(all_constraints)
        try:
            res = linprog(c=Objective_to_be_minimized, A_ub=None, b_ub=None,
                          A_eq=A_eq_sparse, b_eq=all_b_value, bounds=bound, method="highs-ipm")
            if res.success and res.x is not None:
                indices = np.where(res.x == 1)[0]
                cornac_index = [newId_to_cornacId[k] for k in indices.tolist()]
            else:
                cornac_index = []
            return super_dict2, cornac_index
        except ValueError as ve:
            print(f"LP exception: {ve}")
            return {}, []
