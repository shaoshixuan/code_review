from __future__ import print_function
import numpy as np
from collections import OrderedDict, deque, defaultdict
import hashlib


class ItemRanker(object):
    """Graph coloring / round-robin item ranker."""

    cache = {}

    def __init__(self, candidateItems, item_dataframe, dimension, **kwargs):
        self._validate_input(candidateItems, item_dataframe, dimension)
        self.V = len(candidateItems)
        self.dimension = dimension
        self.candidateItems = candidateItems
        self.item_dataframe = item_dataframe.loc[self.candidateItems, dimension]
        category_counts = self.item_dataframe.value_counts()
        self.color_dict = OrderedDict(category_counts.to_dict())
        self.used_color = OrderedDict((key, 0) for key in self.color_dict.keys())
        self.exceeded_max_depth = False
        self.random_walk_score = kwargs.get("random_walk_score", None)

    def _validate_input(self, candidateItems, articleDataframe, dimension):
        if not isinstance(candidateItems, list):
            raise TypeError(f"candidateItems should be a list, but got {type(candidateItems)}.")
        if not isinstance(dimension, str):
            raise TypeError(f"dimension should be a string, but got {type(dimension)}.")
        if dimension not in articleDataframe.columns:
            raise ValueError(f"Dimension '{dimension}' not found in the DataFrame columns.")
        if not set(candidateItems).issubset(articleDataframe.index):
            invalid_items = set(candidateItems) - set(articleDataframe.index)
            raise IndexError(f"The following candidateItems indices are invalid: {invalid_items}.")

    def _generate_cache_key(self):
        key_string = f"{self.candidateItems}-{self.V}-{self.dimension}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def create_color_sequence(self, color):
        result_coloring = list(self.used_color.keys())
        result_coloring = [result_coloring[i] for i in color]
        return result_coloring

    def buildAdjMatrix(self):
        N = self.V
        graph = [[0 for _ in range(N)] for _ in range(N)]
        for i in range(N - 1):
            graph[i][i + 1] = 1
            graph[i + 1][i] = 1
        return graph

    def is_valid_color(self, v, graph, color, c):
        original_color = list(self.used_color.keys())[c]
        if self.used_color[original_color] >= self.color_dict[original_color]:
            return False
        for i in range(self.V):
            if graph[v][i] and c == color[i]:
                return False
        return True

    def graph_coloring(self, graph, m, color, v, recursion_depth=0, max_depth=15):
        if self.exceeded_max_depth:
            return False
        if recursion_depth > max_depth:
            self.exceeded_max_depth = True
            return False
        if v == self.V:
            return True
        for c in range(0, m):
            if self.is_valid_color(v, graph, color, c):
                color[v] = c
                original_color = list(self.used_color.keys())[c]
                self.used_color[original_color] += 1
                if self.graph_coloring(graph, m, color, v + 1, recursion_depth + 1, max_depth):
                    return True
                color[v] = -1
                self.used_color[original_color] -= 1
        return False

    def solve_graph_coloring(self):
        self.exceeded_max_depth = False
        graph = self.buildAdjMatrix()
        color = [-1] * self.V
        m = len(self.color_dict.keys())
        if not self.graph_coloring(graph, m, color, 0):
            return []
        result = self.create_color_sequence(color)
        self.used_color = OrderedDict((key, 0) for key in self.color_dict.keys())
        return result

    def round_robin_rank(self):
        categories = defaultdict(deque)
        for item_id, category in self.item_dataframe.items():
            categories[category].append(item_id)
        if self.random_walk_score is not None:
            for category in categories:
                categories[category] = deque(
                    sorted(categories[category],
                           key=lambda item: self.random_walk_score[item]
                           if 0 <= item < len(self.random_walk_score) else float("-inf"),
                           reverse=True))
        result = []
        category_queues = deque(categories.values())
        while category_queues:
            current_queue = category_queues.popleft()
            if current_queue:
                result.append(current_queue.popleft())
            if current_queue:
                category_queues.append(current_queue)
        return result

    def rank(self):
        cache_key = self._generate_cache_key()
        if cache_key in ItemRanker.cache:
            return ItemRanker.cache[cache_key]
        order_target = self.solve_graph_coloring()
        if len(order_target) == 0:
            ordered_item_ids = self.round_robin_rank()
        else:
            category_to_items = defaultdict(list)
            for item_id, category in self.item_dataframe.items():
                category_to_items[category].append(item_id)
            if self.random_walk_score is not None:
                for category in category_to_items:
                    category_to_items[category].sort(
                        key=lambda item: self.random_walk_score[item]
                        if item < len(self.random_walk_score) else float("-inf"),
                        reverse=True)
            ordered_item_ids = []
            for category in order_target:
                if category in category_to_items and category_to_items[category]:
                    ordered_item_ids.append(category_to_items[category].pop(0))
                else:
                    ordered_item_ids.append(None)
        ItemRanker.cache[cache_key] = ordered_item_ids
        return ordered_item_ids

    @classmethod
    def clear_cache(cls):
        cls.cache = {}
