import random
import numpy as np
import math
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import col
from impurity_calculators import ImpurityCalculator
from feature_preprocessing import preprocess_features
from collections import Counter

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction

    def predict(self, features):
        if self.feature is None:
            return self.prediction
        elif features[self.feature] <= self.threshold:
            return self.left.predict(features)
        else:
            return self.right.predict(features)

class DecisionTree:
    def __init__(self, max_depth=5, min_instances_per_node=10, min_info_gain=0.01):
        self.root = None
        self.max_depth = max_depth
        self.min_instances_per_node = min_instances_per_node
        self.min_info_gain = min_info_gain

    def train(self, data, features, label_col='label'):
        data = preprocess_features(data, features)
        self.root = self.build_tree(data, features, label_col, depth=0)

    def build_tree(self, data, features, label_col, depth):
        if depth >= self.max_depth or data.count() <= self.min_instances_per_node:
            prediction = data.groupBy(label_col).count().orderBy('count', ascending=False).first()[label_col]
            return TreeNode(prediction=prediction)

        best_split = self.find_best_split(data, features, label_col)
        if best_split is None:
            prediction = data.groupBy(label_col).count().orderBy('count', ascending=False).first()[label_col]
            return TreeNode(prediction=prediction)

        impurity_calculator = ImpurityCalculator(data, best_split['feature'], label_col)
        left_data, right_data = impurity_calculator.apply_split(best_split)
        left_child = self.build_tree(left_data, features, label_col, depth + 1)
        right_child = self.build_tree(right_data, features, label_col, depth + 1)

        return TreeNode(feature=best_split['feature'], threshold=best_split['threshold'], left=left_child, right=right_child)

    def find_best_split(self, data, features, label_col):
        best_gain = 0
        best_split = None
        for feature in features:
            impurity_calculator = ImpurityCalculator(data, feature, label_col)
            feature_bins = self.calculate_potential_splits(data, feature)
            for split in feature_bins:
                gain, left_impurity, right_impurity = impurity_calculator.calculate_split_impurity(data, feature, split, label_col)
                if gain > best_gain and gain > self.min_info_gain:
                    best_gain = gain
                    best_split = {'feature': feature, 'threshold': split}
        return best_split

    def calculate_potential_splits(self, data, feature):
        thresholds = data.select(feature).distinct().rdd.flatMap(lambda x: x).collect()
        return thresholds

    def prune_tree(self, node, min_impurity_decrease):
        if node is None or node.is_leaf():
            return
        if node.left:
            if self.calculate_impurity_decrease(node.left) < min_impurity_decrease:
                node.left = None  # Prune
            else:
                self.prune_tree(node.left, min_impurity_decrease)
        if node.right:
            if self.calculate_impurity_decrease(node.right) < min_impurity_decrease:
                node.right = None  # Prune
            else:
                self.prune_tree(node.right, min_impurity_decrease)

    def calculate_impurity_decrease(self, node):
        return 0  # Implement based on your tree's impurity calculation logic


class RandomForest:
    def __init__(self, num_trees=10, max_depth=5, min_instances_per_node=10, min_info_gain=0.01, subsample_size=0.8,
                 feature_subset_strategy='auto'):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_instances_per_node = min_instances_per_node
        self.min_info_gain = min_info_gain
        self.subsample_size = subsample_size
        self.feature_subset_strategy = feature_subset_strategy
        self.trees = []

    def _select_features(self, num_features):
        if self.feature_subset_strategy == 'auto' or self.feature_subset_strategy == 'sqrt':
            num_to_select = int(math.sqrt(num_features))
        elif self.feature_subset_strategy == 'log2':
            num_to_select = int(math.log2(num_features))
        elif isinstance(self.feature_subset_strategy, int):
            num_to_select = self.feature_subset_strategy
        else:
            num_to_select = num_features

        return np.random.choice(range(num_features), num_to_select, replace=False)

    def train(self, data, features, label_col='label'):
        num_features = len(features)
        tree_configs = [(self.max_depth, self.min_instances_per_node, self.min_info_gain) for _ in
                        range(self.num_trees)]

        def train_tree(config):
            max_depth, min_instances_per_node, min_info_gain = config
            tree = DecisionTree(max_depth, min_instances_per_node, min_info_gain)
            subsample = data.sample(withReplacement=True, fraction=self.subsample_size)
            selected_features = [features[i] for i in self._select_features(num_features)]
            tree.train(subsample, selected_features, label_col)
            return tree

        self.trees = SparkContext.getOrCreate().parallelize(tree_configs).map(train_tree).collect()

    def predict(self, features):
        predictions = [tree.root.predict(features) for tree in self.trees]
        return Counter(predictions).most_common(1)[0][0]


# 示例使用
if __name__ == "__main__":
    spark = SparkSession.builder.master("local").appName("RandomForestExample").getOrCreate()
    data = spark.createDataFrame([
        (1, 10.0, "A"),
        (2, 20.0, "B"),
        (3, 30.0, "A"),
        (4, 40.0, "B"),
        (5, 50.0, "A")
    ], ["id", "feature", "label"])

    features = ['feature']
    rf = RandomForest(num_trees=10, max_depth=5, min_instances_per_node=2, min_info_gain=0.01, feature_subset_strategy='sqrt')
    rf.train(data, features, label_col='label')

    prediction = rf.predict({'feature': 25.0})
    print("Prediction:", prediction)
