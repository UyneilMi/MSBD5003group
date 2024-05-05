from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, num_trees=10, max_depth=5, min_instances_per_node=10, min_info_gain=0.01):
        self.trees = [DecisionTree(max_depth, min_instances_per_node, min_info_gain) for _ in range(num_trees)]

    def train(self, data, features, label_col='label'):
        for tree in self.trees:
            sampled_data = data.sample(True, 1.0)  # Bootstrap sampling
            tree.train(sampled_data, features, label_col)

    def predict(self, features):
        predictions = [tree.predict(features) for tree in self.trees]
        return max(set(predictions), key=predictions.count)  # Majority voting
