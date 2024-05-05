from pyspark.sql.functions import col
from tree_utils import calculate_gini, determine_feature_type

class Split:
    def __init__(self, feature, threshold=None, categories=None):
        self.feature = feature
        self.threshold = threshold  # For continuous and ordered features
        self.categories = categories  # For unordered features

def find_best_split(data, feature, feature_bins, feature_type):
    """
    Find the best split point for a given feature using impurity measures from tree_utils.
    :param data: DataFrame containing the dataset.
    :param feature: String, the name of the feature column in data.
    :param feature_bins: List of bins or splits for the feature.
    :param feature_type: Type of the feature ('continuous', 'ordered', 'unordered').
    :return: Best Split object.
    """
    best_split = None
    best_impurity = float('inf')
    for threshold in feature_bins:
        if feature_type == 'unordered':
            left = data.filter(col(feature).isin(threshold))
            right = data.filter(~col(feature).isin(threshold))
        else:
            left = data.filter(col(feature) <= threshold)
            right = data.filter(col(feature) > threshold)

        left_impurity = calculate_gini(left, feature)
        right_impurity = calculate_gini(right, feature)
        total_impurity = (left.count() * left_impurity + right.count() * right_impurity) / data.count()

        if total_impurity < best_impurity:
            best_impurity = total_impurity
            best_split = Split(feature=feature, threshold=threshold if feature_type != 'unordered' else None, categories=threshold if feature_type == 'unordered' else None)

    return best_split

def apply_split(data, split):
    """
    Apply a Split object to partition the data.
    """
    if split.threshold is not None:
        left = data.filter(col(split.feature) <= split.threshold)
        right = data.filter(col(split.feature) > split.threshold)
    else:
        left = data.filter(col(split.feature).isin(split.categories))
        right = data.filter(~col(split.feature).isin(split.categories))

    return left, right
