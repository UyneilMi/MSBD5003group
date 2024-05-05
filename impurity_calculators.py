from pyspark.sql.functions import when, col, sum as sql_sum
import numpy as np
class ImpurityCalculator:
    def __init__(self, data, feature, label_col='label'):
        self.data = data
        self.feature = feature
        self.label_col = label_col
        self.total_count = data.count()

    def calculate_gini(self, subset):
        if subset.count() == 0:
            return 0
        label_counts = subset.groupBy(self.label_col).count().rdd.map(lambda row: row['count']).collect()
        total = sum(label_counts)
        sum_squares = sum((count / total) ** 2 for count in label_counts)
        return 1 - sum_squares

    def split_data(self, value, feature_type='continuous'):
        if feature_type == 'continuous':
            left = self.data.filter(col(self.feature) <= value)
            right = self.data.filter(col(self.feature) > value)
        else:  # for categorical
            left = self.data.filter(col(self.feature).isin(value))
            right = self.data.filter(~col(self.feature).isin(value))
        return left, right

def calculate_impurity_stats(data, feature, split_value, metadata, feature_type='continuous'):
    impurity_calculator = ImpurityCalculator(data, feature)
    left, right = impurity_calculator.split_data(split_value, feature_type)

    left_impurity = impurity_calculator.calculate_gini(left)
    right_impurity = impurity_calculator.calculate_gini(right)

    left_count = left.count()
    right_count = right.count()
    total_count = left_count + right_count

    if left_count < metadata['minInstancesPerNode'] or right_count < metadata['minInstancesPerNode']:
        return None  # Invalid split due to not satisfying the minimum instances constraint

    parent_impurity = impurity_calculator.calculate_gini(data)
    left_weight = left_count / total_count
    right_weight = right_count / total_count

    gain = parent_impurity - (left_weight * left_impurity + right_weight * right_impurity)

    if gain < metadata['minInfoGain']:
        return None  # Invalid split due to not satisfying the minimum information gain

    return {
        'gain': gain,
        'left_impurity': left_impurity,
        'right_impurity': right_impurity,
        'feature': feature,
        'threshold': split_value if feature_type == 'continuous' else None,
        'categories': split_value if feature_type != 'continuous' else None
    }

def split_data(data, feature, value, feature_type):
    """Split data based on feature, value, and type."""
    if feature_type == "continuous":
        left = data.filter(col(feature) <= value)
        right = data.filter(col(feature) > value)
    elif feature_type == "unordered":
        # Assuming value is a list for unordered splits
        left = data.filter(col(feature).isin(value))
        right = data.filter(~col(feature).isin(value))
    else:  # Ordered categorical treated like continuous
        left = data.filter(col(feature) <= value)
        right = data.filter(col(feature) > value)
    return left, right


def entropy(labels):
    """Calculate entropy of given label distribution."""
    probabilities = np.bincount(labels) / len(labels)
    probabilities = probabilities[probabilities > 0]  # Remove zero probabilities to avoid log2(0)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(data, split, label_col):
    """Calculate Information Gain of a split."""
    total_entropy = entropy(data.select(label_col).collect())
    left, right = split_data(data, split.feature, split.threshold, split.feature_type)
    left_entropy = entropy(left.select(label_col).collect())
    right_entropy = entropy(right.select(label_col).collect())
    left_weight = left.count() / data.count()
    right_weight = right.count() / data.count()
    return total_entropy - (left_weight * left_entropy + right_weight * right_entropy)

def gain_ratio(data, split, label_col):
    """Calculate Gain Ratio of a split, which normalizes the Information Gain."""
    ig = information_gain(data, split, label_col)
    # Calculate intrinsic value
    left, right = split_data(data, split.feature, split.threshold, split.feature_type)
    left_prob = left.count() / data.count()
    right_prob = right.count() / data.count()
    intrinsic_value = -left_prob * np.log2(left_prob) - right_prob * np.log2(right_prob)
    return ig / intrinsic_value if intrinsic_value != 0 else 0



# Usage example
# Assuming 'data' is a DataFrame loaded with features and labels
# best_split_info = calculate_impurity_stats(data, 'some_feature', 0.5, metadata)
