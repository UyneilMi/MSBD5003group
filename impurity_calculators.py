from pyspark.sql.functions import when, col, sum as sql_sum
import numpy as np

class Split:
    def __init__(self, feature, threshold=None, categories=None):
        self.feature = feature
        self.threshold = threshold  # For continuous and ordered features
        self.categories = categories  # For unordered features

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

    def calculate_entropy(self, subset):
        if subset.count() == 0:
            return 0
        label_counts = subset.groupBy(self.label_col).count().rdd.map(lambda row: row['count']).collect()
        total = sum(label_counts)
        probabilities = [count / total for count in label_counts if count > 0]
        return -sum(p * np.log2(p) for p in probabilities)

    def split_data(self, value, feature_type='continuous'):
        """
        Split data based on feature, value, and type.
        """
        if feature_type == "continuous":
            left = self.data.filter(col(self.feature) <= value)
            right = self.data.filter(col(self.feature) > value)
        else:  # for categorical
            left = self.data.filter(col(self.feature).isin(value))
            right = self.data.filter(~col(self.feature).isin(value))
        return left, right

    def find_best_split(self, feature_bins, feature_type):
        """
        Find the best split point for the initialized feature using impurity measures.
        :param feature_bins: List of bins or splits for the feature.
        :param feature_type: Type of the feature ('continuous', 'ordered', 'unordered').
        :return: Best Split object.
        """
        best_split = None
        best_impurity = float('inf')

        for threshold in feature_bins:
            left, right = self.split_data(threshold, feature_type)

            left_impurity = self.calculate_gini(left)
            right_impurity = self.calculate_gini(right)
            total_impurity = (left.count() * left_impurity + right.count() * right_impurity) / self.total_count

            if total_impurity < best_impurity:
                best_impurity = total_impurity
                best_split = Split(feature=self.feature, threshold=threshold if feature_type != 'unordered' else None,
                                   categories=threshold if feature_type == 'unordered' else None)

        return best_split

    def apply_split(self, split):
        """
        Apply a Split object to partition the data.
        """
        if split.threshold is not None:
            left = self.data.filter(col(split.feature) <= split.threshold)
            right = self.data.filter(col(split.feature) > split.threshold)
        else:
            left = self.data.filter(col(split.feature).isin(split.categories))
            right = self.data.filter(~col(split.feature).isin(split.categories))

        return left, right

    def calculate_impurity_stats(self, feature, split_value, metadata, feature_type='continuous', criterion='gini'):
        """
        Calculate impurity statistics for a split.
        """
        left, right = self.split_data(split_value, feature_type)

        if criterion == 'gini':
            left_impurity = self.calculate_gini(left)
            right_impurity = self.calculate_gini(right)
            parent_impurity = self.calculate_gini(self.data)
        elif criterion == 'entropy':
            left_impurity = self.calculate_entropy(left)
            right_impurity = self.calculate_entropy(right)
            parent_impurity = self.calculate_entropy(self.data)
        else:
            raise ValueError("Unsupported criterion: choose 'gini' or 'entropy'")

        left_count = left.count()
        right_count = right.count()
        total_count = left_count + right_count

        if left_count < metadata['minInstancesPerNode'] or right_count < metadata['minInstancesPerNode']:
            return None  # Invalid split due to not satisfying the minimum instances constraint

        left_weight = left_count / total_count
        right_weight = right_count / total_count

        gain = parent_impurity - (left_weight * left_impurity + right_weight * right_impurity)

        if criterion == 'entropy' and gain > 0:  # For C4.5, calculate gain ratio
            left_prob = left_weight
            right_prob = right_weight
            intrinsic_value = -left_prob * np.log2(left_prob) - right_prob * np.log2(right_prob)
            gain_ratio = gain / intrinsic_value if intrinsic_value != 0 else 0
            if gain_ratio < metadata['minInfoGain']:
                return None  # Invalid split due to not satisfying the minimum gain ratio
            return {
                'gain': gain_ratio,
                'left_impurity': left_impurity,
                'right_impurity': right_impurity,
                'feature': feature,
                'threshold': split_value if feature_type != 'unordered' else None,
                'categories': split_value if feature_type == 'unordered' else None
            }
        elif gain < metadata['minInfoGain']:
            return None  # Invalid split due to not satisfying the minimum information gain

        return {
            'gain': gain,
            'left_impurity': left_impurity,
            'right_impurity': right_impurity,
            'feature': feature,
            'threshold': split_value if feature_type != 'unordered' else None,
            'categories': split_value if feature_type == 'unordered' else None
        }

    def entropy(self, labels):
        probabilities = np.bincount(labels) / len(labels)
        probabilities = probabilities[probabilities > 0]  # Remove zero probabilities to avoid log2(0)
        return -np.sum(probabilities * np.log2(probabilities))

    def information_gain(self, data, split, label_col):
        total_entropy = self.entropy(data.select(label_col).collect())
        left, right = self.split_data(split.threshold, 'continuous' if split.threshold else 'unordered')
        left_entropy = self.entropy(left.select(label_col).collect())
        right_entropy = self.entropy(right.select(label_col).collect())
        left_weight = left.count() / data.count()
        right_weight = right.count() / data.count()
        return total_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def gain_ratio(self, data, split, label_col):
        ig = self.information_gain(data, split, label_col)
        left, right = self.split_data(split.threshold, 'continuous' if split.threshold else 'unordered')
        left_prob = left.count() / data.count()
        right_prob = right.count() / data.count()
        intrinsic_value = -left_prob * np.log2(left_prob) - right_prob * np.log2(right_prob)
        return ig / intrinsic_value if intrinsic_value != 0 else 0





# Usage example
# 使用基尼指数计算分裂增益
# split_info_gini = calculate_impurity_stats(data, 'feature', 30.0, metadata, feature_type='continuous', criterion='gini')
# print("Gini Impurity Split Info:", split_info_gini)

# 使用熵计算信息增益
# split_info_entropy = calculate_impurity_stats(data, 'feature', 30.0, metadata, feature_type='continuous', criterion='entropy')
# print("Entropy Split Info:", split_info_entropy)


# 使用增益率计算分裂增益
# split_info_gain_ratio = calculate_impurity_stats(data, 'feature', 30.0, metadata, feature_type='continuous', criterion='entropy')
# print("Gain Ratio Split Info:", split_info_gain_ratio)

