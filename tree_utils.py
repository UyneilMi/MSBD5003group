import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, FloatType, DoubleType


def calculate_potential_splits(self, data, feature):
    feature_type = self.determine_feature_type(data, feature)
    if feature_type == "continuous":
        # Get distinct values sorted, then find mid-points
        distinct_values = data.select(feature).distinct().orderBy(feature).rdd.map(lambda row: row[feature]).collect()
        return [(distinct_values[i] + distinct_values[i + 1]) / 2 for i in range(len(distinct_values) - 1)]
    elif feature_type == "unordered":
        # For unordered categorical data, each distinct value is a potential split
        return data.select(feature).distinct().rdd.map(lambda row: row[feature]).collect()
    else:  # Ordered categorical is treated like continuous
        distinct_values = data.select(feature).distinct().orderBy(feature).rdd.map(lambda row: row[feature]).collect()
        return [(distinct_values[i] + distinct_values[i + 1]) / 2 for i in range(len(distinct_values) - 1)]

def determine_feature_type(self, data, feature):
    # This is a placeholder. Ideally, you'd have metadata about each feature type.
    # You might infer it based on the data type or explicitly define it somewhere.
    if isinstance(data.schema[feature].dataType, (IntegerType, FloatType, DoubleType)):
        return "continuous"
    else:
        return "unordered"  # or "ordered" based on your data understanding


def calculate_split_impurity(self, data, feature, value, label_col):
    feature_type = self.determine_feature_type(data, feature)
    left, right = self.split_data(data, feature, value, feature_type)
    left_impurity = self.calculate_gini(left, label_col)
    right_impurity = self.calculate_gini(right, label_col)

    left_count = left.count()
    right_count = right.count()
    total_count = left_count + right_count

    parent_impurity = self.calculate_gini(data, label_col)
    left_weight = left_count / total_count
    right_weight = right_count / total_count

    gain = parent_impurity - (left_weight * left_impurity + right_weight * right_impurity)
    return gain, left_impurity, right_impurity

def split_data(self, data, feature, value, feature_type):
    if feature_type == "continuous":
        left = data.filter(col(feature) <= value)
        right = data.filter(col(feature) > value)
    elif feature_type == "unordered":
        # This assumes value is a list for unordered splits
        left = data.filter(col(feature).isin(value))
        right = data.filter(~col(feature).isin(value))
    else:  # Ordered categorical treated like continuous
        left = data.filter(col(feature) <= value)
        right = data.filter(col(feature) > value)
    return left, right

def calculate_gini(self, subset, label_col):
    if subset.count() == 0:
        return 0
    label_counts = subset.groupBy(label_col).count().rdd.map(lambda row: row['count']).collect()
    total = sum(label_counts)
    sum_squares = sum((count / total) ** 2 for count in label_counts)
    return 1 - sum_squares
