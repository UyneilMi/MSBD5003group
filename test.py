import math
import numpy as np
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import col
from collections import Counter
from ucimlrepo import fetch_ucirepo
from tree_based import DecisionTree
from tree_based import RandomForest

if __name__ == "__main__":
    spark = SparkSession.builder.master("local").appName("RandomForestExample").getOrCreate()

    # fetch dataset
    dry_bean = fetch_ucirepo(id=602)

    # convert pandas DataFrame to Spark DataFrame
    features_pd = dry_bean.data.features
    labels_pd = dry_bean.data.targets.rename('label') # Rename target column to 'label'
    data_pd = features_pd.join(labels_pd)

    data = spark.createDataFrame(data_pd)

    features = features_pd.columns.tolist() # Get feature names

    rf = RandomForest(num_trees=10, max_depth=5, min_instances_per_node=2, min_info_gain=0.01,
                      feature_subset_strategy='sqrt')
    rf.train(data, features, label_col='label')

    # Example prediction
    example_features = {'f1': 0.1, 'f2': 0.2, 'f3': 0.3, 'f4': 0.4} # Example feature vector
    prediction = rf.predict(example_features)
    print("Prediction:", prediction)


