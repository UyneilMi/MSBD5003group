# MSBD5003group
This file is the group code for_ HKUST_MSBD_5003_group_. 

### Proposed File Structure

1. **Start with Data Preprocessing:**
   File: `feature_preprocessing.py`
   Use preprocess_features to prepare dataset. such as:
   
   `processed_data = preprocess_features(raw_data, continuous_features, unordered_features, ordered_features)`

3. Build and Train the Decision Tree or RandomForest:
   Files: `decision_tree.py` and `random_forest.py`
   + For Decision Trees: Usage in `decision_tree.py`:
     Create an instance of DecisionTree.
     Call train with the preprocessed data. such as:
     
     `from decision_tree import DecisionTree
     tree = DecisionTree(max_depth=5, min_instances_per_node=10, min_info_gain=0.01)
     tree.train(processed_data, features_list, 'label_col')`
   + For Random Forests: Usage in `random_forest.py`:
     Create an instance of RandomForest.
     Call train similarly to the decision tree. such as:
     
     `from random_forest import RandomForest
     forest = RandomForest(num_trees=10, max_depth=5, min_instances_per_node=10, min_info_gain=0.01)
     forest.train(processed_data, features_list, 'label_col')`

4. Utilize Utility Functions as Needed During Tree Building (Inside DecisionTree/RandomForest Classes):
   File: `tree_units.py` and `split_management.py`
   These contain all necessary utilities like calculating splits, impurity calculations, and applying splits. such as:
   
   `from tree_units import calculate_potential_splits, calculate_split_impurity
   from split_management import apply_split
   splits = calculate_potential_splits(data, feature)`
   
   `for split in splits:
     gain, left_impurity, right_impurity = calculate_split_impurity(data, feature, split, label_col)`

6. Optional Post-Pruning:
   After the tree is built, you might decide to prune it to avoid overfitting.
   Implemented in `decision_tree.py` such as:
   
   `tree.prune_tree(tree.root, min_impurity_decrease=0.01)`
8. Predictions:
   Use the predict method of either DecisionTree or RandomForest depending on your model choice to make predictions on new data. such as:
   
   `predictions = tree.predict(new_data)`





MLlib_baseline.ipynb is a file run on Colab.
